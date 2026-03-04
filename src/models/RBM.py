import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm,trange
import sys


class RBM(nn.Module):

    def __init__(self,
                 num_v,
                 num_h,
                 gib_step=1,
                 lr=1e-5,
                 lr_decay=False,
                 decay_rate = 0.3,
                 use_gpu=True,
                 xavier_init=False
                 ):

        super(RBM, self).__init__()

        self.num_v = num_v
        self.num_h = num_h
        self.gib_step = gib_step
        self.lr = lr
        self.lr_decay = lr_decay
        self.decay_rate = decay_rate
        self.W_grad_squared = 0
        self.h_grad_squared = 0
        self.v_grad_squared = 0
        self.W_grad = 0
        self.v_grad = 0
        self.h_grad = 0
        self.use_gpu = use_gpu
        self.xavier_init = xavier_init
        device = torch.device('cuda' if torch.cuda.is_available() else 'gpu')
        if xavier_init :
            self.xavier_value = torch.sqrt(torch.FloatTensor([6 / (self.num_v + self.num_h)]))
            self.W = - self.xavier_value + 2*self.xavier_value * torch.rand(self.num_v, self.num_h)
        else:
            self.W = torch.randn(self.num_v, self.num_h) * 0.01
        self.h_bias = torch.zeros(self.num_h).to(device)
        self.v_bias = torch.zeros(self.num_v).to(device)
        self.W = self.W.to(device)

    def v2h(self, v):
        '''
        :param v: given v or v_prob    v=[batch, nv],  W=[nv, nh]
        :return:  (hidden_prob, hidden_sample)
        '''
        h_prob = torch.matmul(v, self.W) + self.h_bias
        h_prob = torch.sigmoid(h_prob)
        h = self.sampling(h_prob)
        return h_prob, h

    def forward(self,input):
        return self.v2h(input)

    def h2v(self, h):
        v_prob = torch.matmul(h, (self.W).T) + self.v_bias
        v_prob = torch.sigmoid(v_prob)
        v = self.sampling(v_prob)
        return v_prob, v

    def sampling(self, prob):
        s = torch.distributions.Bernoulli(prob).sample()
        return s

    def reconstruct(self, v0, gib_step):
        v = v0
        for i in range(gib_step):
            h_prob, h = self.v2h(v)
            v_prob, v = self.h2v(h)
        error = torch.mean((v0 - v_prob) ** 2, dim=0)
        return error, v_prob, v

    def cd(self, input, gib_step, lr=1e-5,training= True):
        '''
        :param input:  given v = [batch, nv]
        :param gib_step: num of gibbs sampling steps
        :return: error, grad for (W, v_bias, h_bias)
        '''
        batch_size = len(input)
        v0 = input
        h0_prob, h0 = self.v2h(v0)
        h_prob = h0_prob
        h = h0

        for i in range(gib_step):
            v_prob , v = self.h2v(h)
            h_prob , h = self.v2h(v_prob)

        W_grad = torch.matmul(v0.t(), h0) - torch.matmul(v_prob.t(), h_prob)
        W_grad /= batch_size
        v_grad = torch.sum(v0 - v_prob, dim=0)/batch_size
        h_grad = torch.sum(h0 - h_prob, dim=0)/batch_size

        if training == True:
            #RMSProp update algorithm:
            self.W_grad_squared += self.decay_rate * self.W_grad_squared + (1 - self.decay_rate) * W_grad * W_grad
            self.h_grad_squared += self.decay_rate * self.h_grad_squared + (1 - self.decay_rate) * h_grad * h_grad
            self.v_grad_squared += self.decay_rate * self.v_grad_squared + (1 - self.decay_rate) * v_grad * v_grad

            self.W += lr * W_grad / ( torch.sqrt(self.W_grad_squared) + 1e-7 )
            self.v_bias += lr * v_grad / ( torch.sqrt(self.v_grad_squared) + 1e-7 )
            self.h_bias += lr * h_grad / ( torch.sqrt(self.h_grad_squared) + 1e-7 )

        error = torch.mean((v0 - v_prob)**2, dim=0)
        return error

    def step(self, input_data, epoch, num_epochs):
        '''
            Includes the foward prop plus the gradient descent
            Use this for training
        '''
        # if self.increase_to_cd_k:
        #     n_gibbs_sampling_steps = int(math.ceil((epoch / num_epochs) * self.k))
        # else:
        #     n_gibbs_sampling_steps = self.k

        if self.lr_decay:
            lr = self.lr / epoch
        else:
            lr = self.lr
        return self.cd(input_data, self.gib_step, lr, training=True)


    def train(self, train_dataloader , num_epochs = 50, batch_size=16):

        self.batch_size = batch_size
        if (isinstance(train_dataloader, torch.utils.data.DataLoader)):
            train_loader = train_dataloader
        else:
            train_loader = torch.utils.data.DataLoader(train_dataloader, batch_size=batch_size)

        for epoch in trange(1, num_epochs + 1):
            epoch_err = 0.0
            n_batches = int(len(train_loader))
            # print(n_batches)

            cost_ = torch.FloatTensor(n_batches, 1)
            grad_ = torch.FloatTensor(n_batches, 1)

            # for i, (batch, _) in tqdm(enumerate(train_loader), ascii=True,
            #                           desc="RBM fitting", file=sys.stdout):
            for i, (batch, _) in enumerate(train_loader):

                batch = batch.view(len(batch), self.num_v)

                if (self.use_gpu):
                    batch = batch.cuda()
                # cost_[i - 1]
                    __ = self.step(batch, epoch, num_epochs)

            # print("Epoch:{} ,avg_cost = {} ,std_cost = {} ,avg_grad = {} ,std_grad = {}".format(epoch, \
            #                                                                                     torch.mean(cost_), \
            #                                                                                     torch.std(cost_)) )# , \
                                                                                                # torch.mean(grad_), \
                                                                                                # torch.std(grad_)))

        return


