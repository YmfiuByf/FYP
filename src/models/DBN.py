import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from RBM import RBM
from tqdm import trange


class DBN(nn.Module):
    def __init__(self,
             v_units,
             h_units,    # len = num_layer, elem = num_hidden_units
             gib_step = 1,
             lr = 1e-5,
             lr_decay = False,
             decay_rate = 0.3,
             use_gpu = True,
             xavier_init = False
             ):
        super(DBN, self).__init__()

        self.n_layers = len(h_units)
        self.rbm_layers = []
        self.rbm_nodes = []

        for i in range(self.n_layers):
            input_size = v_units
            if i == 0:
                input_size = v_units
            else:
                input_size = h_units[i-1]
            self.rbm_layers.append( RBM(input_size, h_units[i],lr_decay=lr_decay,lr=lr,decay_rate=decay_rate,gib_step=gib_step,use_gpu=use_gpu,xavier_init=xavier_init) )

        self.W_rec = [nn.Parameter(self.rbm_layers[i].W.data.clone()) for i in range(self.n_layers - 1)]
        self.W_gen = [nn.Parameter(self.rbm_layers[i].W.data) for i in range(self.n_layers - 1)]
        self.bias_rec = [nn.Parameter(self.rbm_layers[i].h_bias.data.clone()) for i in range(self.n_layers - 1)]
        self.bias_gen = [nn.Parameter(self.rbm_layers[i].v_bias.data) for i in range(self.n_layers - 1)]
        self.W_mem = nn.Parameter(self.rbm_layers[-1].W.data)
        self.v_bias_mem = nn.Parameter(self.rbm_layers[-1].v_bias.data)
        self.h_bias_mem = nn.Parameter(self.rbm_layers[-1].h_bias.data)


    def forward(self, input):
        h = input    # v may have dim higher than 2, so flatten first
        h_prob = 0
        for i in range(self.n_layers):
            h = h.view(len(h), -1).type(torch.FloatTensor)
            h_prob, h = self.rbm_layers[i].v2h(h)
        return h_prob, h


    def reconstruct(self, input):
        h_prob, h = self.forward(input)
        v = h
        v_prob = 0
        for i in range(self.n_layers-1, -1, -1):
            v = v.view(len(v), -1).type(torch.FloatTensor)
            v_prob, v = self.rbm_layers[i].h2v(v)
        error = torch.mean( (input - v_prob) ** 2, dim=0)
        return error, v_prob, v


    def train(self, train_data, train_labels, num_epochs = 50, batch_size = 16):
        ''' train layer by layer'''
        tmp = train_data
        for i in trange(self.n_layers):
            x = tmp.type(torch.FloatTensor)
            y = train_labels.type(torch.FloatTensor)
            dataset = torch.utils.data.TensorDataset(x, y)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True)

            self.rbm_layers[i].train(dataloader, num_epochs = num_epochs, batch_size=batch_size)
            v = tmp.view((tmp.shape[0], -1)).type(torch.FloatTensor)  # flatten
            p_v, v = self.rbm_layers[i].forward(v)
            tmp = v
        return

    def train_ith(self, train_data, train_labels, i_th, num_epochs=50, batch_size=16):
        x = train_data.type(torch.FloatTensor)
        y = train_labels.type(torch.FloatTensor)
        dataset = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True)

        self.rbm_layers[i_th].train(dataloader, num_epochs=num_epochs, batch_size=batch_size)
        #note that num of layer starts at 0
        return
