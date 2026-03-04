from torch import nn, Tensor
import torch

device = 'cuda'

class Conv_Block(nn.Module):
    def __init__(self,m=3,t=3):
        super(Conv_Block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=[m,t],stride=[m//2,t//2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=[m,t],stride=[1,1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=[m, t], stride=[m // 2, t // 2]),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=[m,t],stride=[1,1]),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=t, stride=t//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=t, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=t, stride=t//2),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=t, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5,inplace=False),
        )
    def forward(self,x,f):
        return self.block(x),self.block2(f)

class CNN_attention(nn.Module):
    def __init__(self, ms=[3,5],ts=[3,5],output_size = 6):
        super().__init__()
        self.convs = nn.ModuleList()
        for m in ms:
            for t in ts:
                self.convs.append(Conv_Block(m=m,t=t))
        # self.convs.append(Conv_Block(3,3))
        # self.convs.append(Conv_Block(5,5))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(len(self.convs)*256,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024,output_size)
        )
        self.attentions = nn.ModuleList()
        for _ in range(len(self.convs)):
            self.attentions.append(nn.Linear(256,1))
        self.attention_f = nn.Linear(256,1)

        # print(self.convs,self.linears)
    def forward(self, x,f):
        out = torch.tensor([]).to(device)
        for conv,attention in zip(self.convs,self.attentions):
            x_, f_ = conv(x,f)
            att = self.attention_f(f_.transpose(-1,-2))
            att = torch.softmax(att,dim=-2).unsqueeze(1)
            x_ = torch.matmul(x_,att).squeeze(-1)
            # x_ = torch.flatten(x_,start_dim=1)
            att = attention(x_.transpose(-1,-2))
            att = torch.softmax(att, dim=-2)
            x_ = torch.matmul(x_,att).squeeze(-1)
            out = torch.concat([out,x_],dim=-1)
        out = self.classifier(out)
        # print(out.shape)
        return out