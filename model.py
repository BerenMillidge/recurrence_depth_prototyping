# Prednet model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# Feedforward module
class FFconv2d(nn.Module):
    def __init__(self, inchan, outchan, downsample=False):
        super().__init__()
        self.conv2d = nn.Conv2d(inchan, outchan, kernel_size=3, stride=1, padding=1, bias=False)
        self.downsample = downsample
        if self.downsample:
            self.Downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv2d(x)
        if self.downsample:
            x = self.Downsample(x)
        return x


# Feedback module
class FBconv2d(nn.Module):
    def __init__(self, inchan, outchan, upsample=False):
        super().__init__()
        self.convtranspose2d = nn.ConvTranspose2d(inchan, outchan, kernel_size=3, stride=1, padding=1, bias=False)
        self.upsample = upsample
        if self.upsample:
            self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        if self.upsample:
            x = self.Upsample(x)
        x = self.convtranspose2d(x)
        return x


# FFconv2d and FBconv2d that share weights
class Conv2d(nn.Module):
    def __init__(self, inchan, outchan, sample=False):
        super().__init__()
        self.kernel_size = 3
        self.weights = nn.init.xavier_normal(torch.Tensor(outchan,inchan,self.kernel_size,self.kernel_size))
        self.weights = nn.Parameter(self.weights, requires_grad=True) # ah, okay, this makes perfect sense now. How big is this model?
        self.sample = sample
        if self.sample:
            self.Downsample = nn.MaxPool2d(kernel_size=2, stride=2)
            self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x, feedforward=True):
        if feedforward:
            x = F.conv2d(x, self.weights, stride=1, padding=1)
            if self.sample:
                x = self.Downsample(x)
        else:
            if self.sample:
                x = self.Upsample(x)
            x = F.conv_transpose2d(x, self.weights, stride=1, padding=1)
        return x


def compute_model_size(block_num,sizes):
        ics = []
        ocs = []
        sps = []
        L = len(sizes)
        for i,size in enumerate(sizes):
          if i == 0:
            ics.append(sizes[i])
            ocs.append(sizes[i+1])
            sps.append(False)
          else:
            for n in range(block_num):
              if n == block_num-1 and i != L -1:
                ics.append(sizes[i])
                ocs.append(sizes[i+1])  
              else:           
                ics.append(sizes[i])
                ocs.append(sizes[i])
              if n == 0 and i >1:
                sps.append(True)
              else:
                sps.append(False)
        return ics, ocs, sps


# PredNet
class PredNet(nn.Module):

    def __init__(self, num_classes=10, cls=100, num_blocks=1, use_rate_params=True):
        super().__init__()
        sizes = [3,32,64,128]
        self.use_rate_params = use_rate_params
        ics,ocs,sps = compute_model_size(num_blocks,sizes)
        print(ics, ocs, sps)
        self.cls = cls # num of circles
        print("CLS: ", cls)
        assert len(ics) == len(ocs), 'Input and output channels must be same length'
        self.nlays = len(ics) #number of layers

        # Feedforward layers
        self.FFconv = nn.ModuleList([FFconv2d(ics[i],ocs[i],downsample=sps[i]) for i in range(self.nlays)])
        # Feedback layers
        if cls > 0:
            self.FBconv = nn.ModuleList([FBconv2d(ocs[i],ics[i],upsample=sps[i]) for i in range(self.nlays)])

        # Update rate
        if self.use_rate_params:
          self.a0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ics[i],1,1)+0.5) for i in range(1,self.nlays)])
          self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ocs[i],1,1)+1.0) for i in range(self.nlays)])
        else:
          self.a0 = [torch.zeros(1,ics[i],1,1)+0.5 for i in range(1,self.nlays)]
          self.b0 = [torch.zeros(1,ocs[i],1,1)+1.0 for i in range(self.nlays)]

        # Linear layer
        self.linear = nn.Linear(ocs[-1], num_classes)

    def forward(self, x):

        # Feedforward
        xr = [F.relu(self.FFconv[0](x))]
        for i in range(1,self.nlays):            
            xr.append(F.relu(self.FFconv[i](xr[i-1])))     

        #print("Feedforward complete")     

        # Dynamic process 
        for t in range(self.cls):
            print("cls ", self.cls)
            print("Feedback, ", t)
          # Feedback prediction
            xp = []
            for i in range(self.nlays-1,0,-1):
                #print('xp len ', len(xp))
                xp = [self.FBconv[i](xr[i])] + xp
                a0 = F.relu(self.a0[i-1]).expand_as(xr[i-1])
                xr[i-1] = F.relu(xp[0]*a0 + xr[i-1]*(1-a0))

            # Feedforward prediction error
            b0 = F.relu(self.b0[0]).expand_as(xr[0])
            xr[0] = F.relu(self.FFconv[0](x-self.FBconv[0](xr[0]))*b0 + xr[0])
            for i in range(1, self.nlays):
                b0 = F.relu(self.b0[i]).expand_as(xr[i])
                xr[i] = F.relu(self.FFconv[i](xr[i-1]-xp[i-1])*b0 + xr[i])

        # classifier                
        out = F.avg_pool2d(xr[-1], xr[-1].size(-1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

    def save_model(self,logdir, epoch):
        state = {"state_dict": self.state_dict(), "epoch": epoch}
        torch.save(state, logdir)



# PredNet
class PredNetTied(nn.Module):
    def __init__(self, num_classes=10, cls=3,num_blocks=3, use_rate_params=True):
        super().__init__()
        sizes = [3,32,64,128]
        self.use_rate_params = use_rate_params
        ics,ocs,sps = compute_model_size(num_blocks,sizes)
        print(ics, ocs, sps)
        self.cls = cls # num of circles
        print("CLS: ", cls)
        assert len(ics) == len(ocs), 'Input and output channels must be same length'
        self.nlays = len(ics) #number of layers

        # Convolutional layers
        self.conv = nn.ModuleList([Conv2d(ics[i],ocs[i],sample=sps[i]) for i in range(self.nlays)])

        # Update rate
        if self.use_rate_params:
          self.a0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ics[i],1,1)+0.5) for i in range(1,self.nlays)])
          self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ocs[i],1,1)+1.0) for i in range(self.nlays)])
        else:
          self.a0 = [torch.zeros(1,ics[i],1,1)+0.5 for i in range(1,self.nlays)]
          self.b0 = [torch.zeros(1,ocs[i],1,1)+1.0 for i in range(self.nlays)]

        # Linear layer
        self.linear = nn.Linear(ocs[-1], num_classes)

    def forward(self, x):

        # Feedforward
        xr = [F.relu(self.conv[0](x))]        
        for i in range(1,self.nlays):
            xr.append(F.relu(self.conv[i](xr[i-1])))     

        # Dynamic process 
        for t in range(self.cls):

            # Feedback prediction
            xp = []
            for i in range(self.nlays-1,0,-1):
                xp = [self.conv[i](xr[i],feedforward=False)] + xp
                a = F.relu(self.a0[i-1]).expand_as(xr[i-1])
                xr[i-1] = F.relu(xp[0]*a + xr[i-1]*(1-a))

            # Feedforward prediction error
            b = F.relu(self.b0[0]).expand_as(xr[0])
            xr[0] = F.relu(self.conv[0](x - self.conv[0](xr[0],feedforward=False))*b + xr[0])
            for i in range(1, self.nlays):
                b = F.relu(self.b0[i]).expand_as(xr[i])
                xr[i] = F.relu(self.conv[i](xr[i-1]-xp[i-1])*b + xr[i])  

        # classifier                
        out = F.avg_pool2d(xr[-1], xr[-1].size(-1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def save_model(self,logdir, epoch):
        state = {"state_dict": self.state_dict(), "epoch": epoch}
        torch.save(state, logdir)

