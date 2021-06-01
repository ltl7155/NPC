import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init

import torch 

####
'''
 any model with or without the nn.BatchNorm2d will have big difference

'''
###


class ConvnetMnist(nn.Module):
    def __init__(self,num_classes=10):
        super(ConvnetMnist, self).__init__()
        self.conv1=nn.Conv2d(1, 64, kernel_size=3,padding=0)
        self.relu1=nn.ReLU(inplace=True)
             #nn.BatchNorm2d(64),
        self.conv2=nn.Conv2d(64, 64, kernel_size=3,padding=0)
        self.relu2=nn.ReLU(inplace=True)
             #nn.BatchNorm2d(64),
        self.pool1=nn.MaxPool2d(2)
        self.drop=  nn.Dropout()
        self.dens1 = nn.Linear(64 * 12 * 12, 128)
        self.relu3 = nn.ReLU(inplace=True)
        self.dens2 = nn.Linear(128, num_classes)


    def forward(self, x, need_softmax=False):
        x=self.conv1(x)
        x=self.relu1(x)
        x=self.conv2(x)
        x=self.relu2(x)
        x1 = self.pool1(x)
#         print(x1)

        x1_flatt=x1.view(x1.shape[0], -1)
        x1_flatt=self.drop(x1_flatt)

        x_out= self.dens1(x1_flatt)
        x_out= self.relu3(x_out)
        x_out= self.dens2(x_out)
        if need_softmax:
            return F.log_softmax(x_out, dim=-1)

        return x_out
    def load_state_dict(self,state_dict,**kwargs):
        if "net"  in state_dict:
            state_dict = state_dict["net"]
        return super(ConvnetMnist,self).load_state_dict(state_dict=state_dict,**kwargs)


if __name__=="__main__":
    x=torch.randn(4,3,32,32)
    
    x=torch.randn(4,1,28,28)
    net = ConvnetMnist()
    net.apply(weight_init)
    y=net(x)
    
