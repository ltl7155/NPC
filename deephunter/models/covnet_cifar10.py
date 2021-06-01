import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init

import torch 

####
'''
 any model with or without the nn.BatchNorm2d will have big difference
'''
###


class ConvnetCifar(nn.Module):
    def __init__(self,num_classes=10):
        super(ConvnetCifar, self).__init__()
        self.conv1=nn.Conv2d(3, 32, kernel_size=3,padding=1)
        self.relu1=nn.ReLU(inplace=True)
             #nn.BatchNorm2d(32),
        self.conv2=nn.Conv2d(32, 32, kernel_size=3,padding=1)
        self.relu2=nn.ReLU(inplace=True)
             #nn.BatchNorm2d(32),
        self.pool1= nn.MaxPool2d(2)
        self.conv3=nn.Conv2d(32, 64, kernel_size=3,padding=1)
        self.relu3=nn.ReLU(inplace=True)
             #nn.BatchNorm2d(64),

        self.conv4=nn.Conv2d(64, 64, kernel_size=3,padding=1)
        self.relu4=nn.ReLU(inplace=True)
             #nn.BatchNorm2d(64),

        self.pool2= nn.MaxPool2d(2)
        self.conv5=nn.Conv2d(64, 128, kernel_size=3,padding=1)
        self.relu5=nn.ReLU(inplace=True)
             #nn.BatchNorm2d(128),

        self.conv6=nn.Conv2d(128, 128, kernel_size=3,padding=1)
        self.relu6=nn.ReLU(inplace=True)
             #nn.BatchNorm2d(128),
        self.pool3= nn.MaxPool2d(2)


        self.drop=  nn.Dropout()
            #nn.Dropout(),
        self.dens1=nn.Linear(128 * 4 * 4, 1024)
        self.relu7=nn.ReLU(inplace=True)
            #nn.Dropout(),
        self.dens2=nn.Linear(1024, 512)
        self.relu8=nn.ReLU(inplace=True)
        self.dens3=nn.Linear(512, num_classes)

    def forward(self, x,need_softmax=True):
        x=self.conv1(x)
        x=self.relu1(x)
        x=self.conv2(x)
        x=self.relu2(x)
        x =self.pool1(x)

        x=self.conv3(x)
        x=self.relu3(x)
        x=self.conv4(x)
        x=self.relu4(x)
        x =self.pool2(x)


        x=self.conv5(x)
        x=self.relu5(x)
        x=self.conv6(x)
        x=self.relu6(x)
        x =self.pool3(x)

        x1_flatt=x.view(x.shape[0], -1)

        x_out=self.drop(x1_flatt)
        x_out= self.dens1(x_out)
        x_out= self.relu7(x_out)

        x_out=self.drop(x_out)
        x_out= self.dens2(x_out)
        x_out= self.relu8(x_out)

        x_out= self.dens3(x_out)

        if need_softmax:
            return F.log_softmax(x_out, dim=-1)
        return x_out

    def load_state_dict(self,state_dict,**kwargs):
        if "net"  in state_dict:
            state_dict = state_dict["net"]
        return super(ConvnetCifar,self).load_state_dict(state_dict=state_dict,**kwargs)


if __name__=="__main__":
    x=torch.randn(4,3,32,32)
    
    net = ConvnetCifarBN()
    net.apply(weight_init)
    y=net(x)
    
    