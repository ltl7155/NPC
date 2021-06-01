import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init

import torch 

####
'''
 any model with or without the nn.BatchNorm2d will have big difference

'''
###

class mask_ConvnetMnist(nn.Module):
    def __init__(self, num_classes=10):
        super(mask_ConvnetMnist, self).__init__()
        self.conv1=nn.Conv2d(1, 64, kernel_size=3,padding=0)
        self.relu=nn.ReLU(inplace=True)
             #nn.BatchNorm2d(64),
        self.conv2=nn.Conv2d(64, 64, kernel_size=3,padding=0)
        #self.relu2=nn.ReLU(inplace=True)
             #nn.BatchNorm2d(64),
#             nn.MaxPool2d(2),
        self.drop=  nn.Dropout()
        self.dens1 = nn.Linear(64 * 12 * 12, 128)
        #self.relu3 = nn.ReLU(inplace=True)
        self.dens2 = nn.Linear(128, num_classes)


    def forward(self, x, sensUnits, need_softmax=False):
        x=self.conv1(x)
        x=self.relu(x)
        
        mask = torch.ones_like(x)
        if len(sensUnits[0]) != 0:
            sens = torch.tensor(sensUnits[0])
            mask[:, sens, :, :] = 0
        x = x.mul(mask)
        
        x=self.conv2(x)
        x=self.relu(x)
        
        mask = torch.ones_like(x)
        if len(sensUnits[1]) != 0:
            sens = torch.tensor(sensUnits[1])
            mask[:, sens, :, :] = 0
        x = x.mul(mask)
        
        x1 = F.max_pool2d(x, 2, 2)
        x1_flatt=x1.view(x1.shape[0], -1)
        x1_flatt=self.drop(x1_flatt)
        
        x_out= self.dens1(x1_flatt)
        x_out= self.relu(x_out)
        
        mask = torch.ones_like(x_out)
        if len(sensUnits[2]) != 0:
            sens = torch.tensor(sensUnits[2])
            mask[:, sens] = 0
        x_out = x_out.mul(mask)
        
        x_out= self.dens2(x_out)
        if need_softmax:
            return F.log_softmax(x_out, dim=-1)
        return x_out 

    def load_state_dict(self,state_dict,**kwargs):
        if "net"  in state_dict:
            state_dict = state_dict["net"]
        return super(mask_ConvnetMnist,self).load_state_dict(state_dict=state_dict,**kwargs)

class mask_ConvnetCifar(nn.Module):
    def __init__(self, num_classes=10):
        super(mask_ConvnetCifar, self).__init__()
        self.conv1=nn.Conv2d(3, 32, kernel_size=3,padding=1)
        self.relu=nn.ReLU(inplace=True)
             #nn.BatchNorm2d(32),
        self.conv2=nn.Conv2d(32, 32, kernel_size=3,padding=1)
        #self.relu2=nn.ReLU(inplace=True)
             #nn.BatchNorm2d(32),
            #nn.MaxPool2d(2),
        self.conv3=nn.Conv2d(32, 64, kernel_size=3,padding=1)
        #self.relu3=nn.ReLU(inplace=True)
             #nn.BatchNorm2d(64),

        self.conv4=nn.Conv2d(64, 64, kernel_size=3,padding=1)
        #self.relu4=nn.ReLU(inplace=True)
             #nn.BatchNorm2d(64),

             #nn.MaxPool2d(2),
        self.conv5=nn.Conv2d(64, 128, kernel_size=3,padding=1)
        #self.relu5=nn.ReLU(inplace=True)
             #nn.BatchNorm2d(128),

        self.conv6=nn.Conv2d(128, 128, kernel_size=3,padding=1)
        #self.relu6=nn.ReLU(inplace=True)
             #nn.BatchNorm2d(128),

        self.drop=  nn.Dropout()
            #nn.Dropout(),
        self.dens1=nn.Linear(128 * 4 * 4, 1024)
        #self.relu7=nn.ReLU(inplace=True)
            #nn.Dropout(),
        self.dens2=nn.Linear(1024, 512)
        #self.relu8=nn.ReLU(inplace=True)
        self.dens3=nn.Linear(512, num_classes)

    def forward(self, x, sensUnits, need_softmax=False):
        x=self.conv1(x)
        x=self.relu(x)
        index = 0
        
        mask = torch.ones_like(x)
        if len(sensUnits[index]) != 0:
            sens = torch.tensor(sensUnits[index])
            mask[:, sens, :, :] = 0
        x = x.mul(mask)
        index += 1
        
        x=self.conv2(x)
        x=self.relu(x)
        
        mask = torch.ones_like(x)
        if len(sensUnits[index]) != 0:
            sens = torch.tensor(sensUnits[index])
            mask[:, sens, :, :] = 0
        x = x.mul(mask)
        index += 1
        
        x = F.max_pool2d(x, 2, 2)
        
        x=self.conv3(x)
        x=self.relu(x)
        
        mask = torch.ones_like(x)
        if len(sensUnits[index]) != 0:
            sens = torch.tensor(sensUnits[index])
            mask[:, sens, :, :] = 0
        x = x.mul(mask)
        index += 1
        
        x=self.conv4(x)
        x=self.relu(x)
        
        mask = torch.ones_like(x)
        if len(sensUnits[index]) != 0:
            sens = torch.tensor(sensUnits[index])
            mask[:, sens, :, :] = 0
        x = x.mul(mask)
        index += 1
        
        x = F.max_pool2d(x, 2, 2)
        
        x=self.conv5(x)
        x=self.relu(x)
        
        mask = torch.ones_like(x)
        if len(sensUnits[index]) != 0:
            sens = torch.tensor(sensUnits[index])
            mask[:, sens, :, :] = 0
        x = x.mul(mask)
        index += 1
        
        x=self.conv6(x)
        x=self.relu(x)
        
        mask = torch.ones_like(x)
        if len(sensUnits[index]) != 0:
            sens = torch.tensor(sensUnits[index])
            mask[:, sens, :, :] = 0
        x = x.mul(mask)
        index += 1
        
        x = F.max_pool2d(x, 2, 2)
        
        x1_flatt=x.view(x.shape[0], -1)
        
        x_out=self.drop(x1_flatt)
        x_out= self.dens1(x_out)
        x_out= self.relu(x_out)
        
        mask = torch.ones_like(x_out)
        if len(sensUnits[index]) != 0:
            sens = torch.tensor(sensUnits[index])
            mask[:, sens] = 0
        x_out = x_out.mul(mask)
        index += 1

        x_out=self.drop(x_out)
        x_out= self.dens2(x_out)
        x_out= self.relu(x_out)
        
        mask = torch.ones_like(x_out)
        if len(sensUnits[index]) != 0:
            sens = torch.tensor(sensUnits[index])
            mask[:, sens] = 0
        x_out = x_out.mul(mask)
        index += 1

        x_out= self.dens3(x_out)
        
        if need_softmax:
            return F.log_softmax(x_out, dim=-1)
        return x_out 
    def load_state_dict(self,state_dict,**kwargs):
        if "net"  in state_dict:
            state_dict = state_dict["net"]
        return super(mask_ConvnetCifar,self).load_state_dict(state_dict=state_dict,**kwargs)

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)




if __name__=="__main__":
    x=torch.randn(4,3,32,32)
    
    net = ConvnetCifarBN()
    net.apply(weight_init)
    y=net(x)
    
    net = ConvnetCifar()
    net.apply(weight_init)
    y=net(x)
    
    x=torch.randn(4,1,28,28)
    net = ConvnetMnist()
    net.apply(weight_init)
    y=net(x)
    
    net = ConvnetMnistBN()
    net.apply(weight_init)
    y=net(x)