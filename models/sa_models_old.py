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
        self.features=nn.Sequential(
             nn.Conv2d(1, 64, kernel_size=3,padding=0),
             nn.ReLU(inplace=True),
             #nn.BatchNorm2d(64),
             nn.Conv2d(64, 64, kernel_size=3,padding=0),
             nn.ReLU(inplace=True),
             #nn.BatchNorm2d(64),
             nn.MaxPool2d(2),
            )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 12 * 12, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
            )


    def forward(self, x):
        x1=self.features(x)
        
        x1_flatt=x1.view(x1.shape[0], -1)
        
        x_out= self.classifier(x1_flatt)
        #return F.log_softmax(x_out, dim=-1)
        return x_out

class ConvnetCifar(nn.Module):
    def __init__(self,num_classes=10):
        super(ConvnetCifar, self).__init__()
        self.features=nn.Sequential(
             nn.Conv2d(3, 32, kernel_size=3,padding=1),
             nn.ReLU(inplace=True),
             #nn.BatchNorm2d(32),
             nn.Conv2d(32, 32, kernel_size=3,padding=1),
             nn.ReLU(inplace=True),
             #nn.BatchNorm2d(32),
             nn.MaxPool2d(2),
             nn.Conv2d(32, 64, kernel_size=3,padding=1),
             nn.ReLU(inplace=True),
             #nn.BatchNorm2d(64),

             nn.Conv2d(64, 64, kernel_size=3,padding=1),
             nn.ReLU(inplace=True),
             #nn.BatchNorm2d(64),

             nn.MaxPool2d(2),
             nn.Conv2d(64, 128, kernel_size=3,padding=1),
             nn.ReLU(inplace=True),
             #nn.BatchNorm2d(128),

             nn.Conv2d(128, 128, kernel_size=3,padding=1),
             nn.ReLU(inplace=True),
             #nn.BatchNorm2d(128),

             nn.MaxPool2d(2) )

        self.classifier  =nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x1=self.features(x)
        
        x1_flatt=x1.view(x1.shape[0], -1)
        
        x_out= self.classifier(x1_flatt)
        #return F.log_softmax(x_out, dim=-1)
        return x_out


class mask_ConvnetMnist(nn.Module):
    def __init__(self, num_classes=10):
        super(mask_ConvnetMnist, self).__init__()
        self.features=nn.Sequential(
             nn.Conv2d(1, 64, kernel_size=3,padding=0),
             nn.ReLU(inplace=True),
             #nn.BatchNorm2d(64),
             nn.Conv2d(64, 64, kernel_size=3,padding=0),
             nn.ReLU(inplace=True),
             #nn.BatchNorm2d(64),
             nn.MaxPool2d(2),
            )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 12 * 12, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )


    def forward(self, x, sensUnits):
#         x1=self.features(x)
        index = 1
        for layer in self.features:
            x = layer(x)
            mask = torch.ones_like(x)
            if (isinstance(layer, nn.ReLU)):
                if len(sensUnits[index-1]) != 0:
                    sens = torch.tensor(sensUnits[index-1])
                    mask[:, sens, :, :] = 0
                index += 1   
            x = x.mul(mask)
        x1 = x      
        x1_flatt=x1.view(x1.shape[0], -1)
        x = x1_flatt
        for layer in self.classifier:
            x = layer(x)
            mask = torch.ones_like(x)
            if (isinstance(layer, nn.ReLU)):
                if index <= len(sensUnits):
                    if len(sensUnits[index-1]) != 0:
                        sens = torch.tensor(sensUnits[index-1])
#                         print(sens)
                        mask[:, sens] = 0
                    index += 1   
            x = x.mul(mask)
        #return F.log_softmax(x_out, dim=-1)
        return x

class mask_ConvnetCifar(nn.Module):
    def __init__(self, num_classes=10):
        super(mask_ConvnetCifar, self).__init__()
        self.features=nn.Sequential(
             nn.Conv2d(3, 32, kernel_size=3,padding=1),
             nn.ReLU(inplace=True),
             #nn.BatchNorm2d(32),
             nn.Conv2d(32, 32, kernel_size=3,padding=1),
             nn.ReLU(inplace=True),
             #nn.BatchNorm2d(32),
             nn.MaxPool2d(2),
             nn.Conv2d(32, 64, kernel_size=3,padding=1),
             nn.ReLU(inplace=True),
             #nn.BatchNorm2d(64),

             nn.Conv2d(64, 64, kernel_size=3,padding=1),
             nn.ReLU(inplace=True),
             #nn.BatchNorm2d(64),

             nn.MaxPool2d(2),
             nn.Conv2d(64, 128, kernel_size=3,padding=1),
             nn.ReLU(inplace=True),
             #nn.BatchNorm2d(128),

             nn.Conv2d(128, 128, kernel_size=3,padding=1),
             nn.ReLU(inplace=True),
             #nn.BatchNorm2d(128),

             nn.MaxPool2d(2) )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x, sensUnits):
#         x1=self.features(x)
        index = 1
        for layer in self.features:
            x = layer(x)
            mask = torch.ones_like(x)
            if (isinstance(layer, nn.ReLU)):
                if len(sensUnits[index-1]) != 0:
                    sens = torch.tensor(sensUnits[index-1])
                    mask[:, sens, :, :] = 0
                index += 1   
            x = x.mul(mask)
        x1 = x      
        x1_flatt=x1.view(x1.shape[0], -1)
        x = x1_flatt
        for layer in self.classifier:
            x = layer(x)
            mask = torch.ones_like(x)
            if (isinstance(layer, nn.ReLU)):
                if index <= len(sensUnits):
                    if len(sensUnits[index-1]) != 0:
                        sens = torch.tensor(sensUnits[index-1])
#                         print(sens)
                        mask[:, sens] = 0
                    index += 1   
            x = x.mul(mask)
        #return F.log_softmax(x_out, dim=-1)
        return x







class ConvnetMnistBN(nn.Module):
    def __init__(self,num_classes=10):
        super(ConvnetMnistBN, self).__init__()
        self.features=nn.Sequential(
             nn.Conv2d(1, 64, kernel_size=3,padding=0),
             nn.ReLU(inplace=True),
             nn.BatchNorm2d(64),
             nn.Conv2d(64, 64, kernel_size=3,padding=0),
             nn.ReLU(inplace=True),
          nn.BatchNorm2d(64),
             nn.MaxPool2d(2),
            )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 12 * 12, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
            )


    def forward(self, x):
        x1=self.features(x)
        
        x1_flatt=x1.view(x1.shape[0], -1)
        
        x_out= self.classifier(x1_flatt)
        #return F.log_softmax(x_out, dim=-1)
        return x_out 

class ConvnetCifarBN(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvnetCifarBN, self).__init__()
        self.features=nn.Sequential(
             nn.Conv2d(3, 32, kernel_size=3,padding=1),
             nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
             nn.Conv2d(32, 32, kernel_size=3,padding=1),
             nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
             nn.MaxPool2d(2),
             nn.Conv2d(32, 64, kernel_size=3,padding=1),
             nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

             nn.Conv2d(64, 64, kernel_size=3,padding=1),
             nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

             nn.MaxPool2d(2),
             nn.Conv2d(64, 128, kernel_size=3,padding=1),
             nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),

             nn.Conv2d(128, 128, kernel_size=3,padding=1),
             nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),

             nn.MaxPool2d(2) )

        self.classifier  =nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x1=self.features(x)
        
        x1_flatt=x1.view(x1.shape[0], -1)
        
        x_out= self.classifier(x1_flatt)
        #return F.log_softmax(x_out, dim=-1)
        return x_out 


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
    
    
    
    