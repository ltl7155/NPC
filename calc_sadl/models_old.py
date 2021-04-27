import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init

import torch 

####
'''
 any model with or without the nn.BatchNorm2d will have big difference

'''
###

class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        
        self.features_0 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.features_1 =nn.ReLU(inplace=True)
        self.features_2 =nn.MaxPool2d(kernel_size=2, stride=2)
        self.features_3 =nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.features_4 =nn.ReLU(inplace=True)
        self.features_5 =nn.MaxPool2d(kernel_size=2, stride=2)
        self.features_6 =nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.features_7 =nn.ReLU(inplace=True)
        self.features_8 =nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.features_9 =nn.ReLU(inplace=True)
        self.features_10 =nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.features_11 =nn.ReLU(inplace=True)
        self.features_12 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.classifier_0 = nn.Dropout()
        self.classifier_1 = nn.Linear(256 * 4 * 4, 4096)
        self.classifier_2 = nn.ReLU(inplace=True)
        self.classifier_3 = nn.Dropout()
            #nn.Linear(4096, 4096),
            #nn.ReLU(inplace=True),
        self.classifier_4 = nn.Linear(4096, num_classes)

        
    def forward(self, x):
        feat_conv1 = self.features_0(x)
        feat_conv1_relu = self.features_1(feat_conv1)
#         print(feat_conv1_relu1.size())
        feat_pool1 = self.features_2(feat_conv1_relu)
        feat_conv2 = self.features_3(feat_pool1)
        feat_conv2_relu = self.features_4(feat_conv2)
#         print(feat_conv2_relu2.size())
        feat_pool2 = self.features_5(feat_conv2_relu)
        feat_conv3 = self.features_6(feat_pool2)
        feat_conv3_relu = self.features_7(feat_conv3)
#         print(feat_conv3_relu3.size())
        feat_conv4 = self.features_8(feat_conv3_relu)
        feat_conv4_relu = self.features_9(feat_conv4)
#         print(feat_conv4_relu4.size())
        feat_conv5 = self.features_10(feat_conv4_relu)
        feat_conv5_relu = self.features_11(feat_conv5)
#         print(feat_conv5_relu5.size())
        feat_pool5 = self.features_12(feat_conv5_relu)
#         print(feat_pool5.size())
        
        x = feat_pool5.view(feat_pool5.size(0), -1)
#         y = self.classifier(x)
        y = self.classifier_0(x)
        y = self.classifier_1(y)
        y = self.classifier_2(y)
        y = self.classifier_3(y)
        y = self.classifier_4(y)
        
        return y
    
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


    def forward(self, x, need_softmax=True):
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
    def __init__(self,num_classes=10):
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





import torch
import torch.nn as nn
import torch.nn.functional as F
import math
 
class VGG16(nn.Module):
    def __init__(self,num_classes=10, if_dropout=True):
        super(VGG16, self).__init__()
        
        self.if_dropout = if_dropout
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)#64 32 32
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)#64 32 32
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)#64 16 16
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)#128 16 16
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)#128 16 16
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)#128 8 8
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)#256 8 8
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU(inplace=False)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)#256 8 8
        self.bn6 = nn.BatchNorm2d(256)
        self.relu6 = nn.ReLU(inplace=False)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)#256 8 8
        self.bn7 = nn.BatchNorm2d(256)
        self.relu7 = nn.ReLU(inplace=False)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)#256 4 4
        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, padding=1)#512 4 4
        self.bn8 = nn.BatchNorm2d(512)
        self.relu8 = nn.ReLU(inplace=False)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1)#512 4 4
        self.bn9 = nn.BatchNorm2d(512)
        self.relu9 = nn.ReLU(inplace=False)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)#512 4 4
        self.bn10 = nn.BatchNorm2d(512)
        self.relu10 = nn.ReLU(inplace=False)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)#512 2 2
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1)#512 2 2
        self.bn11 = nn.BatchNorm2d(512)
        self.relu11 = nn.ReLU(inplace=False)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)#512 2 2
        self.bn12 = nn.BatchNorm2d(512)
        self.relu12 = nn.ReLU(inplace=False)
        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, padding=1)#512 2 2
        self.bn13 = nn.BatchNorm2d(512)
        self.relu13 = nn.ReLU(inplace=False)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)#512 1 1
        self.fc1 = nn.Linear(512, num_classes)
        
        self.drop= nn.Dropout( p = 0.3)
                
    def forward(self,x):
        feature_dict = {}
    
        feat_conv1 = self.conv1(x)
        feat_bn1 = self.bn1(feat_conv1)
        feat_conv1_relu = self.relu1(feat_bn1)
        feat_conv2 = self.conv2(feat_conv1_relu)
        feat_bn2 = self.bn2(feat_conv2)
        feat_conv2_relu = self.relu2(feat_bn2)
        feat_pool1 = self.pool1(feat_conv2_relu)
        
        feat_conv3 = self.conv3(feat_pool1)
        feat_bn3 = self.bn3(feat_conv3)
        feat_conv3_relu = self.relu3(feat_bn3)
        feat_conv4 = self.conv4(feat_conv3_relu)
        feat_bn4 = self.bn4(feat_conv4)
        feat_conv4_relu = self.relu4(feat_bn4)
        feat_pool2 = self.pool2(feat_conv4_relu)
        
        feat_conv5 = self.conv5(feat_pool2)
        feat_bn5 = self.bn5(feat_conv5)
        feat_conv5_relu = self.relu5(feat_bn5)
        feat_conv6 = self.conv6(feat_conv5_relu)
        feat_bn6 = self.bn6(feat_conv6)
        feat_conv6_relu = self.relu6(feat_bn6)
        feat_conv7 = self.conv7(feat_conv6_relu)
        feat_bn7 = self.bn7(feat_conv7)
        feat_conv7_relu = self.relu7(feat_bn7)
        feat_pool3 = self.pool3(feat_conv7_relu)
        
        feat_conv8 = self.conv8(feat_pool3)
        feat_bn8 = self.bn8(feat_conv8)
        feat_conv8_relu = self.relu8(feat_bn8)
        feat_conv9 = self.conv9(feat_conv8_relu)
        feat_bn9 = self.bn9(feat_conv9)
        feat_conv9_relu = self.relu9(feat_bn9)
        feat_conv10 = self.conv10(feat_conv9_relu)
        feat_bn10 = self.bn10(feat_conv10)
        feat_conv10_relu = self.relu10(feat_bn10)
        feat_pool4 = self.pool4(feat_conv10_relu)
        
        feat_conv11 = self.conv11(feat_pool4)
        feat_bn11 = self.bn11(feat_conv11)
        feat_conv11_relu = self.relu11(feat_bn11)
        feat_conv12 = self.conv12(feat_conv11_relu)
        #feat_conv12 = self.conv12(feat_pool4)
        feat_bn12 = self.bn12(feat_conv12)
        feat_conv12_relu = self.relu12(feat_bn12)
        feat_conv13 = self.conv13(feat_conv12_relu)
        feat_bn13 = self.bn13(feat_conv13)
        feat_conv13_relu = self.relu13(feat_bn13)
        feat_pool5 = self.pool5(feat_conv13_relu)
        
        feat_pool5 = feat_pool5.view(feat_pool5.size(0),-1)
        
        
        after_dropout= self.drop(feat_pool5)

        feat_fc1 = self.fc1(after_dropout)
        
        return feat_fc1

if __name__=="__main__":
    x=torch.randn(4,3,32,32)
    
    net = ConvnetCifarBN()
    net.apply(weight_init)
    y=net(x)
    
    net = ConvnetCifar()
    net.apply(weight_init)
    y=net(x)
    
    net = VGG16(10)
    net.apply(weight_init)
    y=net(x)
    
    
    x=torch.randn(4,1,28,28)
    net = ConvnetMnist()
    net.apply(weight_init)
    y=net(x)
    
    net = ConvnetMnistBN()
    net.apply(weight_init)
    y=net(x)
    
    
    
    