import pickle
import torch
import torch.nn as nn
import numpy as np
import json
from VGG_16 import VGG16
import os
import torchvision.transforms as T
from collections import Counter
import torchvision.datasets as datasets
import random
import torch.utils.data as Data
import torch.nn.functional as F
import argparse

class mask_VGG16(nn.Module):
    
    def __init__(self, num_classes):
        super(mask_VGG16, self).__init__()
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
        
        
    def forward(self, x, sens_units):
        mask = []
        mask.append(torch.ones(64, 32, 32).cuda())
        mask.append(torch.ones(64, 32, 32).cuda())
        mask.append(torch.ones(128, 16, 16).cuda())
        mask.append(torch.ones(128, 16, 16).cuda())
        mask.append(torch.ones(256, 8, 8).cuda())
        mask.append(torch.ones(256, 8, 8).cuda())
        mask.append(torch.ones(256, 8, 8).cuda())
        mask.append(torch.ones(512, 4, 4).cuda())
        mask.append(torch.ones(512, 4, 4).cuda())
        mask.append(torch.ones(512, 4, 4).cuda())
        mask.append(torch.ones(512, 2, 2).cuda())
        mask.append(torch.ones(512, 2, 2).cuda())
        mask.append(torch.ones(512, 2, 2).cuda())  
        
        
        for layer in range(13):
            r = [i for i in range(0, len(sens_units[layer]))]
            sens = torch.tensor(sens_units[layer])
            if len(sens_units[layer]) != 0:
                mask[layer][sens[r]] = 0
            
        feat_conv1 = self.conv1(x)
        feat_bn1 = self.bn1(feat_conv1)
        feat_conv1_relu = self.relu1(feat_bn1)
#         print(feat_conv1_relu)
        feat_conv1_relu = feat_conv1_relu.mul(mask[0])

        feat_conv2 = self.conv2(feat_conv1_relu)
        feat_bn2 = self.bn2(feat_conv2)
        feat_conv2_relu = self.relu2(feat_bn2)
        feat_conv2_relu = feat_conv2_relu.mul(mask[1])
        feat_pool1 = self.pool1(feat_conv2_relu)

        feat_conv3 = self.conv3(feat_pool1)
        feat_bn3 = self.bn3(feat_conv3)
        feat_conv3_relu = self.relu3(feat_bn3)
        feat_conv3_relu = feat_conv3_relu.mul(mask[2])
        feat_conv4 = self.conv4(feat_conv3_relu)
        feat_bn4 = self.bn4(feat_conv4)
        feat_conv4_relu = self.relu4(feat_bn4)
        feat_conv4_relu = feat_conv4_relu.mul(mask[3])
        feat_pool2 = self.pool2(feat_conv4_relu)

        feat_conv5 = self.conv5(feat_pool2)
        feat_bn5 = self.bn5(feat_conv5)
        feat_conv5_relu = self.relu5(feat_bn5)
        feat_conv5_relu = feat_conv5_relu.mul(mask[4])
        feat_conv6 = self.conv6(feat_conv5_relu)
        feat_bn6 = self.bn6(feat_conv6)
        feat_conv6_relu = self.relu6(feat_bn6)
        feat_conv6_relu = feat_conv6_relu.mul(mask[5])
        feat_conv7 = self.conv7(feat_conv6_relu)
        feat_bn7 = self.bn7(feat_conv7)
        feat_conv7_relu = self.relu7(feat_bn7)
        feat_conv7_relu = feat_conv7_relu.mul(mask[6])
        feat_pool3 = self.pool3(feat_conv7_relu)

        feat_conv8 = self.conv8(feat_pool3)
        feat_bn8 = self.bn8(feat_conv8)
        feat_conv8_relu = self.relu8(feat_bn8)
        feat_conv8_relu = feat_conv8_relu.mul(mask[7])
        feat_conv9 = self.conv9(feat_conv8_relu)
        feat_bn9 = self.bn9(feat_conv9)
        feat_conv9_relu = self.relu9(feat_bn9)
        feat_conv9_relu = feat_conv9_relu.mul(mask[8])
        feat_conv10 = self.conv10(feat_conv9_relu)
        feat_bn10 = self.bn10(feat_conv10)
        feat_conv10_relu = self.relu10(feat_bn10)
        feat_conv10_relu = feat_conv10_relu.mul(mask[9])
        feat_pool4 = self.pool4(feat_conv10_relu)

        feat_conv11 = self.conv11(feat_pool4)
        feat_bn11 = self.bn11(feat_conv11)
        feat_conv11_relu = self.relu11(feat_bn11)
        
        feat_conv11_relu = feat_conv11_relu.mul(mask[10])
        feat_conv12 = self.conv12(feat_conv11_relu)
        feat_bn12 = self.bn12(feat_conv12)
        feat_conv12_relu = self.relu12(feat_bn12)

        
        feat_conv12_relu = feat_conv12_relu.mul(mask[11])
        feat_conv13 = self.conv13(feat_conv12_relu)
        feat_bn13 = self.bn13(feat_conv13)
        feat_conv13_relu = self.relu13(feat_bn13)

        
        
        feat_conv13_relu = feat_conv13_relu.mul(mask[12])
        feat_pool5 = self.pool5(feat_conv13_relu)

        feat_pool5 = feat_pool5.view(feat_pool5.size(0),-1)

        after_dropout = F.dropout(feat_pool5, p=0.3, training=self.training,inplace=False)

        feat_fc1 = self.fc1(after_dropout)
        return feat_fc1
#     , {'feat_conv1': feat_conv1, 'feat_conv2': feat_conv2, 'feat_conv3': feat_conv3, 'feat_conv4': feat_conv4, 'feat_conv5': feat_conv5, 'feat_conv6': feat_conv6, 'feat_conv7': feat_conv7, 'feat_conv8': feat_conv8, 'feat_conv9' : feat_conv9, 'feat_conv10': feat_conv10, 'feat_conv11': feat_conv11, 'feat_conv12': feat_conv12, 'feat_conv13': feat_conv13, 'feat_conv1_relu': feat_conv1_relu, 'feat_conv2_relu': feat_conv2_relu, 'feat_conv3_relu': feat_conv3_relu, 'feat_conv4_relu': feat_conv4_relu, 'feat_conv5_relu': feat_conv5_relu, 'feat_conv6_relu': feat_conv6_relu, 'feat_conv7_relu': feat_conv7_relu, 'feat_conv8_relu': feat_conv8_relu, 'feat_conv9_relu': feat_conv9_relu, 'feat_conv10_relu': feat_conv10_relu, 'feat_conv11_relu': feat_conv11_relu, 'feat_conv12_relu': feat_conv12_relu, 'feat_conv13_relu': feat_conv13_relu, 'feat_fc1': feat_fc1}