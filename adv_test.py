#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
get_ipython().run_line_magic('matplotlib', 'inline')

import sys
sys.path.append("..")

import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
from models.vgg_imagenet import vgg16_bn
import pickle
from models.AlexNet_SVHN import AlexNet

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


model = vgg16_bn(num_classes=10)
model_path = "./trained_models/vgg16_bn_lr0.001_133_imagenet.pkl"
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform_test = transforms.Compose([   
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
])


# In[7]:


data_root = 'adv_samples/adv_pgd_imagenet_vgg16_bn_samples_eps8.npy'
label_root = 'adv_samples/adv_pgd_imagenet_vgg16_bn_labels_eps8.npy'

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

def get_adv_dataset(x, y):
    test_data = torch.from_numpy(x).float()
    test_label = torch.from_numpy(y).long()
    test_data = transforms.functional.normalize(test_data, mean, std, False)
    adv_dataset = torch.utils.data.TensorDataset(test_data, test_label)
    return adv_dataset

#load data
x = np.load(data_root)
x = x.transpose((0,3,1,2))
x = x/255.0
y = np.load(label_root)

#data_loader
dataset = get_adv_dataset(x, y)
data_loader = Data.DataLoader(dataset=dataset, batch_size=64, shuffle=False)


model = model.cuda()
model.eval()
total = 0

max_logits = 0

with torch.no_grad():
    for test_step, (val_x, val_y) in enumerate(data_loader):
        print("step:", test_step)
        val_x = val_x.cuda()
        val_y = val_y.cuda()
        val_output = model(val_x)
        val_pred_logit, val_pred_y = val_output.max(1)
        
        max_logits += val_pred_logit.sum().data
        
        if test_step == 0:
            correct = val_pred_y.eq(val_y).sum().item()
        else:
            correct += val_pred_y.eq(val_y).sum().item()
        total += val_y.size(0)
result = float(correct) * 100.0 / float(total)
print(result)
print('logit_sum:', max_logits)


# In[3]:


data_root = 'adv_samples/adv_pgd_SVHN_alexnet_samples_eps0.012.npy'
label_root = 'adv_samples/adv_pgd_SVHN_alexnet_labels_eps0.012.npy'

def get_adv_dataset(x, y):
    test_data = torch.from_numpy(x).float()
    test_label = torch.from_numpy(y).long()
    adv_dataset = torch.utils.data.TensorDataset(test_data,test_label)
    return adv_dataset

#load data
x = np.load(data_root)
x = x.transpose((0,3,1,2))
x = x/255.0
y = np.load(label_root)
#data_loader
dataset = get_adv_dataset(x, y)
data_loader = Data.DataLoader(dataset=dataset, batch_size=64, shuffle=False)

model = model.cuda()
model.eval()
total = 0

max_logits = 0

with torch.no_grad():
    for test_step, (val_x, val_y) in enumerate(data_loader):
        print("step:", test_step)
        val_x = val_x.cuda()
        val_y = val_y.cuda()
        val_output = model(val_x)
        val_pred_logit, val_pred_y = val_output.max(1)
        
        max_logits += val_pred_logit.sum().data
        
        if test_step == 0:
            correct = val_pred_y.eq(val_y).sum().item()
        else:
            correct += val_pred_y.eq(val_y).sum().item()
        total += val_y.size(0)
result = float(correct) * 100.0 / float(total)
print(result)
print('logit_sum:', max_logits)


# In[4]:


data_root = 'adv_samples/adv_pgd_SVHN_alexnet_samples_eps0.02.npy'
label_root = 'adv_samples/adv_pgd_SVHN_alexnet_labels_eps0.02.npy'

def get_adv_dataset(x, y):
    test_data = torch.from_numpy(x).float()
    test_label = torch.from_numpy(y).long()
    adv_dataset = torch.utils.data.TensorDataset(test_data,test_label)
    return adv_dataset

#load data
x = np.load(data_root)
x = x.transpose((0,3,1,2))
x = x/255.0
y = np.load(label_root)
#data_loader
dataset = get_adv_dataset(x, y)
data_loader = Data.DataLoader(dataset=dataset, batch_size=64, shuffle=False)

model = model.cuda()
model.eval()
total = 0

max_logits = 0

with torch.no_grad():
    for test_step, (val_x, val_y) in enumerate(data_loader):
        print("step:", test_step)
        val_x = val_x.cuda()
        val_y = val_y.cuda()
        val_output = model(val_x)
        val_pred_logit, val_pred_y = val_output.max(1)
        
        max_logits += val_pred_logit.sum().data
        
        if test_step == 0:
            correct = val_pred_y.eq(val_y).sum().item()
        else:
            correct += val_pred_y.eq(val_y).sum().item()
        total += val_y.size(0)
result = float(correct) * 100.0 / float(total)
print(result)
print('logit_sum:', max_logits)


# In[5]:


data_root = 'adv_samples/adv_pgd_SVHN_alexnet_samples_eps0.03.npy'
label_root = 'adv_samples/adv_pgd_SVHN_alexnet_labels_eps0.03.npy'

def get_adv_dataset(x, y):
    test_data = torch.from_numpy(x).float()
    test_label = torch.from_numpy(y).long()
    adv_dataset = torch.utils.data.TensorDataset(test_data,test_label)
    return adv_dataset

#load data
x = np.load(data_root)
x = x.transpose((0,3,1,2))
x = x/255.0
y = np.load(label_root)
#data_loader
dataset = get_adv_dataset(x, y)
data_loader = Data.DataLoader(dataset=dataset, batch_size=64, shuffle=False)

model = model.cuda()
model.eval()
total = 0

max_logits = 0

with torch.no_grad():
    for test_step, (val_x, val_y) in enumerate(data_loader):
        print("step:", test_step)
        val_x = val_x.cuda()
        val_y = val_y.cuda()
        val_output = model(val_x)
        val_pred_logit, val_pred_y = val_output.max(1)
        
        max_logits += val_pred_logit.sum().data
        
        if test_step == 0:
            correct = val_pred_y.eq(val_y).sum().item()
        else:
            correct += val_pred_y.eq(val_y).sum().item()
        total += val_y.size(0)
result = float(correct) * 100.0 / float(total)
print(result)
print('logit_sum:', max_logits)


# In[ ]:





# In[ ]:




