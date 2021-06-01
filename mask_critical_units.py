import pickle
import torch
import torch.nn as nn
import numpy as np

import sys
sys.path.append("models")
import os
import torchvision
import torchvision.transforms as transforms
from collections import Counter
import torchvision.datasets as datasets
import random
import torch.utils.data as Data
import torch.nn.functional as F
import argparse

from tqdm import tqdm 


from deephunter.models import get_net,get_masked_net 

parser = argparse.ArgumentParser(description='model interpretation')
parser.add_argument('--data_train', action='store_true')
parser.add_argument('--paths_path', type=str, default="")
parser.add_argument('--dataset', type=str, default="cifar10")
parser.add_argument('--arc', type=str, default="convcifar10")
parser.add_argument('--fail', action='store_true')
parser.add_argument('--grids', type=int, default=1)
parser.add_argument('--attack', type=str, default="")
parser.add_argument('--gpu', type=str, default="0")
args = parser.parse_args()

attack = args.attack
arch = args.arc

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

data_path = args.paths_path
with open(data_path, 'rb') as fr:
    paths = pickle.load(fr)
    
batch_size = 64
if args.dataset == "mnist":
    transform_test = transforms.Compose([
        transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,)),
    ])
    dataset = torchvision.datasets.MNIST(
        root='~/.torch/', train=args.data_train, download=True, transform=transform_test)
#     data_loader = torch.utils.data.DataLoader(
#         testset, batch_size=batch_size, shuffle=False)
else:
    pass 

    
if args.dataset == "mnist":
    # ori_model = ConvnetMnist() 
    # ori_model.load_state_dict(torch.load("trained_models/mnist_mixup_acc_99.28_ckpt.pth")["net"])
    # net = mask_ConvnetMnist() 
    # net.load_state_dict(torch.load("trained_models/mnist_mixup_acc_99.28_ckpt.pth")["net"])
    ori_model = get_net(name="mnist")
    net = get_masked_net(name="mnist")

ori_model = ori_model.cuda()
ori_model.eval()
net = net.cuda()
net.eval()

grids = args.grids
    
right_count = [0 for i in range(grids)]
prob_count = [0 for i in range(grids)]
remedy_count = [0 for i in range(grids)]
num = [0 for i in range(grids)]
results = []

for index in tqdm(range(len(dataset)),desc="mask each seed of dataset ",total=len(dataset)):
    with torch.no_grad():

        test_x = dataset[index][0].unsqueeze(0).cuda()
        test_y = torch.tensor([dataset[index][1]]).cuda()
        
        if dataset[index][1] < 10:
            ori_test_output = ori_model(test_x)
            ori_cla = torch.max(ori_test_output.squeeze().data, 0)[1].data.item()

            if args.fail:
                if ori_cla == test_y.item():
                    continue

#            print("image index:", index)

            picked_neuron_nums = []
            for i in range(len(paths[index])):
                l = round(len(paths[index][i]) / grids) 
                if l == 0:
                    l = 1
                picked_neuron_nums.append(l)

#            print(picked_neuron_nums)

            for t in range(grids):
                path = []
                for layer in range(len(picked_neuron_nums)):
                    s = picked_neuron_nums[layer] * t
                    e = picked_neuron_nums[layer] * (t+1)

    #                 if e >= len(paths[index][layer]):
    #                     e = len(paths[index][layer])
    #                 s = e - picked_neuron_nums[layer]
                    path.append(paths[index][layer][s:e])

    #             print(path)
                mask_test_output = net(test_x, path)                
                mask_cla = torch.max(mask_test_output.squeeze().data, 0)[1].data.item()

                mask_prob = mask_test_output.squeeze().data[ori_cla]
                ori_prob = ori_test_output.squeeze().data[ori_cla]

                if args.fail:
                    if ori_cla != test_y.item():
                        if mask_cla != ori_cla:
                            right_count[t] += 1
                        if cla == test_y.item():
                            remedy_count[t] += 1
                        prob_count[t] += abs(ori_prob - mask_prob)
                        num[t] += 1
                else:
                    if ori_cla == test_y.item():
                        if mask_cla != ori_cla:
                            right_count[t] += 1
                        prob_count[t] += abs(ori_prob - mask_prob)
                        num[t] += 1
        #         print(ori_prob)
                if num[t] != 0 and index%1000==0:  
                    print("\t per:", t)
                    print("\t count:", right_count[t]/num[t])
                    print("\t prob:", prob_count[t]/num[t])
                    print("\t remedy:", remedy_count[t]/num[t])
                    print("\t num:", num[t])
