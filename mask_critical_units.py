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
# from imagenet10Folder import imagenet10Folder 
#
# from models.VGG_16 import VGG16
# from model_mask_vgg import mask_VGG16
#
# from vgg import vgg16_bn
# from mask_vgg import mask_vgg16_bn

from sa_models import ConvnetMnist, ConvnetCifar, mask_ConvnetMnist, mask_ConvnetCifar
      
# from AlexNet_SVHN import AlexNet
# from mask_AlexNet_SVHN import mask_AlexNet
# from mask_AlexNet_SVHN_lastLayer import mask_AlexNet_lastLayer

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

# if args.dataset == "cifar10":
    # transform_test = transforms.Compose([
            # transforms.ToTensor(),
    # ])
    #
    # dataset = torchvision.datasets.CIFAR10(
            # root = './data/cifar-10',
            # train = args.data_train,
            # transform = transform_test,
            # download = False)
            #
# #     data_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

# if args.dataset == "imagenet":
    # '''
    # we didnot upload the imagenet dataset into github, if you are interest to this dataset, 
    # please download and extract it from imagenet.org by yourself. 
    # '''
    # valdir = "/mnt/dataset/Image__ILSVRC2012/ILSVRC2012_img_train/train/"
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 # std=[0.229, 0.224, 0.225])
    # dataset = imagenet10Folder(
        # valdir,
        # transforms.Compose([
            # transforms.Resize((224, 224)),
            # transforms.ToTensor(),
            # normalize,
        # ]))
    # print(len(dataset))
# #     data_loader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
#
# if args.dataset == "SVHN":
    # transform_test = transforms.Compose([
            # transforms.ToTensor(),
    # ])
    #
    # dataset = torchvision.datasets.SVHN(
            # root = './data/SVHN',
            # split="train",
            # transform = transform_test,
            # download = False)

    
# if args.attack != "":
    # data_root = '../adv_samples/adv_fgsm_mnist_mnist_samples_eps0.02.npy'
    # label_root = '../adv_samples/adv_fgsm_mnist_mnist_labels_eps0.02.npy'
    #
    # def get_adv_dataset(x, y):
        # test_data = torch.from_numpy(x).float()
        # test_label = torch.from_numpy(y).long()
        # adv_dataset = torch.utils.data.TensorDataset(test_data,test_label)
        # return adv_dataset
        #
    # #load data
    # x = np.load(data_root)
    # x = x.transpose((0,3,1,2))
    # x = x/255.0
    # y = np.load(label_root)
    # #data_loader
    # dataset = get_adv_dataset(x, y)
    
    
if args.dataset == "mnist":
    ori_model = ConvnetMnist() 
    ori_model.load_state_dict(torch.load("trained_models/mnist_mixup_acc_99.28_ckpt.pth")["net"])
    net = mask_ConvnetMnist() 
    net.load_state_dict(torch.load("trained_models/mnist_mixup_acc_99.28_ckpt.pth")["net"])

# elif args.dataset == "cifar10" and args.arc == "convcifar10":
    # ori_model = ConvnetCifar() 
    # ori_model.load_state_dict(torch.load("trained_models/cifar_mixup_acc_90.36_ckpt.pth")["net"])
    # net = mask_ConvnetCifar() 
    # net.load_state_dict(torch.load("trained_models/cifar_mixup_acc_90.36_ckpt.pth")["net"])
    #
# elif args.dataset == "cifar10"and args.arc == "vgg":
    # ori_model = VGG16(num_classes=10)
    # model_path = "./trained_models/model_vgg_cifar/vgg_seed32_dropout.pkl"
    # checkpoint = torch.load(model_path)
    # ori_model.load_state_dict(checkpoint)
    #
    # net = mask_VGG16(num_classes=10)
    # net.load_state_dict(checkpoint)
    #
# elif args.dataset == "SVHN" and args.arc == "alexnet":
    # ori_model = AlexNet(num_classes=10)
    # model_path = "./trained_models/alexnet_lr0.0001_39.pkl"
    # checkpoint = torch.load(model_path)
    # ori_model.load_state_dict(checkpoint)
    #
    # net = mask_AlexNet(num_classes=10)
    # net.load_state_dict(checkpoint)
    #
# elif args.dataset == "SVHN" and args.arc == "alexnet_lastLayer":
    # ori_model = AlexNet(num_classes=10)
    # model_path = "./trained_models/alexnet_lr0.0001_39.pkl"
    # checkpoint = torch.load(model_path)
    # ori_model.load_state_dict(checkpoint)
    #
    # net = mask_AlexNet_lastLayer(num_classes=10)
    # net.load_state_dict(checkpoint)
    #
# elif args.dataset == "imagenet" and args.arc == "vgg16_bn":
    # ori_model = vgg16_bn(num_classes=10)
    # model_path = "./trained_models/vgg16_bn_lr0.0001_49_imagenet_train_layer-1_withDataAugment.pkl"
    # checkpoint = torch.load(model_path)
    # ori_model.load_state_dict(checkpoint)  
    #
    # net = mask_vgg16_bn(num_classes=10)
    # net.load_state_dict(checkpoint)  
    #
# elif args.dataset == "imagenet" and args.arc == "alexnet":
    # ori_model = vgg16_bn(num_classes=10)
    # model_path = "./trained_models/vgg16_bn_lr0.001_5_imagenet_train_layer-1.pkl"
    # checkpoint = torch.load(model_path)
    # ori_model.load_state_dict(checkpoint)  
    #
    # net = mask_vgg16_bn(num_classes=10)
    # net.load_state_dict(checkpoint)  

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
