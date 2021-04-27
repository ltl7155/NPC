import torch
import os
import sys
import argparse

import numpy as np
sys.path.append("../models")
sys.path.append("..")
from foolbox.models import PyTorchModel

torch.backends.cudnn.benchmark = True
from sa_models import ConvnetMnist, ConvnetCifar
import torchvision.transforms as T
from torch.utils.data import DataLoader
from VGG_16 import VGG16
from torch.utils.data import Subset
from AlexNet_SVHN import AlexNet
import torchvision
import sys
from vgg import vgg16_bn
from mask_vgg import mask_vgg16_bn
from imagenet10Folder import imagenet10Folder 
from generate_utils import generate_adv_version2
import torch.utils.data as Data


parser = argparse.ArgumentParser(description='attack')
parser.add_argument('--arc', type=str, default="vgg")
parser.add_argument('--dataset', type=str, default="cifar10")
parser.add_argument('--gpu', type=str, default="0")

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES']=args.gpu


if args.dataset == "mnist":
    transform_test = T.Compose([
        T.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_data = torchvision.datasets.MNIST(
        root='~/.torch', train=False, download=True, transform=transform_test)
#     data_loader = torch.utils.data.DataLoader(
#         testset, batch_size=batch_size, shuffle=False)

if args.dataset == "cifar10":
    transform_test = T.Compose([
            T.ToTensor(),
    ])

    test_data = torchvision.datasets.CIFAR10(
            root = '../data/cifar-10',
            train = False,
            transform = transform_test,
            download = False)
    
if args.dataset == "SVHN":
    transform_test = T.Compose([
            T.ToTensor(),
    ])

    test_data = torchvision.datasets.SVHN(
            root = '../data/SVHN',
            split="test",
            transform=transform_test,
            download=False)
    
if args.dataset == "imagenet":
    
    valdir = "/mnt/mfs/litl/ICSE_CriticalPath/data/ILSVRC2012_img_val/"
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
    test_data = imagenet10Folder(
        valdir,
        T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
#             normalize,
        ]))


dataloader = Data.DataLoader(dataset=test_data, batch_size=25, shuffle=False)
# cw_dataset = Subset(test_data, cw_data)
# cw_dataloader =  Data.DataLoader(dataset=cw_dataset, batch_size=64, shuffle=False)

if args.dataset == "mnist":
    ori_model = ConvnetMnist() 
    ori_model.load_state_dict(torch.load("../trained_models/mnist_mixup_acc_99.28_ckpt.pth")["net"])

elif args.dataset == "cifar10" and args.arc == "convcifar10":
    ori_model = ConvnetCifar() 
    ori_model.load_state_dict(torch.load("../trained_models/cifar_mixup_acc_90.36_ckpt.pth")["net"])

elif args.dataset == "cifar10" and args.arc == "vgg":
    ori_model = VGG16(num_classes=10)
    model_path = "../trained_models/model_vgg_cifar/vgg_seed32_dropout.pkl"
    checkpoint = torch.load(model_path)
    ori_model.load_state_dict(checkpoint)
    
elif args.dataset == "SVHN" and args.arc == "alexnet":
    ori_model = AlexNet(num_classes=10)
    model_path = "../trained_models/alexnet_lr0.0001_39.pkl"
    checkpoint = torch.load(model_path)
    ori_model.load_state_dict(checkpoint)
    
elif args.dataset == "imagenet":
    ori_model = vgg16_bn(num_classes=10)
    model_path = "../trained_models/vgg16_bn_lr0.0001_49_imagenet_train_layer-1_withDataAugment.pkl"
    checkpoint = torch.load(model_path)
    ori_model.load_state_dict(checkpoint)  
    
ori_model.eval()
save_path = "../adv_samples/"


# generate_adv_version2(dataloader, ori_model, save_path, _attack="fgsm", dataset=args.dataset, epsilon=0.004, arc=args.arc)
# generate_adv_version2(dataloader, ori_model, save_path, _attack="fgsm", dataset=args.dataset, epsilon=0.012, arc=args.arc)
# generate_adv_version2(dataloader, ori_model, save_path, _attack="fgsm", dataset=args.dataset, epsilon=0.02, arc=args.arc)
# generate_adv_version2(dataloader, ori_model, save_path, _attack="pgd", dataset=args.dataset, epsilon=0.004, arc=args.arc)
# generate_adv_version2(dataloader, ori_model, save_path, _attack="pgd", dataset=args.dataset, epsilon=0.012, arc=args.arc)
# generate_adv_version2(dataloader, ori_model, save_path, _attack="pgd", dataset=args.dataset, epsilon=0.02, arc=args.arc)
generate_adv_version2(dataloader, ori_model, save_path, _attack="pgd", dataset=args.dataset, epsilon=8, arc=args.arc)