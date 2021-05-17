import numpy as np
import sys
sys.path.append("LRP_path")
sys.path.append("calc_sadl")
from innvestigator import InnvestigateModel
from inverter_util import Flatten
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import time
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
from models.VGG_16 import VGG16
# from models.vgg import vgg16_bn
from models.AlexNet_SVHN import AlexNet 
from vgg import vgg16_bn

from models.sa_models import ConvnetMnist, ConvnetCifar
import pickle
from get_a_single_path import getPath

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

batch_size = 32
dataset = {}
dataloader = {}
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
mnist_data = torchvision.datasets.MNIST(
    root='~/.torch', train=False, download=False, transform=transform_test)
mnist_loader = Data.DataLoader(dataset=mnist_data, batch_size=batch_size, shuffle=False)

cifar10_data = torchvision.datasets.CIFAR10(
        root = './data/cifar-10',
        train = True,
        transform = transform_test,
        download = False)
cifar10_loader = Data.DataLoader(dataset=cifar10_data, batch_size=batch_size, shuffle=False)

dataset["mnist"] = mnist_data
dataset["cifar10"] = cifar10_data

dataloader["mnist"] = mnist_loader
dataloader["cifar10"] = cifar10_loader

models = {}

model_convmnist = ConvnetMnist() 
model_convmnist.load_state_dict(torch.load("./trained_models/mnist_mixup_acc_99.28_ckpt.pth")["net"])

model_convcifar = ConvnetCifar() 
model_convcifar.load_state_dict(torch.load("./trained_models/cifar_mixup_acc_90.36_ckpt.pth")["net"])

model_path = "./trained_models/alexnet_lr0.0001_39.pkl"
model_alexnet=AlexNet()
checkpoint = torch.load(model_path)
model_alexnet.load_state_dict(checkpoint)

model_vgg = VGG16(num_classes=10)
model_path = "./trained_models/model_vgg_cifar/vgg_seed32_dropout.pkl"
checkpoint = torch.load(model_path)
model_vgg.load_state_dict(checkpoint)

models["convmnist"] = model_convmnist
models["convcifar10"] = model_convcifar
models["vgg"] = model_vgg
models["alexnet"] = model_alexnet

from calc_sadl.utils_data import get_model, get_dataset, get_filelist, get_cluster_para

import torch
from dataloader import DatasetAdv
from neuron_coverage import Coverager
# convmnist 0.8, 4, 0.8
# convcifar 0.7 7 0.9
# vgg 0.9 7 0.9

num_classes = 10

bucket_m = 100
# dic = {
#     "convcifar10": "1l0n0kqPbqcgEStCy76K9wiiO-Y4dWB35",
#     "convmnist": "199b0k5M7OUliRzVGepMhYDRwdZ1ofuHe",
#     "vgg": "1wpf8uNfTStBn7NkS0omhVfv2OgIlAqbU"
# }

results = {}
results_layer = {}

s = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time()))

# for model_name in ["vgg", "alexnet", "convcifar10", "convmnist"]:
#     for mode in ["manu_100_nature", "manu_100_adv"]:
file_id_list = get_filelist()

dataset = "imagenet"
arch = "vgg16_bn"
x_train = get_dataset(dataset)

model = vgg16_bn(num_classes=10)
model_path = "./trained_models/vgg16_bn_lr0.0001_49_imagenet_train_layer-1_withDataAugment.pkl"
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint)
model.eval()
models["vgg16_bn"] = model

for model_name in ["vgg16_bn"]:
    for mode in ["manu_100_nature", "manu_100_adv"]:
        sample_threshold, cluster_threshold, cluster_num = get_cluster_para(dataset, model_name)
        print(model_name, mode)
        intra = {}
        layer_intra = {}
        m_name = f"{model_name},{mode}"
        test_set = DatasetAdv(file_id_list[m_name])
        
        fetch_func = lambda x:x["your_adv"]

        for index, datax in enumerate(test_set):
            covered_10 = set()
            covered_100 = set() 
            total_100 = total_10 = 0
            
#                 print("index", index)
#             print(datax["key"],datax["your_adv"].shape,datax["your_label"].shape,"lbl")
            keys = datax["key"]
            print("keys:", keys)
#             x_test =torch.from_numpy(datax["your_adv"]) 
#             y_test = torch.from_numpy(datax["your_label"] )
            x_test = datax["your_adv"]
            y_test = torch.rand((x_test.shape[0]))
#             y_test = datax["your_label"] 
#             print(x_test.shape,type(x_test))
#             print(y_test.shape,type(y_test))
            test_loader1=torch.torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(x_test, y_test),
                    batch_size=25)
            
            for step, (x, y) in enumerate(test_loader1):
                print("step", step)
                x = x.cuda()
                
                models[model_name] = models[model_name].cuda()
                cover = Coverager(models[model_name], model_name, cluster_threshold, num_classes=num_classes, num_cluster=cluster_num)
                covered1, total1 = cover.Intra_NPC(x, y, bucket_m, sample_threshold, mode=mode, simi_soft=False, arc=model_name)
                covered2, total2 = cover.Layer_Intra_NPC(x, y, bucket_m, sample_threshold, mode=mode, simi_soft=False, useOldPaths_X=True, arc=model_name)
                total_10 += total1
                total_100 += total2    
                covered_10 = covered_10 | covered1
                covered_100 = covered_100 | covered2 
#                     print(cover.get_simi(x, y, bucket_m, single_threshold, mode=mode, simi_soft=False))
            intra[keys] = round(len(covered_10) / total_1, 5)
            layer_intra[keys] = round(len(covered_100) / total_10, 5)
            print(m_name, intra[keys])
            print(m_name, layer_intra[keys])
        save_filename_10 = "./coverage_results/{}_Our_10_{}".format(mode, m_name)
        save_filename_100 = "./coverage_results/{}_Our_100_{}".format(mode, m_name)
        np.save(save_filename_10, intra) 
        np.save(save_filename_100, layer_intra) 
        results[m_name] = intra
        results_layer[m_name] = layer_intra
print(results)
print(results_layer)
print("start time:", s)
print("end time:", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time())) )
