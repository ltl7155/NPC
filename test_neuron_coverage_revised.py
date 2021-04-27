import numpy as np
import sys
sys.path.append("LRP_path")

from innvestigator import InnvestigateModel
from inverter_util import Flatten
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
from models.VGG_16 import VGG16
# from models.vgg import vgg16_bn
from models.AlexNet_SVHN import AlexNet 


from models.sa_models import ConvnetMnist, ConvnetCifar
import pickle
from get_a_single_path import getPath

#os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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



import torch
from dataloader_adv_test import DatasetAdv
from neuron_coverage_revised import Coverager
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
dic = {
#     "pgd,vgg": "1Gm926_p5_bvhgfDdlQmmV9lUCmjsF5Ft",
#     "pgd,alexnet":"1rFm7KzZLHX4UmdoY5JfisBlMy_EZOvh8",
#     "pgd,convcifar10":"1nhWO0VT131_9e5ubgzs343EhM9Ru0UvY",
#     "pgd,convmnist":"1hc_aj908k7_Zs2L4TsaWYENG-GwtdJe2",
    "adv,vgg": "118iq7YB3I92PZoiFds2bS7SquChfAJjn",
    "adv,alexnet":"1z9HjyzYQw83gpg3gz2LrYKPcBYYuRVRL",
    "adv,convcifar10":"1CSEGygqLyPiI02B5yn0pkqySf7-pndEl",
    "adv,convmnist":"1mk5mtZJNg0qcfYTF46uOyfIhjq8eYEHE",
    "nat,vgg": "1d6Ebf4Hos4p2RhGcFeYEIGR7FQ_l3AO0",
    "nat,alexnet":"1gY-9WRKcrGDBW8tWLVrLCVbnNKPebzlW",
    "nat,convcifar10":"1ukogsUcDEr8lE-JsXr2qTd5DMEsHBWha",
    "nat,convmnist":"1Cr6GYJHqj0K58QToM7UdtspePOXPTtrz",

#"alexnet":"1z9HjyzYQw83gpg3gz2LrYKPcBYYuRVRL",

#    "alexnet": "1gY-9WRKcrGDBW8tWLVrLCVbnNKPebzlW",
}

results = {}
results_layer = {}
for model_name in ["convmnist", "convcifar10", "alexnet", "vgg"]:
    for mode in ["nat"]:
        if model_name == "convmnist":
            cluster_threshold = 0.8
            num_cluster = 4
            single_threshold = 0.8
        elif model_name == "convcifar10":
            cluster_threshold = 0.9
            num_cluster = 7
            single_threshold = 0.7
        elif model_name == "vgg":
            cluster_threshold = 0.9
            num_cluster = 7
            single_threshold = 0.9
        elif model_name == "alexnet":
            cluster_threshold = 0.6
            num_cluster = 4
            single_threshold = 0.7
        print(model_name, mode)
        intra = {}
        layer_intra = {}
        m_name = f"{mode},{model_name}"
        test_set = DatasetAdv(dic[m_name])
        
        fetch_func = lambda x:x["your_adv"]

        for index, datax in enumerate(test_set):
            covered_10 = set()
            covered_100 = set() 
#                 print("index", index)
#             print(datax["key"],datax["your_adv"].shape,datax["your_label"].shape,"lbl")
            keys = datax["key"]
            print("keys:", keys)
#             x_test =torch.from_numpy(datax["your_adv"]) 
#             y_test = torch.from_numpy(datax["your_label"] )
            x_test = datax["your_adv"]
            y_test = datax["your_label"] 
#             print(x_test.shape,type(x_test))
#             print(y_test.shape,type(y_test))
            test_loader1=torch.torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(x_test, y_test),
                    batch_size=x_test.shape[0])
#             if index < 10:
            for step, (x, y) in enumerate(test_loader1):
                print("step", step)
                x = x.cuda()
                
                models[model_name] = models[model_name].cuda()
                cover = Coverager(models[model_name], model_name, cluster_threshold, num_classes=num_classes, num_cluster=num_cluster)
                covered1, total1 = cover.Intra_NPC(x, y, bucket_m, single_threshold, mode=mode, simi_soft=False, arc=model_name)
                covered2, total2 = cover.Layer_Intra_NPC(x, y, bucket_m, single_threshold, mode=mode, simi_soft=False, useOldPaths_X=True, arc=model_name)
                covered_10 = covered_10 | covered1
                covered_100 = covered_100 | covered2 
#                     print(cover.get_simi(x, y, bucket_m, single_threshold, mode=mode, simi_soft=False))
            intra[keys] = round(len(covered_10) / total1, 5)
            layer_intra[keys] = round(len(covered_100) / total2, 5)
            print(intra[keys])
            print(layer_intra[keys])
        save_filename_10 = "./coverage_results/{}_Our_10_{}".format(mode, m_name)
        save_filename_100 = "./coverage_results/{}_Our_100_{}".format(mode, m_name)
        np.save(save_filename_10, intra) 
        np.save(save_filename_100, layer_intra) 
        results[m_name] = intra
        results_layer[m_name] = layer_intra
        
