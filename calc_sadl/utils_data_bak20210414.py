import numpy as np
import time
import argparse

from tqdm import tqdm
from keras.datasets import mnist, cifar10
import sys
sys.path.append("..")
#from keras.models import load_model, Model
from new_sa_torch import fetch_dsa, fetch_lsa, get_sc
from utils import *
import torchvision
import torchvision.transforms as transforms
from torch_modelas_keras import  TorchModel

import pickle

import torch.nn as nn 
import torch.nn.functional as F 
import torch 
from  torchvision.datasets  import utils as dtutil

from models.VGG_16 import VGG16
from imagenet10Folder import imagenet10Folder
from vgg import vgg16_bn
# from models.vgg import vgg16_bn
from  models_old  import ConvnetMnist as NET_MNIST
from  models_old  import ConvnetCifar as NET_CIFAR10
from  models_old  import VGG16 as NET_VGG_CIFAR10
from models.AlexNet_SVHN import AlexNet

def get_model(dataset, arch):
    if dataset == "mnist":
        model = NET_MNIST() 
        model.load_state_dict(torch.load("../trained_models/mnist_mixup_acc_99.28_ckpt.pth")["net"])
        
    elif dataset == "cifar10" and arch == "convcifar10":
        model = NET_CIFAR10() 
        model.load_state_dict(torch.load("../trained_models/cifar_mixup_acc_90.36_ckpt.pth")["net"])
      
    elif dataset == "cifar10" and arch == "vgg":
        model = NET_VGG_CIFAR10(num_classes=10)

        model_path = "../trained_models/model_vgg_cifar/vgg_seed32_dropout.pkl"
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
        
    elif dataset == "imagenet10" or dataset == "imagenet":
        model = vgg16_bn(num_classes=10)
        model_path = "../trained_models/vgg16_bn_lr0.0001_49_imagenet_train_layer-1_withDataAugment.pkl"
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
    
    elif dataset == "SVHN":
        model = AlexNet(num_classes=10)
        model_path = "../trained_models/alexnet_lr0.0001_39.pkl"
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
        
    elif dataset == "SVHN_noDataAug":
        model = AlexNet(num_classes=10)
        model_path = "../trained_models/alexnet_lr0.0001_39_noDataAug.pkl"
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
        
    elif dataset == "SVHN_PAT":
        model = AlexNet(num_classes=10)
        model_path = "../trained_models/PAT/PAT_epoch59_lr0.0001.pkl"
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
        
    if arch == "convmnist":
        layer_names = ["0/relu1", "1/relu2", "2/relu3"]
        num_layer = 3
    elif arch == "convcifar10":
        layer_names = ["0/relu1", "1/relu2", "2/relu3", "3/relu4", "4/relu5", "5/relu6"]
        num_layer = 6
    elif arch == "alexnet":
        layer_names = ['0/features/1', '1/features/4', '2/features/7', '3/features/9', '4/features/11']
        num_layer = 5
    elif arch == "vgg":
#         layer_names = ["5/relu6"]
        layer_names = ['0/relu1', '1/relu2', '2/relu3', '3/relu4', '4/relu5', '5/relu6', '6/relu7', '7/relu8', '8/relu9', '9/relu10']
        num_layer = 10
    elif arch == "vgg16_bn":
#         layer_names = ["5/relu6"]
        layer_names = ['0/features/2', '1/features/5', '2/features/9', '3/features/12', '4/features/16', '5/features/19', '6/features/22', '7/features/26', '8/features/29', '9/features/32']
        num_layer = 10
    return model, layer_names, num_layer
 
def get_cluster_para(dataset, arch):
    if dataset == "mnist":
        sample_threshold = 0.8
        cluster_threshold = 0.8
        cluster_num = 4
        
    elif dataset == "cifar10" and arch == "convcifar10":
#         sample_threshold = 0.6
#         cluster_threshold = 0.9
#         cluster_num = 7
        sample_threshold = 0.6
        cluster_threshold = 0.7
        cluster_num = 7
        
    elif dataset == "cifar10" and arch == "vgg":
        sample_threshold = 0.9
        cluster_threshold = 0.9
        cluster_num = 7
        
    elif dataset == "imagenet":
        sample_threshold = 0.7
        cluster_threshold = 0.7
        cluster_num = 4
    
    elif dataset == "imagenet10":
        sample_threshold = 0.7
        cluster_threshold = 0.7
        cluster_num = 4
    
    elif dataset == "SVHN":
        sample_threshold = 0.7
        cluster_threshold = 0.6
        cluster_num = 4
    
    return sample_threshold, cluster_threshold, cluster_num

def get_dataset(dataset):
    x_train= None
    test_dataset=None
    if  dataset == "mnist":
        (x_train,y_train),(x_test,y_test) = mnist.load_data()
        x_train = x_train.astype("float32")
        x_train = (x_train / 255.0) #- (1.0 - CLIP_MAX)

    elif  "cifar" in  dataset:
        (x_train,y_train),(x_test,y_test) = cifar10.load_data()
        x_train = x_train.astype("float32")
        x_train = (x_train / 255.0)# - (1.0 - CLIP_MAX)

    elif  "SVHN" in   dataset or "alexnet" in dataset:
        import torchvision .datasets as vdt
        import torchvision .transforms as vdf
        test_transform = vdf.ToTensor()
        dataset_train=vdt.SVHN(root="~/.torch",transform=test_transform,split="train")
        x_train, y_train= dataset_train.data, dataset_train.labels
        x_train = x_train.transpose(0,2,3,1)
        x_train = x_train # - (1.0 - CLIP_MAX)
        x_train = x_train.astype("float32")
        
        
    elif dataset == "imagenet" or dataset == "imagenet10":
        import torchvision .datasets as vdt
        import torchvision .transforms as vdf
        
        valdir = "/mnt/dataset/Image__ILSVRC2012/ILSVRC2012_img_train/train/"
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        test_dataset = imagenet10Folder(
            root=valdir,
            transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ]))
    else:
        raise Exception(f"unkown dataset = {dataset}")

    if test_dataset is None :
        if len(x_train.shape)<=3:
            x_train=np.expand_dims(x_train,axis=1)
        if x_train.shape[1] not in [1,3] :
            x_train = x_train.transpose(0,3,1,2)
        test_dataset=  torch.utils.data.TensorDataset(torch.from_numpy(x_train),torch.from_numpy(y_train) )
    print ("load. ....dataset=","..."*8,dataset)
    return test_dataset


'''
pgd 500:  

    "cifar_vgg,pgd":"1ayYMonA4YT_DbccfNEN2RUKMfVLmqyCq",
     "cifar,pgd":"11ue5mjkIte3vKvePnKsMImCOhe4dp7ph",
      "mnist,pgd":"1QR9jBJLbtcLPIGgUbzhK15PGr2kOn2ly",
       "alexnet,pgd":"1_T8FTvfOcxEljzOcCNsa0Th8P82L_VeK",




       pgd 100:  

           "cifar_vgg,pgd":"1Gm926_p5_bvhgfDdlQmmV9lUCmjsF5Ft",
            "cifar,pgd":"1nhWO0VT131_9e5ubgzs343EhM9Ru0UvY",
             "mnist,pgd":"1hc_aj908k7_Zs2L4TsaWYENG-GwtdJe2",
              "alexnet,pgd":"1rFm7KzZLHX4UmdoY5JfisBlMy_EZOvh8",

        "convcifar10,pgd":"11ue5mjkIte3vKvePnKsMImCOhe4dp7ph", #v2
        "mnist,pgd":"1QR9jBJLbtcLPIGgUbzhK15PGr2kOn2ly", #v2
        "cifar_vgg,pgd":"1ayYMonA4YT_DbccfNEN2RUKMfVLmqyCq", #v2
        "alexnet,pgd":"1_T8FTvfOcxEljzOcCNsa0Th8P82L_VeK", #v2
'''
def get_filelist():
    file_id_list= {

#100
#        "convcifar10,pgd":"1nhWO0VT131_9e5ubgzs343EhM9Ru0UvY", #v2
#        "mnist,pgd":"1hc_aj908k7_Zs2L4TsaWYENG-GwtdJe2", #v2
#        "cifar_vgg,pgd":"1Gm926_p5_bvhgfDdlQmmV9lUCmjsF5Ft", #v2
#        "alexnet,pgd":"1rFm7KzZLHX4UmdoY5JfisBlMy_EZOvh8", #v2

#500

#       "convcifar10,manu_100_nature":"1kt26TzKLWFYVEZPIZP2oJHo0Tn7CQws7",
#       "convcifar10,manu_100_nature2":"1DdKsTkTzawSgMqtqI0vLorF--UpB60Ma",
#       "convcifar10,manu_100_nature3":"1Ju4_e9HiBhzY9tB0fXvy8ZI518asjjJ9",

#1000
"1KhCHyzHrXdnne892yDBb-Cb6zCtPfu9D":{'dataset': 'mnist', 'arch': 'convmnist', 'name': '20210409120934_mnist_1111.adv.npz', 'attack': 'manu_100_adv'} ,
"1UchWDO28LQb_nHJgcSFH3vjjv1NXjm8Z":{'dataset': 'mnist', 'arch': 'convmnist', 'name': '20210409120934_mnist_1111.nature.npz', 'attack': 'manu_100_nature'} ,
"15n7hMpS0likxVUlm9IwRvWdvpEMeBr3y":{'dataset': 'cifar10', 'arch': 'convcifar10', 'name': '20210409121021_cifar_1111.adv.npz', 'attack': 'manu_100_adv'} ,
"15PHuUDkhL59uKQAXZ_aRoeNj1fz46Vc0":{'dataset': 'cifar10', 'arch': 'convcifar10', 'name': '20210409121021_cifar_1111.nature.npz', 'attack': 'manu_100_nature'} ,
"1h0qE3F322f2mBn23EEXYTWD77EOGBEa8":{'dataset': 'SVHN', 'arch': 'alexnet', 'name': '20210409121614_svhn_1111.adv.npz', 'attack': 'manu_100_adv'} ,
"1qr7lgOpSkrVQixqlChm3SGa48QnZ1Jkv":{'dataset': 'SVHN', 'arch': 'alexnet', 'name': '20210409121614_svhn_1111.nature.npz', 'attack': 'manu_100_nature'} ,
"1yKmY8eDIJP80MoI6aVgbcuHofdlgmVSD":{'dataset': 'cifar10', 'arch': 'vgg', 'name': '20210409121735_cifar_vgg_1111.adv.npz', 'attack': 'manu_100_adv'} ,
"1s-nIEjJczt02SRLbqfSMZ43wbmB3hK84":{'dataset': 'cifar10', 'arch': 'vgg', 'name': '20210409121735_cifar_vgg_1111.nature.npz', 'attack': 'manu_100_nature'} ,
"1LpZZUh-R-Bq9lQDoWAVKHVX1qfRuLDMz":{'dataset': 'mnist', 'arch': 'convmnist', 'name': '20210409120947_mnist_2222.adv.npz', 'attack': 'manu_100_adv'} ,
"1uGPIUwKDI67qpOpfs0Z5NJgkTg6oGckc":{'dataset': 'mnist', 'arch': 'convmnist', 'name': '20210409120947_mnist_2222.nature.npz', 'attack': 'manu_100_nature'} ,
"1akjLPWPmBRyF9fOJw0CkQhVVmbrF9-wC":{'dataset': 'cifar10', 'arch': 'convcifar10', 'name': '20210409121048_cifar_2222.adv.npz', 'attack': 'manu_100_adv'} ,
"1GCUyxP3a1RsNOqajgXKD1ouC78tgMZqk":{'dataset': 'cifar10', 'arch': 'convcifar10', 'name': '20210409121048_cifar_2222.nature.npz', 'attack': 'manu_100_nature'} ,
"1BV6rcUFaj-vjfUl7tOAgE-6aWrFDHwR4":{'dataset': 'SVHN', 'arch': 'alexnet', 'name': '20210409121657_svhn_2222.adv.npz', 'attack': 'manu_100_adv'} ,
"1Le46bhCKcFn5G19nAx-mWLI57eKLLYSm":{'dataset': 'SVHN', 'arch': 'alexnet', 'name': '20210409121657_svhn_2222.nature.npz', 'attack': 'manu_100_nature'} ,
"1A1kWTwFmujjQ6hU1bmKXIWTrsOC6olOo":{'dataset': 'cifar10', 'arch': 'vgg', 'name': '20210409121806_cifar_vgg_2222.adv.npz', 'attack': 'manu_100_adv'} ,
"1OIYpUhkhf_meyJ3iGrlhMfX_gjcgW_Dk":{'dataset': 'cifar10', 'arch': 'vgg', 'name': '20210409121806_cifar_vgg_2222.nature.npz', 'attack': 'manu_100_nature'} ,
"1mr0LJ1QgN2uQdeEARpfUvgQy0t70FDcr":{'dataset': 'mnist', 'arch': 'convmnist', 'name': '20210409121001_mnist_3333.adv.npz', 'attack': 'manu_100_adv'} ,
"1gf8-Hax8IDM901Hf7b9gZq_KaCBwkI4U":{'dataset': 'mnist', 'arch': 'convmnist', 'name': '20210409121001_mnist_3333.nature.npz', 'attack': 'manu_100_nature'} ,
"1qZzA1rob74SqeBCECrNoS-Np37BKAbf6":{'dataset': 'cifar10', 'arch': 'convcifar10', 'name': '20210409121113_cifar_3333.adv.npz', 'attack': 'manu_100_adv'} ,
"1pEARTqgQuNvIgtgERR_X7_7AHvAA907C":{'dataset': 'cifar10', 'arch': 'convcifar10', 'name': '20210409121113_cifar_3333.nature.npz', 'attack': 'manu_100_nature'} ,
"1sxUbYwjxGGtXTvEfsBotQ1TqQiKFcGDY":{'dataset': 'SVHN', 'arch': 'alexnet', 'name': '20210409121737_svhn_3333.adv.npz', 'attack': 'manu_100_adv'} ,
"1rX1uqheeq2YARs0P6iE-dzhdggT-Vzac":{'dataset': 'SVHN', 'arch': 'alexnet', 'name': '20210409121737_svhn_3333.nature.npz', 'attack': 'manu_100_nature'} ,
"1DZGAF3L3VHZeOHDwkbKxpSEzkw9MSGFN":{'dataset': 'cifar10', 'arch': 'vgg', 'name': '20210409121837_cifar_vgg_3333.adv.npz', 'attack': 'manu_100_adv'} ,
"18hzAqP04NmnugWOh8hVYVceokJPS7QRY":{'dataset': 'cifar10', 'arch': 'vgg', 'name': '20210409121837_cifar_vgg_3333.nature.npz', 'attack': 'manu_100_nature'} ,
"13cViLX_B7qDNCOzGQ6PleAfaGU5mUW3i":{'dataset': 'mnist', 'arch': 'convmnist', 'name': '20210409121015_mnist_4444.adv.npz', 'attack': 'manu_100_adv'} ,
"13K3DGetvSIf3jHME7l3HJ9sv90_W4K98":{'dataset': 'mnist', 'arch': 'convmnist', 'name': '20210409121015_mnist_4444.nature.npz', 'attack': 'manu_100_nature'} ,
"1LLBbeRFGXc0A98IJyDpjgoL55OgdtBUA":{'dataset': 'cifar10', 'arch': 'convcifar10', 'name': '20210409121138_cifar_4444.adv.npz', 'attack': 'manu_100_adv'} ,
"1XvaVXzMxd8p2HDU6pkKOaEG8hEgDndEf":{'dataset': 'cifar10', 'arch': 'convcifar10', 'name': '20210409121138_cifar_4444.nature.npz', 'attack': 'manu_100_nature'} ,
"166d1nsMkH0sXkCwUCe3iMzBIvPaFpFI7":{'dataset': 'SVHN', 'arch': 'alexnet', 'name': '20210409121818_svhn_4444.adv.npz', 'attack': 'manu_100_adv'} ,
"1FOVwX32cBgaShRyuYmRnz5HMr2yz6Pv8":{'dataset': 'SVHN', 'arch': 'alexnet', 'name': '20210409121818_svhn_4444.nature.npz', 'attack': 'manu_100_nature'} ,
"1vaLSQ6UYytv_D7GrshBKFlDlsJ3NLByl":{'dataset': 'cifar10', 'arch': 'vgg', 'name': '20210409121908_cifar_vgg_4444.adv.npz', 'attack': 'manu_100_adv'} ,
"1B1cH3h2tpjgoQXTmS-dOn-_NYMLwP9DV":{'dataset': 'cifar10', 'arch': 'vgg', 'name': '20210409121908_cifar_vgg_4444.nature.npz', 'attack': 'manu_100_nature'} ,
"1y1P86hITxlEfLgdd1qxS8iGbiLQMd4aM":{'dataset': 'mnist', 'arch': 'convmnist', 'name': '20210409121029_mnist_5555.adv.npz', 'attack': 'manu_100_adv'} ,
"1zqZJMpqWoKFGFGBVgJVDlx4MGY0LV3vU":{'dataset': 'mnist', 'arch': 'convmnist', 'name': '20210409121029_mnist_5555.nature.npz', 'attack': 'manu_100_nature'} ,
"1sAmNJpOXIk4_w2XBikFVUIEeO6FpzlWQ":{'dataset': 'cifar10', 'arch': 'convcifar10', 'name': '20210409121203_cifar_5555.adv.npz', 'attack': 'manu_100_adv'} ,
"142XauSXlPYwIieFY_3mA9yQIc3RsBeVy":{'dataset': 'cifar10', 'arch': 'convcifar10', 'name': '20210409121203_cifar_5555.nature.npz', 'attack': 'manu_100_nature'} ,
"1-mOLAnqS3joHcjzuktJRxc0Mi1aS1kAM":{'dataset': 'SVHN', 'arch': 'alexnet', 'name': '20210409121859_svhn_5555.adv.npz', 'attack': 'manu_100_adv'} ,
"14yiyvcoyhGZeAc2GTqJBka98ZIFhEo_8":{'dataset': 'SVHN', 'arch': 'alexnet', 'name': '20210409121859_svhn_5555.nature.npz', 'attack': 'manu_100_nature'} ,
"1hyUd9FEPBn_SgoHYSk0qyBIy7VFXBg0o":{'dataset': 'cifar10', 'arch': 'vgg', 'name': '20210409121939_cifar_vgg_5555.adv.npz', 'attack': 'manu_100_adv'} ,
"14mmJ1W3DgtqKzVLTZ2ZRNYb6MkYF-a11":{'dataset': 'cifar10', 'arch': 'vgg', 'name': '20210409121939_cifar_vgg_5555.nature.npz', 'attack': 'manu_100_nature'} ,

"1PMkzrwmSeNJCLF0Xr4_phsWoCOoVrEgl":{'dataset': 'cifar10', 'arch': 'convcifar10', 'name': 'pgd_results_cifar10_convnetcifar_20210406170020.pkl', 'attack': 'pgd'} ,
"1pBKr3eKD28l0Iakonm20bEQnEVltqif9":{'dataset': 'cifar10', 'arch': 'convcifar10', 'name': 'pgd_results_cifar10_convnetcifar_20210407110526.pkl', 'attack': 'pgd'} ,
"1ddTMu2DIb-ZwbtQMzrRBlmutMco1kONl":{'dataset': 'cifar10', 'arch': 'convcifar10', 'name': 'pgd_results_cifar10_convnetcifar_20210407112628.pkl', 'attack': 'pgd'} ,
"1vyBF3yZTsMYnW2G_6mDbQQ0wp13Yzz5g":{'dataset': 'cifar10', 'arch': 'convcifar10', 'name': 'pgd_results_cifar10_convnetcifar_20210407114827.pkl', 'attack': 'pgd'} ,
"1VfZ2b-vTcILNhujmt6ylM62SI-u4Pzix":{'dataset': 'cifar10', 'arch': 'convcifar10', 'name': 'pgd_results_cifar10_convnetcifar_20210407120724.pkl', 'attack': 'pgd'} ,
"1aH7Wt_r4UlYI1fXgz5tev0SZnW9jfS2-":{'dataset': 'cifar10', 'arch': 'convcifar10', 'name': 'pgd_results_cifar10_convnetcifar_20210407122453.pkl', 'attack': 'pgd'} ,
"1fdPIh1ikEdwsfR6hr1noD5SgQ897CX0g":{'dataset': 'cifar10', 'arch': 'vgg', 'name': 'pgd_results_cifar10vgg_vgg16_20210406170020.pkl', 'attack': 'pgd'} ,
"16ROI2C-KsfbiP_VMxr1c72LAHIVtn0vB":{'dataset': 'cifar10', 'arch': 'vgg', 'name': 'pgd_results_cifar10vgg_vgg16_20210407110524.pkl', 'attack': 'pgd'} ,
"1KOJL4GM1NlfK-q0pzS2DzGS-7dEazzZ5":{'dataset': 'cifar10', 'arch': 'vgg', 'name': 'pgd_results_cifar10vgg_vgg16_20210407112006.pkl', 'attack': 'pgd'} ,
"1IevIUqzcO_0628Nw1L6pIKSBCiVzkpLd":{'dataset': 'cifar10', 'arch': 'vgg', 'name': 'pgd_results_cifar10vgg_vgg16_20210407113635.pkl', 'attack': 'pgd'} ,
"1jNa4lVaaPmmBnz-rQDvUtfXtLh4ZcQtt":{'dataset': 'cifar10', 'arch': 'vgg', 'name': 'pgd_results_cifar10vgg_vgg16_20210407115207.pkl', 'attack': 'pgd'} ,
"1oPjzzcATzDeBUUW4B7kS526_pckLkc6b":{'dataset': 'cifar10', 'arch': 'vgg', 'name': 'pgd_results_cifar10vgg_vgg16_20210407120629.pkl', 'attack': 'pgd'} ,
"1ge0XKMP487WTE3vSD8FAzZR-BQ-4LEIS":{'dataset': 'imagenet10', 'arch': 'vgg16_bn', 'name': 'pgd_results_imagenet_vgg_20210412072817.pkl', 'attack': 'pgd'} ,
"1JvKB-BaYmDjUaqERmrvuv0Jvnd3zWIFJ":{'dataset': 'imagenet10', 'arch': 'vgg16_bn', 'name': 'pgd_results_imagenet_vgg_20210412072818.pkl', 'attack': 'pgd'} ,
"1sjR080-jPHQUeutsS8QN2NUo9YQRtX5l":{'dataset': 'imagenet10', 'arch': 'vgg16_bn', 'name': 'pgd_results_imagenet_vgg_20210412083321.pkl', 'attack': 'pgd'} ,
"1AdomD83ghvAB7egRjKpjjlsWIXtPgSbq":{'dataset': 'imagenet10', 'arch': 'vgg16_bn', 'name': 'pgd_results_imagenet_vgg_20210412083322.pkl', 'attack': 'pgd'} ,
"1aiJqlzKxTpSS2Mf2KqRz4_NVHWOOwvlt":{'dataset': 'imagenet10', 'arch': 'vgg16_bn', 'name': 'pgd_results_imagenet_vgg_20210412083323.pkl', 'attack': 'pgd'} ,
"1b7NE369D_Qi4vrPkzjTbdUgkp9Lb4_Rf":{'dataset': 'imagenet10', 'arch': 'vgg16_bn', 'name': 'pgd_results_imagenet_vgg_20210412083324.pkl', 'attack': 'pgd'} ,
"1fPvYcX3YhIrTtreug7epJd0ApUQJhIVw":{'dataset': 'mnist', 'arch': 'convmnist', 'name': 'pgd_results_mnist_convnetmnist_20210406170020.pkl', 'attack': 'pgd'} ,
"1pm84O8HeYaFaiOllfvbcAaifyiD6M6-S":{'dataset': 'mnist', 'arch': 'convmnist', 'name': 'pgd_results_mnist_convnetmnist_20210406170517.pkl', 'attack': 'pgd'} ,
"1BFNZs2njSqPlua3JGqS68Bx4y9y_k8PQ":{'dataset': 'mnist', 'arch': 'convmnist', 'name': 'pgd_results_mnist_convnetmnist_20210406170829.pkl', 'attack': 'pgd'} ,
"101W-c-B2d63VRc2Hc0soW8uiiius0FSo":{'dataset': 'mnist', 'arch': 'convmnist', 'name': 'pgd_results_mnist_convnetmnist_20210406170918.pkl', 'attack': 'pgd'} ,
"18uZaDIv-bQftqFPV9MQpgI5YegPLBgrN":{'dataset': 'mnist', 'arch': 'convmnist', 'name': 'pgd_results_mnist_convnetmnist_20210406171010.pkl', 'attack': 'pgd'} ,
"1xNmR3IvsAZPjhX7uuvPiSU6ccdkWaZql":{'dataset': 'mnist', 'arch': 'convmnist', 'name': 'pgd_results_mnist_convnetmnist_20210406171103.pkl', 'attack': 'pgd'} ,
"1iw-G-YdVMLhAH3PSlT-5YTt3v7s-QGkg":{'dataset': 'mnist', 'arch': 'convmnist', 'name': 'pgd_results_mnist_convnetmnist_20210406171156.pkl', 'attack': 'pgd'} ,
"1rGItsY0oDyrfHsMbLyruP2qbNalXMjhJ":{'dataset': 'SVHN', 'arch': 'alexnet', 'name': 'pgd_results_svhn_alexnet_20210412154422.pkl', 'attack': 'pgd'} ,
"1jjWcpJpAPjGL4I4b_MVN92ZCJadsjmiO":{'dataset': 'SVHN', 'arch': 'alexnet', 'name': 'pgd_results_svhn_alexnet_20210412154717.pkl', 'attack': 'pgd'} ,
"15qto7sfTK2PirWukvmWHg-eti5nqbTXl":{'dataset': 'SVHN', 'arch': 'alexnet', 'name': 'pgd_results_svhn_alexnet_20210412154718.pkl', 'attack': 'pgd'} ,
"19Rq3MRkuINSS8xS2Jr3ZEcHGHxeDxy6n":{'dataset': 'SVHN', 'arch': 'alexnet', 'name': 'pgd_results_svhn_alexnet_20210412154719.pkl', 'attack': 'pgd'} ,
"1waypBRR-SI3pkHnNpPxlpppqhb-jbJ15":{'dataset': 'SVHN', 'arch': 'alexnet', 'name': 'pgd_results_svhn_alexnet_20210412154720.pkl', 'attack': 'pgd'} ,

         }
    return file_id_list

def get_adv_dataset(attack_mode, dataset, arch, attack_epi):
    data_root = f"../adv_samples/adv_{attack_mode}_{dataset}_{arch}_samples_eps{attack_epi}.npy"
    label_root = f"../adv_samples/adv_{attack_mode}_{dataset}_{arch}_labels_eps{attack_epi}.npy"

    def get_adv(x, y):
        test_data = torch.from_numpy(x).float()
        test_label = torch.from_numpy(y).long()
        adv_dataset = torch.utils.data.TensorDataset(test_data,test_label)
        return adv_dataset

    #load data
    x = np.load(data_root)
    x_train = x.astype("float32")
    x_train = x_train/255.0
    
    print(x_train.shape)
    return x_train
