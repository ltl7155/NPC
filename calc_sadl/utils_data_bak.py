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

        "convcifar10,pgd":"11ue5mjkIte3vKvePnKsMImCOhe4dp7ph", #v2
        "mnist,pgd":"1QR9jBJLbtcLPIGgUbzhK15PGr2kOn2ly", #v2
        "cifar_vgg,pgd":"1ayYMonA4YT_DbccfNEN2RUKMfVLmqyCq", #v2
        "alexnet,pgd":"1_T8FTvfOcxEljzOcCNsa0Th8P82L_VeK", #v2



       "convcifar10,manu_100_nature":"1iTFyhOmOhWhgQlDfHco-jEndnok7l-Ic", #v1
#       "convcifar10,manu_100_nature":"1Ju4_e9HiBhzY9tB0fXvy8ZI518asjjJ9", #v2
#       "convcifar10,manu_100_nature":"1DdKsTkTzawSgMqtqI0vLorF--UpB60Ma", #3
#       "convcifar10,manu_100_nature":"1N3aQDEXFZ_6YcDAxz1i57S4FrfpaDf0Q",  #5 rm10 add20
#       "convcifar10,manu_100_nature":"1kt26TzKLWFYVEZPIZP2oJHo0Tn7CQws7", #v4
#       "convcifar10,manu_100_nature":"1kt26TzKLWFYVEZPIZP2oJHo0Tn7CQws7", #v4
       "convcifar10,manu_100_adv":"1B7JSLvJ79TixF8GLh_ZIsWjht8Cfygl2",
       "mnist,manu_100_nature":"1569zmQMm-11E5oH8wcnmPfwDYxvqKS6X",
       "mnist,manu_100_adv":"1ZGa3sBgrlk93rIfVZ-az_Dz_Ikj9Ktpf",
       "cifar_vgg,manu_100_nature":"1hBoMfHaCX_h27OzuLBMkvs8c70AjI4w0",
       "cifar_vgg,manu_100_adv":"1BoqlN91VUHL7nmuQC4ONkPTo425oRGKT",
       "alexnet,manu_100_nature":"1NFzJQ8LImPq2So9Hqq76_lGIJfVd0A8v",
       "alexnet,manu_100_adv":"17CK6b-FQZsHOx1LV2U1ZrWaX4DrqS3No",
       "imagenet,manu_100_nature":"1r3Y6gryLmXa1avJtEfiHoeRwtxqoYwrD",
       "imagenet,manu_100_adv":"1G6LEs4Wl9rPqF0rPeqmA3PDuUBtLZBe7",
       "vgg16_bn,manu_100_nature":"1r3Y6gryLmXa1avJtEfiHoeRwtxqoYwrD",
       "vgg16_bn,manu_100_adv":"1G6LEs4Wl9rPqF0rPeqmA3PDuUBtLZBe7",
#       "convcifar10,manu_100_nature":"1kt26TzKLWFYVEZPIZP2oJHo0Tn7CQws7",
#       "convcifar10,manu_100_nature2":"1DdKsTkTzawSgMqtqI0vLorF--UpB60Ma",
#       "convcifar10,manu_100_nature3":"1Ju4_e9HiBhzY9tB0fXvy8ZI518asjjJ9",
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
