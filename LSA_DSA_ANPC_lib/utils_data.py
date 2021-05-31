import numpy as np
import time
import argparse

from tqdm import tqdm
from keras.datasets import mnist, cifar10
import sys
sys.path.append("..")
#from keras.models import load_model, Model
# from new_sa_torch import fetch_dsa, fetch_lsa, get_sc
# from utils import *
import torchvision
import torchvision.transforms as transforms
from .torch_modelas_keras import  TorchModel
import os
import pickle

import torch.nn as nn 
import torch.nn.functional as F 
import torch 
from  torchvision.datasets  import utils as dtutil

import time 
# import  deephunter.models  as models 
# from models.VGG_16 import VGG16
# from imagenet10Folder import imagenet10Folder
# from vgg import vgg16_bn
# from models.vgg import vgg16_bn
# from  models_old  import ConvnetMnist as NET_MNIST
# from  models_old  import ConvnetCifar as NET_CIFAR10
# from  models_old  import VGG16 as NET_VGG_CIFAR10
# from models.AlexNet_SVHN import AlexNet

from deephunter.models import get_net
#dict_arch2deephunter ={"convmnist":"covnet","convcifar10":"covnet", }
def get_model(dataset, arch):
#    arch2 = dict_arch2deephunter.get(arch,arch)
    model  = get_net(name=arch,dt_name=dataset)
    
   
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
        sample_threshold = 0.7
        cluster_threshold = 0.9
        cluster_num = 4
        
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
    else :
        raise Exception("unkown ")
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
        '''
        this is intranet path, if you are interest ,please contact us by github
        '''
        raise Exception("not support on this version")
    
    
        from deephunter.datasets import imagenet10Folder
        # from torchvision.datasets import ImageFolder

        root="/mnt/mfs/litl/ICSE_CriticalPath/data/10Class_imagenet/train/"
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        test_dataset = imagenet10Folder.imagenet10Folder(
            root=root,
            transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ]),
        )
        print ("fetch data from imagenet dataloader..")
        start_time = time.time() 
        
        test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=128,num_workers=8)
        x_train_list, y_train_list= [],[]
        for x_train,y_train in test_dataloader:
            x_train_list.append(x_train)
            y_train_list.append(y_train)
        
        x_train= torch.cat(x_train_list)
        y_train= torch.cat(y_train_list)
        print ("x_train",x_train.shape,"y_train",y_train.shape)
        
        test_dataset=  torch.utils.data.TensorDataset(x_train,
                                                      y_train)

        end_time = time.time() 
        print ("total cost time,",end_time-start_time)
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



# def get_filelist():
    # from deephunter.datareader import fileid_list
    #
    # return fileid_list.file_id_list

def get_adv_dataset(attack_mode, dataset, arch, attack_epi):
    cur_dir = os.path.dirname( os.path.abspath(__file__) )
    
    data_root = f"{cur_dir}/../adv_samples/adv_{attack_mode}_{dataset}_{arch}_samples_eps{attack_epi}.npy"
    label_root = f"{cur_dir}/../adv_{attack_mode}_{dataset}_{arch}_labels_eps{attack_epi}.npy"

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
