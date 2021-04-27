import numpy as np

from innvestigator import InnvestigateModel
from inverter_util import Flatten
import sys
sys.path.append("..")

import os
import torch
import time 
import torch.nn as nn
from torch.autograd import Variable
import argparse
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
from models.VGG_16 import VGG16
# from models.vgg import vgg16_bn
from models.sa_models import ConvnetMnist, ConvnetCifar

import pickle



parser = argparse.ArgumentParser(description='get the paths')
parser.add_argument('--gpu', type=str, default="0")
parser.add_argument('--arc', type=str, default="vgg")
parser.add_argument('--attack', type=str, default="")
parser.add_argument('--per', type=float, default=0.1)
parser.add_argument('--dataset', type=str, default="cifar10")
parser.add_argument('--data_train', action='store_true')
parser.add_argument('--last', type=bool, default=False)
parser.add_argument('--useOldRelev', action='store_true')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

batch_size = 1000

if args.dataset == "mnist":
    transform_test = transforms.Compose([
        transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,)),
    ])
    testset = torchvision.datasets.MNIST(
        root='~/.torch', train=args.data_train, download=True, transform=transform_test)
    data_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False)

if args.dataset == "cifar10":
    transform_test = transforms.Compose([
            transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_data = torchvision.datasets.CIFAR10(
            root = './data/cifar-10',
            train = args.data_train,
            transform = transform_test,
            download = False)

    data_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

if args.dataset == "imagenet":
    valdir = "../../data/Image__ILSVRC2012/val/"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    test_dataset = torchvision.datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]))
    print(len(test_dataset))
    data_loader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

if args.attack != "":
    data_root = '../adv_samples/adv_fgsm_cifar10_vgg_samples_eps0.02.npy'
    label_root = '../adv_samples/adv_fgsm_cifar10_vgg_labels_eps0.02.npy'

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
    data_loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    
def pick_neurons_layer(relev, threshold=0.1, last=False): 
    if len(relev.shape) != 2:
        rel = torch.sum(relev, [2, 3])
    else: 
        rel = relev
    units_all = []
    rest_all = []
    sec_cri_all = []
    for i in range(rel.shape[0]):
        rel_i = rel[i]
        values, units = torch.topk(rel_i, rel_i.shape[0])    
        sum_value = 0
        tmp_value = 0

        part = 0
        if not last:
            for v in values:
                if v > 0:
                    sum_value += v
            for i, v in enumerate(values):
                tmp_value += v
                if tmp_value >= sum_value * threshold or v <= 0:
                    part = i
                    break
            units_picked = units[:part+1].tolist()
            rest = units[part+1:].tolist()
            if part * 2 >= len(units):
                sec_cri = units[part:].tolist()
            else:
                sec_cri = units[part:part*2].tolist()
        else:
            for v in values:
                if v < 0:
                    part = i
            units_picked = units[part:].tolist()
            rest = units[:part].tolist()
        units_all.append(units_picked)
        rest_all.append(rest)
        sec_cri_all.append(sec_cri)
    return units_all, rest_all, sec_cri_all

def pick_neurons_layer_per(relev, last=False): 
    if len(relev.shape) != 2:
        rel = torch.sum(relev, [2, 3])
    else: 
        rel = relev
    units_all = []
    rest_all = []

    for i in range(rel.shape[0]):
        rel_i = rel[i]
        values, units = torch.topk(rel_i, round(rel_i.shape[0] * args.per))
        rest = [j for j in range(rel_i.shape[0]) if j not in units]
        units_all.append(units)
        rest_all.append(rest)

    return units_all, rest_all

feature_size = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
with torch.no_grad():
    
    if args.dataset == "mnist":
        model = ConvnetMnist() 
        model.load_state_dict(torch.load("../trained_models/mnist_mixup_acc_99.28_ckpt.pth")["net"])
        
    elif args.dataset == "cifar10" and args.arc == "convcifar10":
        model = ConvnetCifar() 
        model.load_state_dict(torch.load("../trained_models/cifar_mixup_acc_90.36_ckpt.pth")["net"])
        
    elif args.dataset == "cifar10" and args.arc == "vgg":
        model = VGG16(num_classes=10)

        model_path = "../trained_models/model_vgg_cifar/vgg_seed32_dropout.pkl"
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
        
    elif args.dataset == "imagenet":
        model = vgg16_bn(pretrained=True)
        

    s_time = time.time()
    model = model.cuda()
    model.eval()
                # Convert to innvestigate model
    inn_model = InnvestigateModel(model, lrp_exponent=2,
                                  method="b-rule",
                                  beta=.5)
    if not args.useOldRelev:
        for i, (data, target) in enumerate(data_loader):

            print("index:", i)

            data, target = data.cuda(), target.cuda()
            batch_size = int(data.size()[0])

            model_prediction, _ , true_relevance = inn_model.innvestigate(in_tensor=data)
            
            true_relevance = true_relevance[:]
#             print(true_relevance[0].shape)

#             tmp = torch.sum(true_relevance[0].squeeze(), [1, 2])

            if i == 0:
                relev = true_relevance
            else:
                for l in range(len(relev)):
                    relev[l] = torch.cat((relev[l], true_relevance[l]), 0)
        print("done")
        print(len(relev[0]))
        
        e_time = time.time()
        
        
        if args.data_train:
            if args.attack != "":
                torch.save(relev, "{}_relev_{}_train.pt".format(args.attack, args.arc))
            else:
                torch.save(relev, "{}_relev_{}_train.pt".format(args.dataset, args.arc))
        
        if args.attack != "":
            torch.save(relev, "{}_relev_{}_test.pt".format(args.attack, args.arc))
        else:
            torch.save(relev, "{}_relev_{}_test.pt".format(args.dataset, args.arc))

    if args.useOldRelev:    
        if args.data_train:
            if args.attack != "":
                relev = torch.load("{}_relev_{}_train.pt".format(args.attack, args.arc))
            else:
                relev = torch.load("{}_relev_{}_train.pt".format(args.dataset, args.arc))
        if args.attack != "":
            relev = torch.load("{}_relev_{}_test.pt".format(args.attack, args.arc))
        else:
            relev = torch.load("{}_relev_{}_test.pt".format(args.dataset, args.arc))
        size = []
        for l in range(len(relev)):
            size.append(relev[l].shape[1])
        print(size[::-1])
            

    num_layers = len(relev) - 1
    
    sample_neurons = {}
    sample_rests = {}
    sample_sec = {}
    for layer in range(len(relev)):
        true_layer = num_layers - layer
        print("layer:", true_layer)
        r = relev[true_layer]
        units, rests = pick_neurons_layer_per(r, args.last)
        
        for i in range(len(units)):
            if layer == 0:
                sample_neurons[i] = []
                sample_rests[i] = []
                sample_sec[i] = []
            sample_neurons[i].append(units[i])
            sample_rests[i].append(rests[i])
         
    if not args.data_train:
        save_path = "./{}_{}_lrp_path_per{}_test.pkl".format(args.dataset, args.arc, args.per)
        save_path_rest = "./{}_{}_lrp_path_per{}_test_rest.pkl".format(args.dataset, args.arc, args.per)
        save_path_sec = "./{}_{}_lrp_path_per{}_test_sec.pkl".format(args.dataset, args.arc, args.per)
    else:
        save_path = "./{}_{}_lrp_path_per{}_train.pkl".format(args.dataset, args.arc, args.per)
        save_path_rest = "./{}_{}_lrp_path_per{}_train_rest.pkl".format(args.dataset, args.arc, args.per)
        save_path_sec = "./{}_{}_lrp_path_per{}_train_sec.pkl".format(args.dataset, args.arc, args.per)
        
    if args.attack != "":
        save_path = "./{}_{}_lrp_path_per{}_test.pkl".format(args.attack, args.arc, args.per)
        save_path_rest = "./{}_{}_lrp_path_per{}_test_rest.pkl".format(args.attack, args.arc, args.per)
        save_path_sec = "./{}_{}_lrp_path_per{}_test_sec.pkl".format(args.attack, args.arc, args.per)
    print(e_time - s_time)

    output = open(save_path, 'wb')
    pickle.dump(sample_neurons, output)  
    output_rest = open(save_path_rest, 'wb')
    pickle.dump(sample_rests, output_rest)  
    output_sec = open(save_path_sec, 'wb')
    pickle.dump(sample_sec, output_sec) 
    print("done")