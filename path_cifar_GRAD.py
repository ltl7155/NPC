import pickle
import torch
import numpy as np
import json
from VGG_16_featuredict import VGG16
import os
import argparse
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from collections import Counter
import torch.utils.data as Data
import torchvision.datasets as datasets
import torch.nn as nn
import os
import json


all_blocks = True

parser = argparse.ArgumentParser(description='get the paths by loss')
parser.add_argument('--gpu', type=str, default="1")
parser.add_argument('--arch', type=str, default="resnet56")
parser.add_argument('--weight', type=float, default=0.3)
parser.add_argument('--data_train', action='store_true')
parser.add_argument('--unit_pick_percent', type=int, default=100)
args = parser.parse_args()

print(args)
attacks = ["clean"]
arch = args.arch
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

unit_pick_percent = args.unit_pick_percent

def read_data_label(data_path, label_path):
    with open(data_path, 'rb') as fr:
        test_data = pickle.load(fr)
        size = len(test_data)
    with open(label_path, 'rb') as fr:
        test_label = pickle.load(fr)
    return test_data, test_label, size

if arch == "vgg":
    transform_test = T.Compose([
                T.ToTensor()
    ])
elif arch == "resnet56":
    transform_test = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])


dataset = datasets.CIFAR10(
        root = './data/cifar-10',
        train = args.data_train,
        transform = transform_test,
        download = False
)

def getPath(model, index, arch):
    if arch == "vgg":
        for layer in range(12, -1, -1):
            fore_layer = 'feat_conv{}_relu'.format(layer+1)
            unit_pick_num = int(test_feature_dict[fore_layer].size(1) * unit_pick_percent / 100)
            model.zero_grad()
            weight = torch.ones(loss.size()).cuda()
            gradient = torch.autograd.grad(outputs=loss,
                                           inputs=test_feature_dict[fore_layer],
                                           grad_outputs=weight,
                                           retain_graph=True,
                                           create_graph=True,
                                           only_inputs=True)
            gra = gradient[0].data.cpu().abs().sum((0, 2, 3))
            # gra = gradient[0].data.cpu().sum((0, 2, 3))
            values, units = torch.topk(gra, unit_pick_num)
            for i, v in enumerate(values):
                if v < values[0] * args.weight:
                    t = i
                    break
            units = units[t:]
#             print(values)
            key_path[layer] = units.numpy().tolist()
    elif arch == "resnet56":
        if all_blocks:
            num = 27
        else:
            num = 3
#             print(test_feature_dict)
        for layer in range(num, -1, -1):
            fore_layer = 'feat_conv{}_relu'.format(layer+1)
            model.zero_grad()
            unit_pick_num = int(test_feature_dict[fore_layer].size(1) * unit_pick_percent / 100)
            weight = torch.ones(loss.size()).cuda()
            gradient = torch.autograd.grad(outputs=loss,
                                           inputs=test_feature_dict[fore_layer],
                                           grad_outputs=weight,
                                           retain_graph=True,
                                           create_graph=True,
                                           only_inputs=True)
            gra = gradient[0].data.cpu().abs().sum((0, 2, 3))
            values, units = torch.topk(gra, unit_pick_num)
            t = len(values)
            for i, v in enumerate(values):
                if v < values[0] * args.weight:
                    t = i
                    break
            if layer > 25:
                units = units[t:]
            else:
                units = units[-5:]
            key_path[layer] = units.numpy().tolist()
        
    elif arch == "densenet":
        pass

if arch == "vgg":
    net = VGG16(num_classes=10).cuda()
    load_model_path = "model_vgg_cifar/vgg_seed32_dropout.pkl"
    if os.path.exists(load_model_path):
        net.load_state_dict(torch.load(load_model_path))
        print('load model.')
    else:
        print("load failed.")
elif arch == "resnet56":
    from resnet import *
    models_dir = 'pretrained_models/cifar10/' 
    net = resnet56().cuda()
    state_dict = torch.load(models_dir + 'resnet56.th', map_location='cuda')['state_dict'] # best_prec1, state_dict
    new_state_dict = {}
    for k, v in state_dict.items():
        if 'module' in k:
            k = k.replace('module.', '')
        new_state_dict[k]=v   
    net.load_state_dict(new_state_dict)
elif arch == "desnet":
    pass

net.eval()


def get_adv_dataset(x, y):
    test_data = torch.from_numpy(x).float()
    test_label = torch.from_numpy(y).long()
    adv_dataset = torch.utils.data.TensorDataset(test_data,test_label)
    return adv_dataset

if args.attack != "":
    data_root = 'adv_samples/adv_{}_cifar10_samples.npy'.format(args.attack)
    label_root = 'adv_samples/adv_{}_cifar10_labels.npy'.format(args.attack)

    #load data
    x = np.load(data_root)
    x = x.transpose((0,3,1,2))
    x = x/255.0
    y = np.load(label_root)
    #data_loader
    dataset = get_adv_dataset(x, y)

for attack in attacks:

    key_paths = {}
    right_num = 0

    loss_func = nn.CrossEntropyLoss()

    for index in range(len(dataset)):

        print("image index:", index)
        test_x = dataset[index][0].unsqueeze(0).cuda()
        test_y = torch.tensor([dataset[index][1]]).cuda()
        if arch == "resnet56":
            test_output, test_feature_dict = net(test_x, all_blocks=all_blocks)
        elif arch == "vgg":
            test_output, test_feature_dict = net(test_x)
            
        cla = torch.max(test_output.squeeze().data, 0)[1].data.item()
        
        if cla == dataset[index][1]:
            right_num += 1
        
        loss = loss_func(test_output, test_y)
        if arch == "resnet56":
            if all_blocks:
                key_path = [None] * 28
            else:
                key_path = [None] * 4
        elif arch == "vgg":
            key_path = [None] * 13
            
        getPath(net, index, arch)
        key_paths[index] = key_path
        # print("key_path:", key_path)
        print("target:", test_y)
        print("class:", cla)
        print("right_num:", right_num)
#         print(key_path)
    
    if args.attack != "":    
        if all_blocks:
            save_path = "./grad_path_{}_per{}_weight{}_{}.pkl".format(arch, str(args.unit_pick_percent), args.weight, args.attack)
        else:
            save_path = "./grad_path_{}_per{}_weight{}_{}.pkl".format(arch, str(args.unit_pick_percent), args.weight, args.attack)
    else:
        if all_blocks:
            save_path = "./grad_path_{}_per{}_weight{}_allBlocks_test_last_sepcial.pkl".format(arch, str(args.unit_pick_percent), args.weight)
        else:
            save_path = "./grad_path_{}_per{}_weight{}_partBlocks_test.pkl".format(arch, str(args.unit_pick_percent), args.weight)
    output = open(save_path, 'wb')
    pickle.dump(key_paths, output)




# from collections import Counter
# for i in range(13):
#     c = Counter(units[i])
#     print(c.most_common(1))