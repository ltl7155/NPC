import pickle
import torch
import numpy as np
import torchvision
import sys
sys.path.append("models")
from VGG_16 import VGG16
import os
import torchvision.transforms as transforms

from collections import Counter

import torchvision.datasets as datasets
import random
import torch.utils.data as Data
# from torch.utils.data.sampler import Sampler
from sklearn.cluster import KMeans
import numpy as np
from torch.utils.data import Subset
import argparse
from model_mask_vgg import mask_VGG16
from sa_models import ConvnetMnist, ConvnetCifar, mask_ConvnetMnist, mask_ConvnetCifar


parser = argparse.ArgumentParser(description='model interpretation')
parser.add_argument('--paths_path', type=str, default="LRP_path/lrp_path_threshold0.9_test.pkl")
parser.add_argument('--arc', type=str, default="vgg")
parser.add_argument('--data_train', action='store_true')
parser.add_argument('--b_cluster', action='store_true')
parser.add_argument('--useOldCluster', action='store_true')
parser.add_argument('--dataset', type=str, default="cifar10")
parser.add_argument('--attack', type=str, default="")
parser.add_argument('--gpu', type=str, default="0")
parser.add_argument('--n_clusters', type=int, default=3)
parser.add_argument('--grids', type=int, default=1)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

def get_acc(model, data_loader):
#     right_count = [0.0 for i in range(args.grids)]
#     prob_count = [0.0 for i in range(args.grids)]
#     remedy_count = [0.0 for i in range(args.grids)]
#     num = [0.0 for i in range(args.grids)]
    total = 0
    for test_step, (val_x, val_y) in enumerate(data_loader):
#         print("step:", test_step)
        val_x = val_x.cuda()
        val_y = val_y.cuda()
        val_output = model(val_x)
        _, val_pred_y = val_output.max(1)
        if test_step == 0:
            correct = val_pred_y.eq(val_y).sum().item()
        else:
            correct += val_pred_y.eq(val_y).sum().item()
        total += val_y.size(0)
    #         print(ori_prob)
#     print(correct/total)
    return correct/total

data_path = args.paths_path
with open(data_path, 'rb') as fr:
    paths = pickle.load(fr)
    
if args.dataset == "mnist":
    transform_test = transforms.Compose([
        transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,)),
    ])
    dataset = torchvision.datasets.MNIST(
        root='~/.torch', train=args.data_train, download=True, transform=transform_test)
#     data_loader = torch.utils.data.DataLoader(
#         testset, batch_size=batch_size, shuffle=False)

if args.dataset == "cifar10":
    transform_test = transforms.Compose([
            transforms.ToTensor(),
    ])

    dataset = torchvision.datasets.CIFAR10(
            root = './data/cifar-10',
            train = args.data_train,
            transform = transform_test,
            download = False)
    
if args.attack != "":
    data_root = '../adv_samples/adv_{}_cifar10_samples.npy'.format(args.attack)
    label_root = '../adv_samples/adv_{}_cifar10_labels.npy'.format(args.attack)

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
    
if args.dataset == "mnist":
    ori_model = ConvnetMnist() 
    ori_model.load_state_dict(torch.load("trained_models/mnist_mixup_acc_99.28_ckpt.pth")["net"])
    net = mask_ConvnetMnist() 
    net.load_state_dict(torch.load("trained_models/mnist_mixup_acc_99.28_ckpt.pth")["net"])

elif args.dataset == "cifar10" and args.arc == "convcifar10":
    ori_model = ConvnetCifar() 
    ori_model.load_state_dict(torch.load("trained_models/cifar_mixup_acc_90.36_ckpt.pth")["net"])
    net = mask_ConvnetCifar() 
    net.load_state_dict(torch.load("trained_models/cifar_mixup_acc_90.36_ckpt.pth")["net"])

elif args.dataset == "cifar10"and args.arc == "vgg":
    ori_model = VGG16(num_classes=10)
    model_path = "./trained_models/model_vgg_cifar/vgg_seed32_dropout.pkl"
    checkpoint = torch.load(model_path)
    ori_model.load_state_dict(checkpoint)
    
    net = mask_VGG16(num_classes=10)
    net.load_state_dict(checkpoint)

elif args.dataset == "imagenet":
    ori_model = vgg16_bn(pretrained=True)
    net = mask_vgg16_bn(pretrained=True)

ori_model = ori_model.cuda()
ori_model.eval()
net = net.cuda()
net.eval()

if args.arc == "vgg":
    feature_size = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
elif args.arc == "resnet18":
    feature_size = [64, 64, 64, 128, 128, 256, 256, 512, 512]
elif args.arc == "convmnist":
    feature_size = [64, 64, 128]
elif args.arc == "convcifar10":
    feature_size = [32, 32, 64, 64, 128, 128, 1024, 512]
    

samples_class = [[] for c in range(10)]
paths_class = [[] for c in range(10)]
flatten_paths = [[] for _ in range(10)]
binary_paths = [[] for _ in range(10)]
for index in range(len(dataset)):
    test_y = dataset[index][1]
    samples_class[test_y].append(index)  
    
for c in range(10):
    for i in samples_class[c]:
        paths_class[c].append(paths[i])
        
for cl in range(10):
    for p in paths_class[cl]:
        u = []
        for layer, layer_units in enumerate(p):  
            u.extend(layer_units)
            pad = [-1 for _ in range(feature_size[layer]-len(layer_units))]
            u.extend(pad)
#             print(len(u))
        flatten_paths[cl].append(u)
    
for cl in range(10):
    for p in paths_class[cl]:  
        u = []
        for layer, layer_units in enumerate(p):  
            tmp = [0 for _ in range(feature_size[layer])]
            for k in layer_units:
                tmp[k] = 1
            u.extend(tmp)  
#         print(len(u))
        binary_paths[cl].append(u)

count_class = [[] for _ in range(10)]
prob_class = [[] for _ in range(10)]

maxs = []
mins = []
avgs = []

right_num = 0
bucket_num = 20
buckets = [0 for _ in range(bucket_num)]

counts = np.array([0.0 for _ in range(3)])
probs = np.array([0.0 for _ in range(3)])

for cla in range(10):
    
    num_layers = len(feature_size)
    
    if args.b_cluster:
        X = binary_paths[cla]
    else:
        X = flatten_paths[cla]
#     print(len(X[0]))
    print("class:", cla)
    if not args.useOldCluster:
        kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(X)

    cluster_acc = []

    for cluster in range(args.n_clusters):
        print("cluster:", cluster)
        
        if not args.useOldCluster:
            picked_samples = []
    #         print(kmeans.labels_)
            for i, label in enumerate(kmeans.labels_):
                if label == cluster:
                    picked_samples.append(samples_class[cla][i])
    #             else:
    #                 dropped_samples.append(samples[cla][i])
            if args.b_cluster:
                root_path = "./cluster_paths/{}_binary_cluster/".format(args.arc)
            else:
                root_path = "./cluster_paths/{}/".format(args.arc)
            
            if os.path.exists(root_path) == False:
                os.makedirs(root_path)

            picked_samples_fname = root_path + "num_cluster{}_class{}_cluster{}_picked_samples.pkl".format(args.n_clusters, cla, cluster)
            output = open(picked_samples_fname, 'wb')
            pickle.dump(picked_samples, output)  
            print("all samples clustered!!!!!")
        else:
            if args.b_cluster:
                root_path = "./cluster_paths/{}_binary_cluster/".format(args.arc)
            else:
                root_path = "./cluster_paths/{}/".format(args.arc)
            picked_samples_fname = root_path + "num_cluster{}_class{}_cluster{}_picked_samples.pkl".format(args.n_clusters, cla, cluster)
            with open(picked_samples_fname, "rb") as f:
                unpickler = pickle.Unpickler(f)
                picked_samples = unpickler.load()
        
        num_picked_samples = len(picked_samples)
        print("num_picked_samples,", num_picked_samples)
        
#         print(picked_samples[:10])
        
#         neurons = [[] for _ in range(num_layers)]
#         sensUnits = [[] for _ in range(num_layers)]
#         sta = [[] for _ in range(num_layers)]
        
#         for index in picked_samples:
#             for layer in range(num_layers):
#                 neurons[layer].extend(paths[index][layer])
# #       没出现过的单元也让他出现一次，赋值为1
#         for layer in range(num_layers):
#                 neurons[layer].extend([i for i in range(feature_size[layer])])        
#     #       merging 
        
#         for layer in range(num_layers): 
#             sens = []
#             s = []
#             c = Counter(neurons[layer])
#             mc = c.most_common()
#             for a, b in mc:
#                 sens.append(a)
#                 s.append(b)
#             sensUnits[layer] = sens
#             sta[layer] = s
#         print(sta)
               
#         picked_units = [[] for _ in range(num_layers)]
#         rest_units = [[] for _ in range(num_layers)]
#         sec_picked_units = [[] for _ in range(num_layers)]
        
#         for layer in range(num_layers):
#             for t, s in enumerate(sta[layer]):
#                 if s < round(num_picked_samples * args.threshold):
#                     picked_units[layer] = sensUnits[layer][:t]
#                     rest_units[layer] = sensUnits[layer][t:]
#                     if t * 2 < len(sensUnits[layer]):
#                         sec_picked_units[layer] = sensUnits[layer][t: t*2]
#                     else:
#                         sec_picked_units[layer] = sensUnits[layer][t:]
#                     break
#                 if t == len(sta[layer])-1:
#                     print("warning")
#                     picked_units[layer] = sensUnits[layer]
#                     rest_units[layer] = []
#                     sec_picked_units[layer] = []
        
#         path_fname = root_path + "num_cluster{}_threshold{}_class{}_cluster{}_paths.pkl".format(args.n_clusters, args.threshold, cla, cluster)
#         output = open(path_fname, 'wb')
#         pickle.dump([picked_units, rest_units, sec_picked_units], output)  
        
#         lens = []
#         lens_rest = []
#         for i in picked_units:
#             lens.append(len(i))
#         for i in rest_units:
#             lens_rest.append(len(i))
        sub_dataset = Subset(dataset, picked_samples)
        data_loader = torch.utils.data.DataLoader(sub_dataset, batch_size=64, shuffle=False)
        
#         print("mask picked")
#         print(lens)
        acc = get_acc(ori_model, data_loader)
        
        
#         print("mask rest")
#         print(lens_rest)
#         rest_count, rest_prob, _, _ = mask_units(picked_samples, rest_units)
# #         print("mask sec")
# #         sec_count, sec_prob, _ = mask_units(picked_samples, sec_picked_units)
#         print("result:", picked_count, acc)
    
#         right_num += right_num_cluster
        
        cluster_acc.append(acc)
        buck = round(acc * bucket_num)
        if buck == 20:
            buck = 19
        
#         print("result")
#         print(np.mean(class_acc))
#         print(min(class_acc))
#         print(max(class_acc))
    
#     maxs.append(clu_max)
#     mins.append(clu_min)
#     avgs.append(clu_avg)    
#         counts[0] += picked_count
#         probs[0] += picked_prob
#         counts[1] += rest_count
#         probs[1] += rest_prob
#         counts[2] += 0
#         probs[2] += 0
        
#         print("counts", counts)
#         print("probs", probs)
    class_acc.append(cluster_acc)
    
print("result")
print(np.mean(class_acc))
print(min(class_acc))
print(max(class_acc))

    
    
        
        
        
        
        
    
            
            
    
    
        

    
    
    