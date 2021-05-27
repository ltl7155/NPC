# python cluster_three_level_mask.py --paths_path LRP_path/mnist_convmnist_lrp_path_threshold0.8_train.pkl --arc convmnist --data_train --b_cluster --dataset mnist --n_clusters 4 --threshold 0.8
# python cluster_three_level_mask.py --paths_path LRP_path/mnist_convmnist_lrp_path_threshold0.8_train.pkl --arc convmnist --data_train --b_cluster --dataset mnist --n_clusters 4 --threshold 0.8
# python cluster_three_level_mask.py --paths_path LRP_path/cifar10_vgg_lrp_path_threshold0.9_train.pkl --arc vgg --data_train --b_cluster --dataset mnist --n_clusters 7 --threshold 0.9

#  python cluster_three_level_mask.py --paths_path LRP_path/mnist_convmnist_lrp_path_threshold0.8_train.pkl --arc convmnist --b_cluster --dataset imagenet --gpu 1 --n_clusters 4 --threshold 0.8 --grids 5 --data_train

# python cluster_three_level_mask.py --paths_path LRP_path/cifar10_convcifar10_lrp_path_threshold0.7_train.pkl --arc convcifar10 --b_cluster --dataset cifar10 --gpu 1 --n_clusters 7 --threshold 0.9 --grids 5 --data_train

# python cluster_three_level_mask.py --paths_path LRP_path/cifar10_vgg_lrp_path_threshold0.9_train.pkl --arc vgg --b_cluster --dataset cifar10 --gpu 1 --n_clusters 7 --threshold 0.9 --grids 5 --data_train

# python cluster_three_level_mask.py --paths_path LRP_path/SVHN_alexnet_lrp_path_threshold0.7_train.pkl --arc alexnet --b_cluster --dataset SVHN --gpu 1 --n_clusters 4 --threshold 0.6 --grids 5 --data_train

# python cluster_three_level_mask.py --paths_path LRP_path/imagenet_vgg16_bn_lrp_path_threshold0.7_train.pkl --arc vgg16_bn --b_cluster --dataset imagenet --gpu 1 --n_clusters 4 --threshold 0.7 --grids 5 --data_train
import pickle
import torch
import numpy as np
import torchvision
import sys
# sys.path.append("models")
import os
import torchvision.transforms as transforms

from collections import Counter
from torch.utils.data import Subset

import torchvision.datasets as datasets
import random
import torch.utils.data as Data
from sklearn.cluster import KMeans
import numpy as np
import argparse
from models.sa_models import ConvnetMnist, ConvnetCifar
# from AlexNet_SVHN import AlexNet
# from vgg import vgg16_bn

from models.sa_models import mask_ConvnetMnist, mask_ConvnetCifar
# from model_mask_vgg import mask_VGG16 # imagenet's vgg
# from mask_vgg import mask_vgg16_bn
# from mask_AlexNet_SVHN import mask_AlexNet


# from imagenet10Folder import imagenet10Folder # dataset for imagenet

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
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--threshold', type=float, default=0.3)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from tqdm import tqdm 

def mask_units(picked_samples, picked_units):
    right_count = [0.0 for i in range(args.grids)]
    prob_count = [0.0 for i in range(args.grids)]
    remedy_count = [0.0 for i in range(args.grids)]
    num = [0.0 for i in range(args.grids)]
    
    sub_dataset = Subset(dataset, picked_samples)
    data_loader = torch.utils.data.DataLoader(sub_dataset, batch_size=25, shuffle=False)
    total = 0
    for test_step, (val_x, val_y) in enumerate(data_loader):
        val_x = val_x.cuda()
        val_output = ori_model(val_x)
        masked_val_output = net(val_x, picked_units)
        _, val_pred_y = val_output.max(1)
        _, masked_val_pred_y = masked_val_output.max(1)
        if test_step == 0:
            correct = val_pred_y.eq(masked_val_pred_y).sum().item()
        else:
            correct += val_pred_y.eq(masked_val_pred_y).sum().item()
            
        total += val_y.size(0)
    return total-correct, total


batch_size = args.batch_size
     
data_path = args.paths_path
with open(data_path, 'rb') as fr:
    paths = pickle.load(fr)
    
if args.dataset == "mnist":
    transform_test = transforms.Compose([
        transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,)),
    ])
    dataset = torchvision.datasets.MNIST(
        root='~/.torch/', train=args.data_train, download=True, transform=transform_test)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False)

# if args.dataset == "cifar10":
    # transform_test = transforms.Compose([
            # transforms.ToTensor(),
    # ])
    #
    # dataset = torchvision.datasets.CIFAR10(
            # root = './data/cifar-10',
            # train = args.data_train,
            # transform = transform_test,
            # download = True)
    # data_loader = torch.utils.data.DataLoader(
        # dataset, batch_size=batch_size, shuffle=False)
        #
# if args.dataset == "imagenet":
    # '''
    # this is a intranet path  
    # for any imagenet dataset's issues, please contact us by github 
    # '''
    # if args.data_train:
        # valdir = "/mnt/dataset/Image__ILSVRC2012/ILSVRC2012_img_train/train/"
    # else:
        # valdir = "/mnt/mfs/litl/ICSE_CriticalPath/data/ILSVRC2012_img_val/"
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 # std=[0.229, 0.224, 0.225])
    # dataset = test_dataset = imagenet10Folder (
        # valdir,
        # transforms.Compose([
            # transforms.Resize((224, 224)),
            # transforms.ToTensor(),
            # normalize,
        # ]))
    # print(len(dataset))
    # data_loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
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
            # download = True)
    # data_loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    #
    
if args.attack != "":
    '''
    this is a intranet path  
    if you are interest about the adv dataset, please contact us by github for this specifical detail.
    '''
    data_root = '../adv_samples/adv_fgsm_mnist_mnist_samples_eps0.02.npy'
    label_root = '../adv_samples/adv_fgsm_mnist_mnist_labels_eps0.02.npy'

    def get_adv_dataset(x, y):
        test_data = torch.from_numpy(x).float()
        test_label = torch.from_numpy(y).long()
        adv_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        return adv_dataset

    #load data
    x = np.load(data_root)
    x = x.transpose((0,3,1,2))
    x = x/255.0
    y = np.load(label_root)
    #data_loader
    dataset = get_adv_dataset(x, y)
    data_loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    
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
# elif args.dataset == "SVHN"and args.arc == "alexnet":
    # ori_model = AlexNet(num_classes=10)
    # model_path = "./trained_models/alexnet_lr0.0001_39.pkl"
    # checkpoint = torch.load(model_path)
    # ori_model.load_state_dict(checkpoint)
    #
    # net = mask_AlexNet(num_classes=10)
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

ori_model = ori_model.cuda()
ori_model.eval()
net = net.cuda()
net.eval()

if args.arc == "vgg" or args.arc == "vgg16_bn":
    feature_size = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
elif args.arc == "resnet18":
    feature_size = [64, 64, 64, 128, 128, 256, 256, 512, 512]
elif args.arc == "convmnist":
    feature_size = [64, 64, 128]
elif args.arc == "convcifar10":
    feature_size = [32, 32, 64, 64, 128, 128, 1024, 512]
elif args.arc == "alexnet":
    feature_size = [64, 192, 384, 256, 256]
    

samples_class = [[] for c in range(10)]
right_samples_class = [[] for c in range(10)]
paths_class = [[] for c in range(10)]
flatten_paths = [[] for _ in range(10)]
binary_paths = [[] for _ in range(10)]

root_path = "./cluster_paths/{}_binary_cluster/".format(args.arc)
if os.path.exists(root_path) == False:
    os.makedirs(root_path)
samples_class_file = root_path + "samples_class_{}.pkl".format(args.dataset)
right_samples_class_file = root_path + "right_samples_class_{}.pkl".format(args.dataset)

if os.path.exists(samples_class_file):
    with open(samples_class_file, "rb") as f:
        unpickler = pickle.Unpickler(f)
        samples_class = unpickler.load()
    with open(right_samples_class_file, "rb") as f:
        unpickler = pickle.Unpickler(f)
        right_samples_class = unpickler.load()
else: 
    start_index = end_index = 0
    for step, (val_x, val_y) in tqdm(enumerate(data_loader),total=len(data_loader)):
        val_x = val_x.cuda()
        start_index = end_index
        print("step:", step)
        val_y = val_y
        val_output = ori_model(val_x)
        _, val_pred_y = val_output.max(1)
        end = False
        for i, t in enumerate(val_pred_y):
            if val_y[i] >= 10:
                end = True
                break
            if t < 10:
                samples_class[t].append(i+start_index)
            if t == val_y[i]:
                right_samples_class[t].append(i+start_index)
        if end:
            break
        end_index = start_index + val_x.shape[0]
     
    output = open(samples_class_file, 'wb')
    pickle.dump(samples_class, output)  
    output = open(right_samples_class_file, 'wb')
    pickle.dump(right_samples_class, output)  

samples_class = right_samples_class


# for index in range(len(dataset)):
#     samples_class[dataset[index][1]].append(index)

# for i, j in zip(samples_class, right_samples_class):
    # print(len(i), len(j))
    
    
for c in tqdm(range(10),desc="assign value to paths_class",total=10):
    for i in tqdm(samples_class[c],leave=False):
        paths_class[c].append(paths[i])
        
for cl in tqdm(range(10),desc="flatten_paths",total=10):
    for p in tqdm(paths_class[cl],leave=False):
        u = []
        for layer, layer_units in tqdm(enumerate(p) ,leave=False):  
            u.extend(layer_units)
            pad = [-1 for _ in range(feature_size[layer]-len(layer_units))]
            u.extend(pad)
#             print(len(u))
        flatten_paths[cl].append(u)
    
for cl in tqdm(range(10),desc="binary_paths",total=10):
    for p in tqdm(paths_class[cl],leave=False):  
        u = []
        for layer, layer_units in tqdm(enumerate(p),leave=False):  
            tmp = [0 for _ in range(feature_size[layer])]
            for k in layer_units:
                tmp[k] = 1
            u.extend(tmp)  
#         print(len(u))
        binary_paths[cl].append(u)

count_class = [[] for _ in range(10)]
prob_class = [[] for _ in range(10)]
class_acc = []
right_num = 0

counts = np.array([0.0 for _ in range(3)])
probs = np.array([0.0 for _ in range(3)])

for cla in tqdm(range(10),desc="cluste each class", total=10):
    
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

    for cluster in tqdm(range(args.n_clusters),total=args.n_clusters,leave=False):
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
        
        neurons = [[] for _ in range(num_layers)]
        sensUnits = [[] for _ in range(num_layers)]
        sta = [[] for _ in range(num_layers)]
        
        for index in picked_samples:
#             print("paths", paths[1])
            for layer in range(num_layers):
#                 print("layer", layer)
                n = paths[index][layer]
                neurons[layer].extend(n)
#       没出现过的单元也让他出现一次，赋值为1
        for layer in range(num_layers):
            neurons[layer].extend([i for i in range(feature_size[layer])])        
    #       merging 
        
        for layer in range(num_layers): 
            sens = []
            s = []
            c = Counter(neurons[layer])
            mc = c.most_common()
            for a, b in mc:
                sens.append(a)
                s.append(b)
            sensUnits[layer] = sens
            sta[layer] = s
#         print(sta)
               
        picked_units = [[] for _ in range(num_layers)]
        rest_units = [[] for _ in range(num_layers)]
        sec_picked_units = [[] for _ in range(num_layers)]
        
        for layer in range(num_layers):
            for t, s in enumerate(sta[layer]):
                if s < round(num_picked_samples * args.threshold):
                    picked_units[layer] = sensUnits[layer][:t+1]
                    rest_units[layer] = sensUnits[layer][t+1:]
                    if t * 2 < len(sensUnits[layer]):
                        sec_picked_units[layer] = sensUnits[layer][t: t*2]
                    else:
                        sec_picked_units[layer] = sensUnits[layer][t:]
                    break
                if t == len(sta[layer])-1:
                    print("warning")
                    picked_units[layer] = sensUnits[layer]
                    rest_units[layer] = []
                    sec_picked_units[layer] = []
        
        path_fname = root_path + "num_cluster{}_threshold{}_class{}_cluster{}_paths.pkl".format(args.n_clusters, args.threshold, cla, cluster)
        output = open(path_fname, 'wb')
        pickle.dump([picked_units, rest_units, sec_picked_units], output)  
        
        lens = []
        lens_rest = []
        avg_counts_girds = []
        picked_neuron_nums = []
        
        for i in range(num_layers):
            l = round(len(picked_units[i]) / args.grids) 
            if l == 0:
                l = 1
            picked_neuron_nums.append(l)
            
        for t in range(args.grids):
            picked_units_girds = []
            for layer in range(len(picked_neuron_nums)):
                s = picked_neuron_nums[layer] * t
                e = picked_neuron_nums[layer] * (t+1)
                picked_units_girds.append(picked_units[layer][s:e])

            for i in picked_units_girds:
                lens.append(len(i))
            for i in rest_units:
                lens_rest.append(len(i))

            # print("mask picked")
            # print(lens)
            picked_count, right_num_cluster = mask_units(picked_samples, picked_units_girds)

            # print("mask rest")
            # print(lens_rest)
            rest_count, _ = mask_units(picked_samples, rest_units)


            # print("picked_{} vs rest_{}".format(picked_count, rest_count))
    #         print("mask sec")
    #         sec_count, sec_prob, _ = mask_units(picked_samples, sec_picked_units)
    #         print("result:", picked_count, acc)

            right_num += right_num_cluster

    #         cluster_acc.append(acc)

            counts[0] += picked_count
            probs[0] += 0
            counts[1] += rest_count
            probs[1] += 0
            counts[2] += 0
            probs[2] += 0
            avg_counts_girds.append(counts / right_num)
            print("counts", counts)
#         print("probs", probs)
        
#     class_acc.append(cluster_acc)
    

# print("acc:", class_acc)
print("avg_counts", avg_counts_girds)
# print("avg_ligitDiff", probs / right_num)
    
        
        
        
        
        
    
            
            
    
    
        

    
    
    
