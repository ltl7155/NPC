import torch
from time import time
import pickle
import torch
import numpy as np
import json

import difflib

import os
import torchvision.transforms as T
from collections import Counter

import torchvision.datasets as datasets
import random
import torch.utils.data as Data
import gc
from sklearn.cluster import KMeans
import numpy as np
from numpy import dot
from numpy.linalg import norm


def sim_units(unitsA, unitsB, mode="jaccard"):
    
    if mode == "jaccard":
        u = list(set(unitsA) | set(unitsB))
        i = list(set(unitsA) & set(unitsB))
        return len(i)/len(u)
    
    elif mode == "cosine":
        a = np.array(unitsA)
        b = np.array(unitsB)
        cos_sim = dot(a, b)/(norm(a)*norm(b))
        return cos_sim
    
    elif mode == "match":
        a = np.array(unitsA)
        b = np.array(unitsB)
        sim = difflib.SequenceMatcher(None, a, b)
        return sim.ratio()
        
def sim_paths(pathsA, pathsB, mode="jaccard"):
    s = 0
    sims = []
    num = 0
    for layer in range(min(len(pathsA), len(pathsB))):
        s_layer = round(sim_units(pathsA[layer], pathsB[layer], mode=mode), 4)
        s += s_layer
        num += 1
        sims.append(s_layer)
    s = round(s/num, 4)
    return s, sims

def sim_samples(paths1, paths2, samples1, samples2, mode="jaccard"):
    s_all = 0
    num = 0
    sims_all = np.array([0.0 for _ in range(len(paths1[0]))])
    for s1 in samples1:
        p1 = paths1[s1]
        for s2 in samples2:
            p2 = paths2[s2]
            s, sims = sim_paths(p1, p2, mode=mode)
            s_all += s
            sims_all += np.array(sims)  
            num += 1
    s_avg = s_all / num
    return round(s_avg, 4), np.around(sims_all/num, decimals=4)

def sim_samples_oneone(paths1, paths2, samples1, mode="jaccard"):
    s_all = 0
    num = 0
    sims_all = np.array([0.0 for _ in range(len(paths1[0]))])
    for s1 in samples1:
        p1 = paths1[s1]
        p2 = paths2[s1]
        s, sims = sim_paths(p1, p2, mode=mode)
#             print(sims)
        s_all += s
        sims_all += np.array(sims)   
        num += 1
    s_avg = s_all / num
    return round(s_avg, 4), np.around(sims_all/num, decimals=4)

def sim_samples_cluster(paths, samples1, path2, mode="jaccard"):
    s_all = 0
    num = 0
    sims_all = np.array([0.0 for _ in range(len(paths[0]))])
    for s1 in samples1:
        p1 = paths[s1]
        p2 = path2
        s, sims = sim_paths(p1, p2, mode=mode)
#             print(sims)
        s_all += s
        num += 1
        sims_all = sims_all + np.array(sims)       
    s_avg = s_all / num
    return round(s_avg, 4), np.around(sims_all/num, decimals=4)

def get_random_samples(samples, num_picked, badboys=[]):
    picked = []
    indexs = []
    while len(picked) < num_picked:
        index = random.randint(0, len(samples)-1)
        if index not in indexs and index not in badboys:
            indexs.append(index)
            picked.append(samples[index])
    return picked


    
# def get_picked_samples_test(dataset="cifar10", arch="alxnet", attack="", attack_epi=0.03):
#     batch_size = 1000
#     samples_class = [[] for c in range(10)]   
#     prefix = dataset if attack == "" else "{}_{}_{}".format(dataset, attack, str(attack_epi))
#     samples_class_file = "cluster_paths/{}_{}_binary_cluster/right_samples_class_test.pkl".format(prefix, arch)

#     if os.path.exists(samples_class_file):
#         with open(samples_class_file, "rb") as f:
#             unpickler = pickle.Unpickler(f)
#             samples_class = unpickler.load()
#     else: 
#         if attack == "":
#             train_data, test_data = Generate_Dataset(dataset)  
#         else:
#         #     this train_data is same with the test_data
#             train_data, test_data = generate_adv_dataset(dataset, arch, attack, attack_epi)
        
#         model = Generate_Model(dataset, arch)
#         model = load_resume_model(model, dataset, arch)
#         model = model.cuda()
#         model.eval()
        
#         train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
#         val_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
#         data_loader = val_loader
#         start_index = end_index = 0
#         for step, (val_x, val_y) in enumerate(data_loader):
#             start_index = end_index
#             print("step:", step)
#             val_x = val_x.cuda()
#             val_y = val_y
#             val_output = model(val_x)
#             _, val_pred_y = val_output.max(1)
#             for i, t in enumerate(val_pred_y):
#                 if attack == "":
#                     if t == val_y[i]:
#                         samples_class[t].append(i+start_index)
#                 else:
#                     if t != val_y[i]:
#                         samples_class[t].append(i+start_index)
                    
#             end_index = start_index + val_x.shape[0]

#         output = open(samples_class_file, 'wb')
#         pickle.dump(samples_class, output)  
#         output.close()
#     print("Done!")
#     return samples_class

# picked_samples_class = get_picked_samples_test(dataset="cifar10", arch="alexnet")
# adv_picked_samples_class = get_picked_samples_test(dataset="cifar10", arch="alexnet", attack="pgd", attack_epi=0.03)


# cluster_paths = [[] for _ in range(10)]
# num_cluster = 4
# for cla in range(10):
#     for clu in range(num_cluster):
#         path_fname = "cluster_paths/cifar10_alexnet_binary_cluster/num_cluster{}_threshold{}_\
#                                   class{}_cluster{}_path.pkl".format(num_cluster, 0.6, cla, clu)
#         with open(path_fname, "rb") as f:
#             unpickler = pickle.Unpickler(f)
#             cluster_paths[cla].append(unpickler.load())
            
            
# def getSims(samples_class, paths):
#     s_all = 0
#     num = 0
#     s_dict = {}
#     num_cluster = 4
#     sims_avg = np.array([0.0 for _ in range(len(paths[0]))])

#     for cla in range(10):
#         print("class:", cla)
#         for index in samples_class[cla]:
#             max_s = 0
#             for clu in range(num_cluster):   
# #                 print(paths[index])
# #                 print(cluster_paths[cla][clu])
#                 s, sims = similarity(paths[index], cluster_paths[cla][clu])
#                 if s > max_s:
#                     max_s = s
#                     max_sims = sims
#             s_all += max_s
#             sims_avg += max_sims
#             num += 1

#     print(s_all/num)
#     print(sims_avg / num)
    
    
# getSims(picked_samples_class, paths)
# getSims(adv_picked_samples_class, adv_paths)

# def get_avg_width(picked_samples_class, paths):
#     ws = []
#     for layer in range(len(paths[0])):
#         w = 0
#         num = 0
#         for cla in range(10):
#             for s in picked_samples_class[cla]:
#                 w += len(paths[s][layer])
#                 num += 1
#         ws.append(w / num)
#     return ws

# print(get_avg_width(picked_samples_class, paths))
# print(get_avg_width(adv_picked_samples_class, adv_paths))