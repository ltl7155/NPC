import numpy as np
import random
import math
import pickle
import torch
import os

from LSA_DSA_ANPC_lib.get_a_single_path import getPath


PREFIX= "./data"


class Coverager():
    def __init__(self, model, arc, threshold, num_classes=10, num_cluster=10):
        self.model = model
        self.model.eval()
        self.threshold = threshold
        self.paths_all = []
        self.class_num = num_classes
        self.total_cluster = num_cluster
        
        for cla in range(num_classes):
            paths_class = []
            for clu in range(num_cluster):
                picked_samples_fname = "cluster_paths/{}_binary_cluster/num_cluster{}_threshold{}_class{}_cluster{}_paths.pkl".format(arc, num_cluster, threshold, cla, clu)
                picked_samples_fname = os.path.join(PREFIX ,picked_samples_fname,)
                assert os.path.isfile(picked_samples_fname) and os.path.getsize(picked_samples_fname)>0,f"{picked_samples_fname} is empty"
                with open(picked_samples_fname, "rb") as f:
                    unpickler = pickle.Unpickler(f)
                    path = unpickler.load()
                    path = path[0]
                    paths_class.append(path)
            self.paths_all.append(paths_class)
        self.layer_num = len(self.paths_all[0][0])
        
    def random_nn(self):
        layer_num = random.randint(5, 20)
        neuron_in_layer = np.random.choice(128, layer_num, replace=False) + 64
        return neuron_in_layer
    def random_dataset(self, num):
        return np.random.choice(class_num, num)
    def get_rand_path(self, dnn):
        path = []
        for layer_num in dnn:
            width = random.randint(30,70) / 100
            path.append(np.random.choice(layer_num, int(width*layer_num), replace=False))
        return path
    def get_rand_dg(self, dnn):
        abstract_paths = []
        for i in range(class_num):
            abst_path = []
            for j in range(total_cluster):
                abst_path.append(get_rand_path(dnn))
            abstract_paths.append(abst_path)
        return abstract_paths

    def cal_neuron_sim(self, neuron_li_1, neuron_li_2, simi_soft=False):
        intersection = len(neuron_li_1.intersection(neuron_li_2))
        union = (len(neuron_li_1) + len(neuron_li_2)) - intersection
        if simi_soft:
            union = len(neuron_li_1) if len(neuron_li_1) > len(neuron_li_2) else len(neuron_li_2)
        return intersection, union, intersection/union
    
    def cal_path_similaity(self, p1, p2, simi_soft=False):
        lenth = len(p1)
#         print("len p1", len(p1))
#         print("len p2", len(p2))
        sims = []
        path_inter = 0
        path_union = 0
        for i in range(lenth):
            neuron_li_1 = set(p1[i])
            neuron_li_2 = set(p2[i])

            intersection, union, n_sim = self.cal_neuron_sim(neuron_li_1, neuron_li_2, simi_soft=simi_soft)

            path_inter += intersection
            path_union += union

            sims.append(n_sim)
        return sims, path_inter/path_union
#         return sims, np.mean(sims)

    def get_simi(self, X, Y, bucket_m, single_threshold, mode="mix", simi_soft=False, arc=""):
        total = self.class_num * self.total_cluster * bucket_m
        covered = set()
        unit = 1/bucket_m
        
        X = X.cuda()
        batch_size = X.shape[0]
        val_output = self.model(X)
        _, val_pred_y = val_output.max(1)
        path_x = getPath(X, self.model, self.threshold, arc=arc)
        dises = []
        for index in range(batch_size): 
            tmps = []
            label = val_pred_y[index]
            
#             print(label.item())
#             print("y", Y[index])
            if mode == "mix":
                pass
            elif mode == "right":
                if label.item() != Y[index].item():
                    continue
            elif mode == "wrong":
                if label.item() == Y[index].item():
                    continue
            intra_paths = self.paths_all[label]  
            for i, path in enumerate(intra_paths):
                dis = self.cal_path_similaity(path_x[index], path, simi_soft=simi_soft)[1]
#                 print("dis", dis)
                tmps.append(dis)
            dises.append(max(tmps))
        if len(dises) != 0:
            return np.mean(dises)
        else:
            return 0 

    def Intra_NPC(self, X, Y, bucket_m, threshold, mode="mix", simi_soft=False, useOldPaths_X=False, arc=""):
        total = self.class_num * self.total_cluster * bucket_m
        covered = set()
        unit = 1/bucket_m
        
        X = X.cuda()
        batch_size = X.shape[0]
        val_output = self.model(X)
        _, val_pred_y = val_output.max(1)
        
        if useOldPaths_X:
            path_x = self.path_x
        else:
            path_x = getPath(X, self.model, threshold, arc=arc)
            self.path_x = path_x
        
        for index in range(batch_size):  
            label = val_pred_y[index]
            
#             print(label.item())
#             print("y", Y[index])
            if mode == "mix":
                pass
#             elif mode == "right":
#                 if label.item() != Y[index].item():
#                     continue
#             elif mode == "wrong":
#                 if label.item() == Y[index].item():
#                     continue
            intra_paths = self.paths_all[label] 
            
            for i, path in enumerate(intra_paths):
#                 print("single path", path_x[index])
#                 print("cluster path", path)
                dis = self.cal_path_similaity(path_x[index], path, simi_soft=simi_soft)[1]
#                 print(dis)
                bucket_id = math.ceil(dis/unit)
                covered.add(str(label.item())+'-'+str(i)+'-'+str(bucket_id))
        return covered, total


    def Layer_Intra_NPC(self, X, Y, bucket_m, threshold, mode="mix", simi_soft=False, useOldPaths_X=False, arc=""):
        total = self.class_num * self.total_cluster * bucket_m * self.layer_num
        covered = set()
        unit = 1/bucket_m
        
        X = X.cuda()
        batch_size = X.shape[0]
        val_output = self.model(X)
        _, val_pred_y = val_output.max(1)
            
        if useOldPaths_X:
            path_x = self.path_x
        else:
            path_x = getPath(X, self.model, threshold, arc=arc)
        
        for index in range(batch_size): 
            label = val_pred_y[index]
#             print(label.item())
#             print("y", Y[index])
            # if mode == "mix":
                # pass
#             elif mode == "right":
#                 if label.item() != Y[index].item():
#                     continue
#             elif mode == "wrong":
#                 if label.item() == Y[index].item():
#                     continue
            intra_paths = self.paths_all[label] 
            for i, path in enumerate(intra_paths):
                dis = self.cal_path_similaity(path_x[index], path, simi_soft=simi_soft)[0]
                for j, layer_dis in enumerate(dis):
                    bucket_id = math.ceil(layer_dis / unit)
                    covered.add(str(label.item())+'-'+str(i)+'-'+str(j)+'-'+str(bucket_id))
        return covered, total


if __name__ == '__main__':
    #generate a dummy dnn
    dnn = random_nn()
    #generate the decision graph with n abstract paths
    dg = get_rand_dg(dnn)
    #generate test suite
    X = random_dataset(2000)
    bucket = 100
    print("Intra-P-NPC", Intra_NPC(dnn, dg, X, bucket))
    print("Intra-L-NPC", Layer_Intra_NPC(dnn, dg, X, bucket))



