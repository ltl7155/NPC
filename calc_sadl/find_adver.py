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

CLIP_MIN = -0.5
CLIP_MAX = 0.5

import pickle
import  dataloader #import  DatasetAdv, save_score_method

import torch.nn as nn 
import torch.nn.functional as F 
import torch 
from  torchvision.datasets  import utils as dtutil

from models.VGG_16 import VGG16
# from models.vgg import vgg16_bn
from  models_old  import ConvnetMnist as NET_MNIST
from  models_old  import ConvnetCifar as NET_CIFAR10
from  models_old  import VGG16 as NET_VGG_CIFAR10
from models.AlexNet_SVHN import AlexNet

from utils_data import get_model, get_dataset, get_filelist, get_adv_dataset


device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")
    parser.add_argument(
        "--lsa", "-lsa", help="Likelihood-based Surprise Adequacy", action="store_true"
    )
    parser.add_argument(
        "--dsa", "-dsa", help="Distance-based Surprise Adequacy", action="store_true"
    )
    parser.add_argument(
        "--last_layer", action="store_true"
    )
    parser.add_argument(
        "--path", action="store_true"
    )
    parser.add_argument(
        "--target",
        "-target",
        help="Target input set (test or adversarial set)",
        type=str,
        default="fgsm",
    )
    parser.add_argument(
        "--save_path", "-save_path", help="Save path", type=str, default="./feature_maps/"
    )
    parser.add_argument(
        "--batch_size", "-batch_size", help="Batch size", type=int, default=128
    )
    parser.add_argument(
        "--var_threshold",
        "-var_threshold",
        help="Variance threshold",
        type=int,
        default=1e-5,
    )
    parser.add_argument(
        "--upper_bound", "-upper_bound", help="Upper bound", type=int, default=2000
    )
    parser.add_argument(
        "--n_bucket",
        "-n_bucket",
        help="The number of buckets for coverage",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--num_classes",
        "-num_classes",
        help="The number of classes",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--is_classification",
        "-is_classification",
        help="Is classification task",
        type=bool,
        default=True,
    )
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--arch', type=str, default="convmnist")
    parser.add_argument('--dataset', type=str, default="mnist")
    parser.add_argument("--attack_mode",  help="a", type=str, default="pgd")
    parser.add_argument("--attack_epi", type=float, default=0.03)

    args = parser.parse_args()
    print(args)
    args_attack = args.attack_mode
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    x_train = get_dataset(args)
    x_test = get_adv_dataset(args)
    model, threshold, layer_names, num_layer = get_model(args)
    model = model.cuda()
    model = TorchModel(model)
    model.eval()
    
    paths = [[] for _ in range(10)]
    num_cluster = 1
    for cla in range(10):
        for clu in range(num_cluster):
            l = "{}_{}".format(cla, clu)
            picked_samples_fname = "../cluster_paths/{}_binary_cluster/num_cluster{}_threshold{}_class{}_cluster{}_paths.pkl".format(args.arch, num_cluster, threshold,  cla, clu)
            with open(picked_samples_fname, "rb") as f:
                unpickler = pickle.Unpickler(f)
                path = unpickler.load()
            paths[cla].append(path[0])

    file_id_list = get_filelist() 
    results_lsa = {}
    results_dsa = {}   
    
    if args.last_layer:        
        results_lsa[args_attack] = []
        results_dsa[args_attack] = []
        if args.lsa:
            print (f"=======lsa_path======"*10)
            target_name = "test" + str(idx)
            target_lsa = fetch_lsa(model, x_train, x_test, target_name, 
                                   [layer_names[-1]], args, paths, num_layer-1, path=args.path)
#                 print (target_lsa)
            target_cov = get_sc(
                np.amin(target_lsa), 2000 , args.n_bucket, target_lsa
            )            
#             test_lsa_score = np.mean(test_lsa)
            print ("test_lsa_path_score--->", target_cov)
            results_lsa[args_attack].append(round(target_cov, 3))
            save_dir="./"
            save_filename= "lsa_{}_{}.npy".format(args_attack, data_fileid)
            save_filename=os.path.join(save_dir, save_filename)
            
        if args.dsa:
            print ("-=======dsa_path======"*10)
            target_name = args.attack_mode + str(args.attack_epi)
            target_dsa, oods = fetch_dsa(model, x_train, x_test, target_name, 
                                   [layer_names[-1]], args, paths, num_layer-1, path=args.path)
            target_cov = get_sc(
                np.amin(target_dsa), 5.0, args.n_bucket, target_dsa
                )
            results_dsa[args_attack].append(round(target_cov, 3))
            
            suffix = "last_layer_path" if args.path else "last_layer"
            file_name = f"dsa_{args.dataset}_{args.arch}_{args.attack_mode}{args.attack_epi}_{suffix}.p"
            with open(file_name, 'wb') as handle:
                pickle.dump(oods, handle, protocol=pickle.HIGHEST_PROTOCOL)   
#             print(oods)
 
    else:
        results_lsa[args_attack] = []
        results_dsa[args_attack] = []

        if args.lsa:
            print ("-=======lsa_path======"*10)
            key = data_dict["key"]
            x_test=data_dict["your_adv"].to(device)
            x_test.squeeze_(dim=0)
            assert type(key)==list,"we only  get the fisrt key ,if you want batch_size, pls set key map to the lsa_result"
            buckets_every_layer = []
            covered = set()
            target_name = "test" + str(idx)
            for layer in range(num_layer):
                target_lsa = fetch_lsa(model, x_train, x_test, 
                                       target_name, [layer_names[layer]], args, paths, layer, path=args.path)
                lower = np.amin(target_lsa)
                upper = args.upper_bound
                buckets = np.digitize(target_lsa, np.linspace(lower, upper, args.n_bucket))
                buckets_every_layer.append(buckets)
            for i in range(len(buckets_every_layer[0])):
                for layer in range(num_layer):   
                    name = str(buckets_every_layer[layer][i]) + '-'
                covered.add(name)
#                     print(covered)
            target_cov = len(covered) / (float(args.n_bucket)*num_layer) * 100
            results_lsa[args_attack].append(round(target_cov, 3))

        if args.dsa:
            print ("-=======dsa_path======"*10)
            buckets_every_layer = []
            covered = set()
            target_name = args.attack_mode + str(args.attack_epi)
            oods_layer = {}
            for layer in range(num_layer):
                target_dsa, oods = fetch_dsa(model, x_train, x_test, 
                                       target_name, [layer_names[layer]], args, paths, layer, path=args.path)
                
                name = layer_names[layer].replace("/", "_")
                oods_layer[name] = oods
                lower = np.amin(target_dsa)
                upper = 5.0
                buckets = np.digitize(target_dsa, np.linspace(lower, upper, args.n_bucket))
                buckets_every_layer.append(buckets)
            for i in range(len(buckets_every_layer[0])):
                for layer in range(num_layer):   
                    name = str(buckets_every_layer[layer][i]) + '-'
                covered.add(name)
            target_cov = len(covered) / (float(args.n_bucket) * num_layer) * 100
            results_dsa[args_attack].append(round(target_cov, 3))
            print ("test_dsa_path_score--->", target_cov)
            
            suffix = "all_layers_path" if args.path else "all_layers"
            file_name = f"dsa_{args.dataset}_{args.arch}_{args.attack_mode}{args.attack_epi}_{suffix}.p"
            with open(file_name, 'wb') as handle:
                pickle.dump(oods_layer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            

    print("lsa:", results_lsa)
    print("dsa:", results_dsa)
    
    
