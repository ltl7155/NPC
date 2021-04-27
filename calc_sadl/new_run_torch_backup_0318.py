import numpy as np
import time
import argparse

from tqdm import tqdm
from keras.datasets import mnist, cifar10
import sys
sys.path.append("..")
#from keras.models import load_model, Model
from new_sa_torch import fetch_dsa, fetch_lsa, get_sc, fetch_newMetric
from utils_calc import *
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
from utils_data import get_model, get_dataset, get_filelist, get_cluster_para


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
        "--nma", "-nma", help="NM Adequacy", action="store_true"
    )
    parser.add_argument(
        "--last_layer", action="store_true"
    )
    parser.add_argument(
        "--path", action="store_true"
    )
    parser.add_argument(
        "--rest", action="store_true"
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
        default=200,
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
    parser.add_argument("-attack",  help="a", type=str, default="pgd")
    

    args = parser.parse_args()
    print(args)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    x_train = get_dataset(args.dataset)
    ori_model, layer_names, num_layer = get_model(args.dataset, args.arch)
    ori_model = ori_model.cuda()
    model = TorchModel(ori_model)
    model.eval()
    ori_model.eval()
    
    sample_threshold, cluster_threshold, cluster_num = get_cluster_para(args.dataset, args.arch)
    
    cluster_paths = [[] for _ in range(10)]
    samples_clusters = [[] for _ in range(10)]
    num_cluster = cluster_num
    
    for cla in range(10):
        for clu in range(num_cluster):
            l = "{}_{}".format(cla, clu)
            picked_samples_fname = "../cluster_paths/{}_binary_cluster/num_cluster{}_threshold{}_class{}_cluster{}_paths.pkl".format(args.arch, num_cluster, cluster_threshold,  cla, clu)
            with open(picked_samples_fname, "rb") as f:
                unpickler = pickle.Unpickler(f)
                path = unpickler.load()
            cluster_paths[cla].append(path[0])
            
    for cla in range(10):
        for clu in range(num_cluster):
            l = "{}_{}".format(cla, clu)
            picked_samples_fname = "../cluster_paths/{}_binary_cluster/num_cluster{}_class{}_cluster{}_picked_samples.pkl".format(args.arch, num_cluster, cla, clu)
            with open(picked_samples_fname, "rb") as f:
                unpickler = pickle.Unpickler(f)
                samples = unpickler.load()
            samples_clusters[cla].append(samples)
    

    file_id_list = get_filelist()
    
    results_lsa = {}
    results_dsa = {}
    results_nma = {}
    
    if args.lsa:
        args.save_path = "./feature_maps_lsa/"
    elif args.dsa:
        args.save_path = "./feature_maps_dsa/"
    elif args.nma:
        args.save_path = "./feature_maps_nm/"
    
    if args.last_layer:   
        n_bucket = args.n_bucket * num_cluster
#         n_bucket = args.n_bucket
        for  args_attack in  [ "manu_100_adv", "manu_100_nature"]:
            results_lsa[args_attack] = []
            results_dsa[args_attack] = []
            key = [args.d, args_attack]
            key= ",".join(key)
            if key not in file_id_list:
                print ("not finf",key)
                continue
            assert key in file_id_list, f"expxect the key {key} in {file_id_list}"
            data_fileid = file_id_list[key]
            setattr(args,"fileid",data_fileid)
            adv_dt = dataloader.DatasetAdv(file_id_or_local_path=data_fileid)

            if args.lsa:
                print (f"=======lsa_path======"*10)
                map_result= {}
                dl = torch.utils.data.DataLoader(adv_dt, batch_size=1, num_workers=0)
                
                for idx, data_dict in enumerate(dl):
                    key = data_dict["key"]
                    x_test=data_dict["your_adv"].to(device)

                    x_test.squeeze_(dim=0)
                    assert type(key)==list,"we only  get the first key ,if you want batch_size, pls set key map to the lsa_result"
                    target_name =  args_attack + "_test" + str(idx)
                    target_lsa = fetch_lsa(model, x_train, x_test, target_name, 
                                           [layer_names[-1]], args, cluster_paths, num_layer-1, path=args.path)
    #                 print (target_lsa)
                    target_cov, buckets = get_sc(
                        np.amin(target_lsa), 2000 , n_bucket, target_lsa
                    )            
        #             test_lsa_score = np.mean(test_lsa)
                    map_result[key[0]]= target_cov
                    print ("test_lsa_path_score--->", target_cov)
                    results_lsa[args_attack].append(round(target_cov, 3))
                save_dir="./"
                save_filename= "lsa_{}_{}.npy".format(args_attack, data_fileid)
                save_filename=os.path.join(save_dir, save_filename)
                np.save(save_filename, map_result) 

            if args.dsa:
                print ("-=======dsa_path======"*10)
                #test_dsa = fetch_dsa(model, x_train, x_test, "test", layer_names, args)
                map_result= {}  
#                 map_buckets = {}
                dl = torch.utils.data.DataLoader(adv_dt, batch_size=1, num_workers=0)

                for idx, data_dict in enumerate(dl):
#                     x_train=data_dict["your_data"].to(device)
#                     x_train.squeeze_(dim=0)
                    key = data_dict["key"]
                    x_test=data_dict["your_adv"].to(device)
                    x_test.squeeze_(dim=0)
                    assert type(key)==list,"we only  get the first key ,if you want batch_size, pls set key map to the lsa_result"
                    
                    target_name =  args_attack + "_test" + str(idx)
                    target_dsa, _, a_dists, b_dists = fetch_dsa(model, x_train, x_test, target_name, 
                                           [layer_names[-1]], args, cluster_paths, num_layer-1, path=args.path)
                    print("max a_dists", max(a_dists))
                    print("min b_dists", min(b_dists))
                    print("max target_dsa", max(target_dsa))
                    target_cov, buckets  = get_sc(
                        np.amin(target_dsa), 5.0, n_bucket, target_dsa
                        )
#                     target_cov, buckets = get_sc(
#                         0.0, 5.0, args.n_bucket, target_dsa
#                         )
                    results_dsa[args_attack].append(round(target_cov, 3))
                    print ("test_dsa_path_score--->", target_cov)                
                    map_result[key[0]]= target_cov
#                     map_buckets[key[0]]= buckets
                    save_buckets_name= "buckets/dsa_buckets_{}_{}_{}.npy".format(args.dataset, args.arch, target_name)
                    np.save(save_buckets_name, buckets) 
                    
                save_dir="./"
                #save_filename= "dsa_{}_{}.npy".format(args.dataset, args.arch)
                save_filename= "dsa_{}_{}.npy".format(args_attack, data_fileid)
                save_filename=os.path.join(save_dir, save_filename)
                np.save(save_filename, map_result)    
#         exit()
    
    else:
        for  args_attack in  [ "manu_100_adv", "manu_100_nature"]:
            key = [args.d, args_attack]
            key= ",".join(key)
            results_lsa[args_attack] = []
            results_dsa[args_attack] = []
            results_nma[args_attack] = []
            if key not in file_id_list:
                print ("not finf",key)
                continue
            assert key in file_id_list, f"expxect the key {key} in {file_id_list}"
            data_fileid = file_id_list[key]
            setattr(args,"fileid",data_fileid)
            adv_dt = dataloader.DatasetAdv(file_id_or_local_path=data_fileid)

            if args.lsa:
                print ("-=======lsa_path======"*10)
                map_result= {}
                dl = torch.utils.data.DataLoader(adv_dt, batch_size=1, num_workers=0)
                for idx, data_dict in enumerate(dl):
                    key = data_dict["key"]
                    x_test=data_dict["your_adv"].to(device)
                    
                    x_test.squeeze_(dim=0)
                    assert type(key)==list,"we only  get the fisrt key ,if you want batch_size, pls set key map to the lsa_result"
                    buckets_every_layer = []
                    covered = set()
                    target_name =  args_attack + "_test" + str(idx)
                    for layer in range(num_layer):
                        target_lsa = fetch_lsa(model, x_train, x_test, 
                                               target_name, [layer_names[layer]], args, cluster_paths, layer, path=args.path)
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
                    map_result[key[0]]= target_cov
                    results_lsa[args_attack].append(round(target_cov, 3))
                    print ("test_lsa_path_score--->", target_cov)
                save_dir="./"
                save_filename= "lsa_{}_{}.npy".format(args_attack, data_fileid)
                save_filename=os.path.join(save_dir, save_filename)
                np.save(save_filename,map_result) 

            if args.dsa:
                print ("-=======dsa_path======"*10)
                #test_dsa = fetch_dsa(model, x_train, x_test, "test", layer_names, args)
                map_result= {}        
                dl = torch.utils.data.DataLoader(adv_dt, batch_size=1, num_workers=0)

                for idx, data_dict in enumerate(dl):
                    key = data_dict["key"]
                    x_test=data_dict["your_adv"].to(device)
                    x_test.squeeze_(dim=0)
                    assert type(key)==list,"we only  get the first key ,if you want batch_size, pls set key map to the lsa_result" 
                    buckets_every_layer = []
                    covered = set()
                    target_name =  args_attack + "_test" + str(idx)
                    for layer in range(num_layer):
                        target_dsa = fetch_dsa(model, x_train, x_test, 
                                               target_name, [layer_names[layer]], args, cluster_paths, layer, path=args.path)
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
                    map_result[key[0]]= target_cov

                save_dir="./"
                save_filename= "dsa_{}_{}.npy".format(args_attack, data_fileid)
                save_filename=os.path.join(save_dir, save_filename)
                np.save(save_filename, map_result) 
                
            if args.nma:
                ts = []
                print ("-=======nma_path======"*10)
                #test_dsa = fetch_dsa(model, x_train, x_test, "test", layer_names, args)
                map_result= {}        
                dl = torch.utils.data.DataLoader(adv_dt, batch_size=1, num_workers=0)

                for idx, data_dict in enumerate(dl):
                    
                    key = data_dict["key"]
                    print(f"-----{key[0]}-----"*4)
                    x_test=data_dict["your_adv"].to(device)
                    y_test = data_dict["your_label"]
                    x_test.squeeze_(dim=0)
                    y_test.squeeze_(dim=0)
                    assert type(key)==list,"we only  get the first key ,if you want batch_size, pls set key map to the nma_result" 
                    buckets_every_layer = []
                    covered = set()
                    target_name =  args_attack + "_test" + str(idx) + args.fileid
                    temp_t = 0
                    for layer in range(num_layer):
                        print("for layer " + layer_names[layer])
                        
                        target_nma, clu_all, cla_all, t = fetch_newMetric(model, ori_model, x_train, x_test, y_test, 
                                                                       sample_threshold,target_name, [layer_names[layer]], 
                                                                       args, cluster_paths, layer, samples_clusters,
                                                                       fakePath=False, rest=args.rest)
#                         print(target_nma)
                        
                        if args.dataset == "mnist":
                            upper = [2.0, 2.0, 2.0]
#                             upper = [1, 1, 12.0]
                        else:
                            upper = [2.0 for i in range(num_layer)] 
                        print("max:", max(target_nma))
                        n_bucket = args.n_bucket
#                         n_bucket = int(args.n_bucket/num_cluster)
#                         buckets = np.digitize(target_nma, np.linspace(np.amin(target_nma), upper[layer], args.n_bucket))
                        buckets = np.digitize(target_nma, np.linspace(np.amin(target_nma), upper[layer], n_bucket))
                        new_buckets = [str(cla_all[i]) + "_" + str(clu_all[i]) + "_" + str(buckets[i]) for i in range(len(buckets))]
                        buckets_every_layer.append(new_buckets)
                        temp_t += t
            
                    ts.append(temp_t)
                    print("time:", ts)
                    for i in range(len(buckets_every_layer[0])):
                        for layer in range(num_layer):   
                            name = str(buckets_every_layer[layer][i]) + '-'
                        covered.add(name)
                    target_cov = len(covered) / (float(n_bucket) * num_layer * num_cluster * 10) * 100 
                    results_nma[args_attack].append(round(target_cov, 3))
                    print ("test_nma_path_score--->", target_cov)
                    map_result[key[0]]= target_cov
                    
                
                print(map_result)
                print("time:", ts)
                save_dir="./"
                save_filename= "nma_{}_{}.npy".format(args_attack, data_fileid)
                save_filename=os.path.join(save_dir, save_filename)
                np.save(save_filename, map_result) 
        
    print("lsa:", results_lsa)
    print("dsa:", results_dsa)
    print("nma:", results_nma)
    print("end time:", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time())) )
    
    
