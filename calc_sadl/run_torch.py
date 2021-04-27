import numpy as np
import time
import argparse

from tqdm import tqdm
from keras.datasets import mnist, cifar10
#from keras.models import load_model, Model
from sa_torch import fetch_dsa, fetch_lsa, get_sc
from utils_calc import *
from torch_modelas_keras import  TorchModel

CLIP_MIN = -0.5
CLIP_MAX = 0.5


import torch.nn as nn 
import torch.nn.functional as F 
import torch 
from  torchvision.datasets  import utils as dtutil
from utils_data import get_model, get_dataset, get_filelist, get_cluster_para
from  models_old  import ConvnetMnist as NET_MNIST
from  models_old  import ConvnetCifar as NET_CIFAR10
from  models_old  import VGG16 as NET_VGG_CIFAR10

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
        "--target",
        "-target",
        help="Target input set (test or adversarial set)",
        type=str,
        default="fgsm",
    )
    parser.add_argument(
        "--save_path", "-save_path", help="Save path", type=str, default="./tmp/"
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
        "--upper_bound", "-upper_bound", help="Upper bound", type=int, default=500
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
    parser.add_argument("-attack",  help="a", type=str, default="pgd")
    parser.add_argument('--arch', type=str, default="convmnist")
    parser.add_argument('--dataset', type=str, default="mnist")

    args = parser.parse_args()

    x_train = get_dataset(args.dataset)
    ori_model, layer_names, num_layer = get_model(args.dataset, args.arch)
    layer_names = [layer_names[-1]]
    print("layer_names"*10, layer_names)
    ori_model = ori_model.cuda()
    model = TorchModel(ori_model)
    model.eval()
    ori_model.eval()
    
#     sample_threshold, cluster_threshold, cluster_num = get_cluster_para(args.dataset, args.arch)
        #data_fileid ="1Gm926_p5_bvhgfDdlQmmV9lUCmjsF5Ft"
        
    #from dataloader import  DatasetAdv, save_score_method
    import  dataloader #import  DatasetAdv, save_score_method

    file_id_list = get_filelist()
    
    #for  args_attack in  []:
    for  args_attack in  ["pgd","manu_100_nature","manu_100_adv"]:
            
        key = [args.d,args_attack]
        key= ",".join(key)
        if key not in file_id_list:
            print ("not finf",key)
            continue
        assert key in file_id_list,f"expxect the key {key} in {file_id_list}"
        data_fileid = file_id_list[key]
        adv_dt = dataloader.DatasetAdv(file_id_or_local_path=data_fileid)
        
        
        if 1==2:
            map_result= {}
            dl = torch.utils.data.DataLoader(adv_dt,batch_size=1,num_workers=0)
              
            for idx,data_dict in enumerate(dl):
                #x_train=data_dict["your_data"].to(device)
                #x_train.squeeze_(dim=0)
                key = data_dict["key"]
                x_test=data_dict["your_adv"].to(device)
                x_test.squeeze_(dim=0)
                assert type(key)==list,"we only  get the fisrt key ,if you want batch_size, pls set key map to the lsa_result"
                print (x_train.shape,"x_train")
                print (x_test.shape,"x_test")
    #             
    #             exit()
                target_lsa = fetch_lsa(model, x_train, x_test, "test", layer_names, args)
#                 print (target_lsa)
                target_cov = get_sc(
                    np.amin(target_lsa), 2000 , args.n_bucket, target_lsa
                )            
    #             test_lsa_score = np.mean(test_lsa)
                map_result[key[0]]= target_cov
                print ("test_lsa_score--->",target_cov)
                 
            save_dir="./"
            save_filename= "lsa_{}_{}.npy".format(args_attack,data_fileid)
            save_filename=os.path.join(save_dir,save_filename)
            np.save(save_filename,map_result) 
        
        print ("-=======dsa======"*10)
        if 2==2:
            #test_dsa = fetch_dsa(model, x_train, x_test, "test", layer_names, args)
            map_result= {}        
            dl = torch.utils.data.DataLoader(adv_dt,batch_size=1,num_workers=0)
             
            for idx,data_dict in enumerate(dl):
                #x_train=data_dict["your_data"].to(device)
                #x_train.squeeze_(dim=0)
                key = data_dict["key"]
                x_test=data_dict["your_adv"].to(device)
                x_test.squeeze_(dim=0)
                assert type(key)==list,"we only  get the fisrt key ,if you want batch_size, pls set key map to the lsa_result"
     
                 
                target_dsa = fetch_dsa(model, x_train, x_test, "test", layer_names, args)
                target_cov = get_sc(
                    np.amin(target_dsa), 5.0, args.n_bucket, target_dsa
                    )
                 
                print ("test_dsa_score--->",target_cov)
                map_result[key[0]]= target_cov
    
            save_dir="./"
            save_filename= "dsa_{}_{}.npy".format(args_attack,data_fileid)
            save_filename=os.path.join(save_dir,save_filename)
            np.save(save_filename,map_result) 
        
        
#         exit()
    
    
    
    
    
