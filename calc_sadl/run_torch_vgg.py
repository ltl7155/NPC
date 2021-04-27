import numpy as np
import time
import argparse

from tqdm import tqdm
from keras.datasets import mnist, cifar10
#from keras.models import load_model, Model
from sa_torch import fetch_dsa, fetch_lsa, get_sc
from utils import *
from torch_modelas_keras import  TorchModel

CLIP_MIN = -0.5
CLIP_MAX = 0.5


import torch.nn as nn 
import torch.nn.functional as F 
import torch 
from  torchvision.datasets  import utils as dtutil

from  models  import ConvnetMnist as NET_MNIST
from  models  import ConvnetCifar as NET_CIFAR10
from  models  import VGG16 as NET_VGG_CIFAR10

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
    parser.add_argument("-attack",  help="a", type=str, default="pgd")
    parser.add_argument("-layidx",  help="0,1,2,3", type=int, default=0)

    args = parser.parse_args()
    assert args.d in ["mnist", "cifar","cifar_vgg"], "Dataset should be either 'mnist' or 'cifar'"
    assert args.lsa ^ args.dsa, "Select either 'lsa' or 'dsa'"
    print(args)

# dict_keys(['0/relu1', '1/relu2', '2/relu3', '3/relu4', '4/relu5', '5/relu6', '6/relu7', '7/relu8'])

    if args.d == "cifar_vgg":
        model = NET_VGG_CIFAR10() 
        model = model.to(device)

        file_id="1Ys3-0QuxN6tbzcAh_nlNHJcrX6pVc-r2"

        dtutil.download_file_from_google_drive(file_id=file_id, root="./model/", filename=f"torch_{file_id}.pth")
        model.load_state_dict(torch.load(f"./model/torch_{file_id}.pth"))

        model = TorchModel(net=model)

        layer_names_list = ['6/relu7', '7/relu8', '8/relu9', '9/relu10', '10/relu11', '11/relu12', '12/relu13'][::-1]
        
        layer_names = [layer_names_list[args.layidx]]
        print ("layer_names--->",layer_names)
        #data_fileid ="1Gm926_p5_bvhgfDdlQmmV9lUCmjsF5Ft"
        
    assert args.d == "cifar_vgg"
    #from dataloader import  DatasetAdv, save_score_method
    import  dataloader #import  DatasetAdv, save_score_method
    file_id_list= {
        "mnist,cw":"1aRN20FXhxvWqsIQTdDSNwei_jYSx5S7a",
        "mnist,pgd":"1hc_aj908k7_Zs2L4TsaWYENG-GwtdJe2",
        "cifar,cw":"1MrRngrHDuSm2fEd044mNktkZtDSM33U4",
        "cifar,pgd":"1nhWO0VT131_9e5ubgzs343EhM9Ru0UvY",
        "cifar_vgg,cw":"1Gm926_p5_bvhgfDdlQmmV9lUCmjsF5Ft",
        "cifar_vgg,pgd":"1-X1d-qaYGpUacej9McI5DOq20dDOk9ia",
        }

    key = [args.d,args.attack]
    key= ",".join(key)
    assert key in file_id_list,f"expxect the key {key} in {file_id_list}"
    data_fileid = file_id_list[key]
    adv_dt = dataloader.DatasetAdv(file_id_or_local_path=data_fileid)
    
    
    if 1==1:
        map_result= {}
        dl = torch.utils.data.DataLoader(adv_dt,batch_size=1,num_workers=0)
        
        for idx,data_dict in enumerate(dl):
            x_train=data_dict["your_data"].to(device)
            x_train.squeeze_(dim=0)
            key = data_dict["key"]
            x_test=data_dict["your_adv"].to(device)
            x_test.squeeze_(dim=0)
            assert type(key)==list,"we only  get the fisrt key ,if you want batch_size, pls set key map to the lsa_result"
#             print (x_train.shape,"x_train")
#             print (x_test.shape,"x_train",key)
#             
#             exit()
            target_lsa = fetch_lsa(model, x_train, x_test, "test", layer_names, args)
            
            target_cov = get_sc(
                np.amin(target_lsa), args.upper_bound, args.n_bucket, target_lsa
            )            
#             test_lsa_score = np.mean(test_lsa)
            map_result[key[0]]= target_cov
#             print ("test_lsa_score--->",target_cov)
            
        save_dir="./"
        save_filename= "lsa_{}_{}_lyidx{}.npy".format(args.attack,data_fileid,args.layidx)
        save_filename=os.path.join(save_dir,save_filename)
        np.save(save_filename,map_result) 
    
        
    if 2==2:
        #test_dsa = fetch_dsa(model, x_train, x_test, "test", layer_names, args)
        map_result= {}        
        dl = torch.utils.data.DataLoader(adv_dt,batch_size=1,num_workers=0)
        
        for idx,data_dict in enumerate(dl):
            x_train=data_dict["your_data"].to(device)
            x_train.squeeze_(dim=0)
            key = data_dict["key"]
            x_test=data_dict["your_adv"].to(device)
            x_test.squeeze_(dim=0)
            assert type(key)==list,"we only  get the fisrt key ,if you want batch_size, pls set key map to the lsa_result"

            
            target_dsa = fetch_dsa(model, x_train, x_test, "test", layer_names, args)
            target_cov = get_sc(
                np.amin(target_dsa), args.upper_bound, args.n_bucket, target_dsa
                )
            
            map_result[key[0]]= target_cov

        save_dir="./"
        save_filename= "dsa_{}_{}_lyidx{}.npy".format(args.attack,data_fileid,args.layidx)
        save_filename=os.path.join(save_dir,save_filename)
        np.save(save_filename,map_result) 
    
    
    
    
    
    
    