import numpy as np
import sys
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import time
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data


import pickle

import torch
# from dataloader import DatasetAdv
from deephunter.datasets import manu_datasets_reader as dataloader  
from deephunter.models import get_net  
from data  import manual_seed_RQ3  as gdrive_fileids 

from SNPC.get_a_single_path import getPath
from SNPC.neuron_coverage import Coverager
from LSA_DSA_ANPC_lib import utils_data as calc_sadl_utils_data
# convmnist 0.8, 4, 0.8
# convcifar 0.7 7 0.9
# vgg 0.9 7 0.9
torch.random.manual_seed(123)

num_classes = 10
bucket_m = 100
results = {}
results_layer = {}

# s = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time()))

BATCH_SIZE = os.environ.get("batch_size",128)
BATCH_SIZE = int(BATCH_SIZE)

    
device= torch.device("cuda")
    
def load_npc_lnpc(
        fileid,
        ) :
    file_info = gdrive_fileids.rq3_fileid_list[args_fid]

    model_name = file_info["arch"]
    mode = file_info["attack"]
    dataset = file_info ["dataset"]


    sample_threshold, cluster_threshold, cluster_num = calc_sadl_utils_data.get_cluster_para(dataset,
                                                                                              model_name)
    #print(model_name, mode)
    intra = {}
    layer_intra = {}
    m_name = f"{model_name},{mode}"

    # models[model_name]
    nn_model = get_net(name=model_name)
    # test_set = DatasetAdv(file_id_list[m_name])
    test_set = dataloader.get_dataloader( fileid )
    
    fetch_func = lambda x:x["your_adv"]

    time_collect_list = []
    
    for index, datax in enumerate(test_set):
        covered_10 = set()
        covered_100 = set() 
        total_100 = total_10 = 0
        
        keys = datax["key"]
        # print("keys:", keys)

        x_test = datax["your_adv"]
        y_test = torch.rand((x_test.shape[0]))
        #print ("x_test:",x_test.mean(),x_test.std(),"y_test:",y_test.mean(),y_test.std())

        test_loader1=torch.torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(x_test, y_test),
                batch_size=BATCH_SIZE)
        
        for step, (x, y) in enumerate(test_loader1):
            # print("step", step)
            x = x.to(device)
            
            # models[model_name] = models[model_name].to(device)
            # cover = Coverager(models[model_name], model_name, cluster_threshold, num_classes=num_classes, num_cluster=cluster_num)
            cover = Coverager(nn_model, model_name, cluster_threshold, num_classes=num_classes, num_cluster=cluster_num)
            #print ("model_name",model_name,"cluster_threshold",cluster_threshold,"num_classes",num_classes,"clust",cluster_num)
            
            start_time1= time.time()
            covered1, total1 = cover.Intra_NPC(x, y, bucket_m, sample_threshold, mode=mode, simi_soft=False, arc=model_name)
            start_time2= time.time()
            covered2, total2 = cover.Layer_Intra_NPC(x, y, bucket_m, sample_threshold, mode=mode, simi_soft=False, useOldPaths_X=True, arc=model_name)
            start_time3= time.time()
            
            total_10 += total1
            total_100 += total2    
            covered_10 = covered_10 | covered1
            covered_100 = covered_100 | covered2 
            time_collect_list.append((start_time1 , start_time2 , start_time3 ))

#                     print(cover.get_simi(x, y, bucket_m, single_threshold, mode=mode, simi_soft=False))
        intra[keys] = round(len(covered_10) / total1, 5)
        
        layer_intra[keys] = round(len(covered_100) / total2, 5)
        #print(m_name, intra[keys],"<---intra")
        #print(m_name, layer_intra[keys],"<---layer_intra")
    return intra,layer_intra,time_collect_list


if __name__=="__main__":    
    
    parser = argparse.ArgumentParser()

    parser.add_argument("-fid",  help="file", type=str, default='1v_YWYe7s2MlL_ZyRnixlxe1xdOIDXwdL')
    parser.add_argument("-fr",action="store_true")

    args = parser.parse_args()

    args_fid= args.fid 
    assert args_fid is not None ,"expect a fileid"

    file_info = gdrive_fileids.rq3_fileid_list[args_fid]

    # model_name = file_info["arch"]
    attack_name = file_info["attack"]
    # dataset = file_info ["dataset"]

    #os.makedirs("./",exist_ok=True)
    save_filename_10 = "./Our10_{}_{}.npy".format(attack_name, args_fid)
    save_filename_100 = "./Our100_{}_{}.npy".format(attack_name, args_fid)

    if os.path.isfile(save_filename_10) and not args.fr:
        print ("exist,", save_filename_10)
        exit()

    intra,layer_intra,timecost= load_npc_lnpc(args_fid)
    
    timecost =[(y-x,z-y) for x,y,z in timecost]
    timecost_our10 =float( sum([x for x,_ in timecost]) )
    timecost_our100 =float( sum([x for _,x in timecost]) )
    
    np.save(save_filename_10, intra) 
    np.save(save_filename_100, layer_intra) 

    np.save(save_filename_10.replace(".npy",".time.log"), timecost_our10) 
    np.save(save_filename_100.replace(".npy",".time.log"), timecost_our100) 


    print(intra)
    print(layer_intra)

# for model_name in ["vgg16_bn"]:
    # for mode in ["manu_100_nature", "manu_100_adv"]:


        # save_filename_10 = "./coverage_results/{}_Our_10_{}".format(mode, m_name)
        # save_filename_100 = "./coverage_results/{}_Our_100_{}".format(mode, m_name)
        # np.save(save_filename_10, intra) 
        # np.save(save_filename_100, layer_intra) 
        #
            # end_time = start_time 
            # start_time = time.time()
            # np.save(save_filename.replace(".npy",".time.log"), start_time-end_time) 
            #
            #
        # results[m_name] = intra
        # results_layer[m_name] = layer_intra
# print(results)
# print(results_layer)
# print("start time:", s)
# print("end time:", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time())) )