import numpy as np
import sys

sys.path.append("../")

from LRP_path.innvestigator import InnvestigateModel
from LRP_path.inverter_util import Flatten
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
#from models.VGG_16 import VGG16
# from models.vgg import vgg16_bn
#from models.sa_models import ConvnetMnist, ConvnetCifar

import pickle


def getPath(data, model, width_threshold, other_label=False, target=None, arc=""):
    def pick_neurons_layer(relev, threshold=0.8, last=False): 
        if len(relev.shape) != 2:
            rel = torch.sum(relev, [2, 3])
        else: 
            rel = relev
        units_all = []
        rest_all = []
        sec_cri_all = []
        for i in range(rel.shape[0]):
            rel_i = rel[i]
            values, units = torch.topk(rel_i, rel_i.shape[0])    
            sum_value = 0
            tmp_value = 0

            part = 0
            if not last:
                for v in values:
                    if v > 0:
                        sum_value += v
                for i, v in enumerate(values):
                    tmp_value += v
                    if tmp_value >= sum_value * threshold or v <= 0:
                        part = i
                        break   
                units_picked = units[:part+1].tolist()
                rest = units[part+1:].tolist()
                if part * 2 >= len(units):
                    sec_cri = units[part:].tolist()
                else:
                    sec_cri = units[part:part*2].tolist()
            else:
                for v in values:
                    if v < 0:
                        part = i
                units_picked = units[part:].tolist()
                rest = units[:part].tolist()
            units_all.append(units_picked)
            rest_all.append(rest)
            sec_cri_all.append(sec_cri)
        return units_all, rest_all, sec_cri_all


    with torch.no_grad():

        model = model.cuda()
        model.eval()
                    # Convert to innvestigate model
        inn_model = InnvestigateModel(model, lrp_exponent=2,
                                      method="b-rule",
                                      beta=.5)
        

        data = data.cuda()
        if not other_label:
            model_prediction, _ , true_relevance = inn_model.innvestigate(in_tensor=data) 
        else:
            model_prediction, _ , true_relevance = inn_model.innvestigate(in_tensor=data, rel_for_class=other_label) 
#         print(true_relevance)
        relev = true_relevance[::-1]
        if arc == "alexnet":
            relev = relev[:-1]
        if arc == "vgg16_bn":
            relev = relev[:-2]
            
        sample_neurons = {}
        for layer in range(len(relev)):
            r = relev[layer]
            units, _, _ = pick_neurons_layer(r, width_threshold)
            for i in range(len(units)):
                if layer == 0:
                    sample_neurons[i] = []
                sample_neurons[i].append(units[i])     
        return sample_neurons

            
