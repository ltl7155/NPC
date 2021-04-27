

import torch
import numpy as np

def mtest(model, dataloader):
    model.eval()
    total = 0
    correct = 0
    right_conf = 0.0
    wrong_conf = 0.0
    right_index = []
    wrong_index = []
    to_np = lambda x: x.data.cpu().numpy()
    offset = 0
    with torch.no_grad():
        for (inputs, labels) in dataloader:
            total += labels.size(0)
            inputs = inputs.cuda()
            labels = labels.cuda()
            scores, _ = model(inputs)
            msps, preds = torch.max(torch.softmax(scores.data,1),1)
            correct += (preds == labels).sum().item()
            right_conf += msps[preds == labels].sum().item()
            wrong_conf += msps[preds != labels].sum().item()
            preds = to_np(preds)
            labels = to_np(labels)
            right_index.append(np.where(labels==preds)[0]+offset)
            wrong_index.append(np.where(labels!=preds)[0]+offset)
            offset += len(labels)
            
    wrong = total - correct
    acc = np.round(100 * correct / total, 2)
    c_conf = np.round((100 * right_conf) / correct , 2) if correct != 0 else -1
    w_conf = np.round((100 * wrong_conf) / wrong , 2) if wrong != 0 else -1
    right_index = np.concatenate(right_index,axis=0)
    wrong_index = np.concatenate(wrong_index,axis=0)
    return acc, c_conf, w_conf, right_index, wrong_index

from torch.utils.data import DataLoader
# from data.UserDataset import UserDataset
# from mutators import Mutators
def generate_mutation(samples, labels, test_transform, net, img_shape=(32,32), batch_num=256):
    mutated_samples = []
    ground_truth = []
    dataset_train = UserDataset(samples ,img_shape, labels = labels, transform = test_transform)
    dataloader  = DataLoader(dataset_train, batch_size=batch_num, num_workers=4, pin_memory=True)
    acc, c_conf, w_conf, right_index, wrong_index = mtest( net, dataloader )
    print("[Initial] Acc: {}({}/{})\tC_conf:{}\tW_conf:{}".format(acc,len(right_index),len(samples),c_conf, w_conf))
    
    mutated_samples.append(samples[wrong_index])
    ground_truth.append(labels[wrong_index])
    count = len(wrong_index)
    total = len(samples)
    samples, labels = samples[right_index], labels[right_index]
    print("Start mutation process...")
    print(">>> Add {} natural adversarial examples({}/{})".format(count, count, total))
    iteration = 0
    mutation_time=3
    while count < total and iteration < 5000:

        if iteration > 2000:
            mutation_time=4
        cur_mutated = []
        for sample in samples:
            # "translation":0,"scale":1,"shear":2,"rotation":3,"pixel_change":4,"noise":5,"contrast":6,"brightness":7,"blur":8
            mutation = Mutators.mutate(sample, mutation_name=None, mutation_time=mutation_time ,mutation_type=0)
            cur_mutated.append(mutation)
        cur_mutated = np.array(cur_mutated)
        
        dataset_train = UserDataset(cur_mutated, img_shape, labels, test_transform)
        dataloader  = DataLoader(dataset_train, batch_num, num_workers=4, )
        acc, c_conf, w_conf, right_index, wrong_index = mtest( net, dataloader )
        if len(wrong_index) == 0:
            print("Iteration {}: No update({}/{},m:{})".format(iteration,count,total,mutation_time))
            if len(right_index) == 0:
                print("Iteration {}:Finished!".format(iteration))
        else:
            mutated_samples.append(cur_mutated[wrong_index])
            ground_truth.append(labels[wrong_index])
            new_add = len(wrong_index)
            count += new_add
            print("Iteration {}: Add {} new mutations({}/{},m:{})".format(iteration,new_add, count, total, mutation_time))
            if len(right_index) == 0:
                print("Iteration {}:Finished !".format(iteration))
            else:
                samples, labels = samples[right_index], labels[right_index]
        iteration += 1
    return np.concatenate(mutated_samples,axis=0),np.concatenate(ground_truth,axis=0)

from tqdm import tqdm
from foolbox.attacks import LinfFastGradientAttack,LinfDeepFoolAttack
from foolbox.attacks import LinfProjectedGradientDescentAttack
from foolbox.attacks import SaltAndPepperNoiseAttack
from foolbox.attacks import L2CarliniWagnerAttack
# from data.UserDataset import UserDataset
from foolbox.models import PyTorchModel
torch.backends.cudnn.benchmark = True
import os
def generate_adv(images, labels, model_path, save_path, _attack="fgsm", dataset = "cifar10"):
    if dataset == "cifar10":
        adv_transform = T.Compose([T.ToTensor(),])
        net = ResNet18(num_classes=10).cuda()
        net.load_state_dict(torch.load(model_path)['state_dict_backbone'])
        net.eval()
        cifar10_mean = [0.4914, 0.4822, 0.4465] 
        cifar10_std = [0.247, 0.2435, 0.2616]  
        preprocessing = dict(mean=cifar10_mean, std=cifar10_std, axis=-3)
        bounds = (0,1)
        batch_size = 256
        labels = labels.squeeze()
        dataset = UserDataset(images,(32,32), labels, adv_transform)
        dataloader  = DataLoader(dataset, batch_size,
                                num_workers=4, pin_memory=True)
    else:
        assert False,dataset
    fmodel = PyTorchModel(net, bounds=bounds, num_classes = 10,
                        preprocessing=preprocessing)


    if attack == 'fgsm':
        attack  = LinfFastGradientAttack(fmodel)
    elif attack == "pgd":
        attack = LinfProjectedGradientDescentAttack(fmodel)
    elif attack == 'deepfool':
        attack = LinfDeepFoolAttack(fmodel)
    elif attack == 'cw':
        attack = L2CarliniWagnerAttack(fmodel)
        
    results = []
    ground_truth = []
    for samples,labels in tqdm(dataloader):
        samples,labels = samples.numpy(),labels.numpy()
        if attack == 'fgsm':
            adversarials = attack(samples, labels, [0.01,0.05,0.1])
        else:
            adversarials = attack(samples, labels)
        fail_idx = np.where(np.isnan(adversarials[:,0,0,0]))[0]
        success_idx = [i for i in range(len(samples)) if i not in fail_idx]
        success_idx = np.array(success_idx)
        results.append(adversarials[success_idx])
        ground_truth.append(labels[success_idx])
    results = np.concatenate(results,axis=0)
    ground_truth = np.concatenate(ground_truth, axis=0)
    success = len(results)
    total = len(dataloader.dataset)
    asr = np.round(success*100/total,2)
    print("[{}]ASR:{}({}/{})".format(_attack,asr,success,total))
    advs = np.uint8(results.transpose(0,2,3,1)*255)


    np.save(os.path.join(save_path,"adv_{}_{}_samples.npy".format(_attack,dataset)),advs)
    np.save(os.path.join(save_path,"adv_{}_{}_labels.npy".format(_attack,dataset)),ground_truth)
    print("[{},{}]Save samples and labels successfully!".format(attack,dataset))
    # return results,ground_truth
    
def generate_adv_version2(dataloader, net, save_path, _attack="fgsm", dataset="cifar10", epsilon=0.03, arc="vgg"):
    bounds = (0,1)
    
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225] 
    preprocessing = dict(mean=imagenet_mean, std=imagenet_std, axis=-3)
    
    if dataset == "imagenet":
        fmodel = PyTorchModel(net, bounds=bounds, preprocessing=preprocessing)
    else:
        fmodel = PyTorchModel(net, bounds=bounds)
    
    epsilons = [epsilon]
    
#     print(fmodel)
    
    if _attack == 'fgsm':
        attack  = LinfFastGradientAttack()
    elif _attack == "pgd":
        attack = LinfProjectedGradientDescentAttack()
    elif _attack == 'deepfool':
        attack = LinfPGD()
    elif _attack == 'cw':
        attack = L2CarliniWagnerAttack()
    
    results = []
    s = 0
    ground_truth = []
    for samples, labels in tqdm(dataloader):
        labels_cuda = labels
        samples, labels_cuda = samples.cuda(), labels_cuda.cuda()
        if _attack == 'cw':
            adversarials, _, success = attack(fmodel, samples, labels_cuda, epsilons=None)
        else:
            adversarials, _, success = attack(fmodel, samples, labels_cuda, epsilons=epsilons)
        adversarials = adversarials[0].cpu().numpy()
        results.append(adversarials)
        ground_truth.append(labels)
        s += success.sum().item()
        
    results = np.concatenate(results, axis=0)
#     ground_truth = ground_truth.cpu().numpy()
    ground_truth = np.concatenate(ground_truth, axis=0)
    success = s
    total = len(dataloader.dataset)
    asr = np.round(success*100/total, 2)
    print("[{}]ASR:{}({}/{})".format(_attack,asr,success,total))
    if _attack == "cw":
        advs = np.uint8(results.transpose(1,2,0)*255)
    else:
        advs = np.uint8(results.transpose(0,2,3,1)*255)
    

    if save_path != "":
        np.save(os.path.join(save_path,"adv_{}_{}_{}_samples_eps{}.npy".format(_attack, dataset, arc, epsilon)), advs)
        np.save(os.path.join(save_path,"adv_{}_{}_{}_labels_eps{}.npy".format(_attack, dataset, arc, epsilon)), ground_truth)
    print("[{},{}]Save samples and labels successfully!".format(attack, dataset))

# def generate_adv_version2(dataloader, net, save_path, preprocessing, _attack="fgsm", dataset="cifar10", epsilon=0.03, arch="vgg"):
#     bounds = (0, 1)  
#     fmodel = PyTorchModel(net, bounds=bounds, preprocessing=preprocessing)  
#     epsilons = [epsilon]
    
# #     print(fmodel)
    
#     if _attack == 'fgsm':
#         attack  = LinfFastGradientAttack()
#     elif _attack == "pgd":
#         attack = LinfProjectedGradientDescentAttack()
#     elif _attack == 'deepfool':
#         attack = LinfDeepFoolAttack()
#     elif _attack == 'cw':
#         attack = L2CarliniWagnerAttack()
    
#     results = []
#     s = 0
#     ground_truth = []
#     for samples, labels in tqdm(dataloader):
#         labels_cuda = labels
#         samples, labels_cuda = samples.cuda(), labels_cuda.cuda()
#         if _attack == 'cw':
#             adversarials, _, success = attack(fmodel, samples, labels_cuda, epsilons=None)
#         else:
#             adversarials, _, success = attack(fmodel, samples, labels_cuda, epsilons=epsilons)
#         adversarials = adversarials[0].cpu().numpy()
#         results.append(adversarials)
#         ground_truth.append(labels)
#         s += success.sum().item()
        
#     results = np.concatenate(results, axis=0)
# #     ground_truth = ground_truth.cpu().numpy()
#     ground_truth = np.concatenate(ground_truth, axis=0)
#     success = s
#     total = len(dataloader.dataset)
#     asr = np.round(success*100/total, 2)
#     print("[{}]ASR:{}({}/{})".format(_attack, asr, success, total))
#     if _attack == "cw":
#         advs = np.uint8(results.transpose(1,2,0)*255)
#     else:
#         advs = np.uint8(results.transpose(0,2,3,1)*255)
    
#     if save_path != "":
#         np.save(os.path.join(save_path,"adv_{}_{}_{}_samples_eps{}_{}.npy".format(_attack, 
#                                                                                   dataset, arch, epsilon, asr)), asr)
#         np.save(os.path.join(save_path,"adv_{}_{}_{}_samples_eps{}.npy".format(_attack, 
#                                                                                   dataset, arch, epsilon)), advs)
#         np.save(os.path.join(save_path,"adv_{}_{}_{}_labels_eps{}.npy".format(_attack, 
#                                                                                  dataset, arch, epsilon)), ground_truth)
#     print("[{},{}]Save samples and labels successfully!".format(attack, dataset))