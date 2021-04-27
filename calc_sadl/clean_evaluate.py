import torch 



import torch.nn as nn 
import torch.nn.functional as F 
import torch 
from  torchvision.datasets  import utils as dtutil

# from  models  import ConvnetMnist as NET_MNIST
# from  models  import ConvnetCifar as NET_CIFAR10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


import  torchvision
from keras.datasets import mnist, cifar10


import torch 
import numpy as np 
import os
# load torch model and   trained  with torch 
import  models 


def set_random_seed_for_reproducibility(seed_value=123456):
    import random 
    import numpy as np 
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    os.environ['PYTHONHASHSEED'] = str(seed_value) # Python hash buildin
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False
set_random_seed_for_reproducibility()


tansform_simple= torchvision.transforms.ToTensor()

test_dataset= torchvision.datasets.MNIST(root="~/.torch", download=True,train=False,transform=tansform_simple)
test_dataloader  = torch.utils.data.DataLoader(test_dataset,
                       batch_size=128,
                       num_workers=4,
                       shuffle=False ,
                   )



test_dataset_cifar= torchvision.datasets.CIFAR10(root="~/.torch", download=True,train=False,transform=tansform_simple)
test_dataloader_cifar  = torch.utils.data.DataLoader(test_dataset_cifar,
                       batch_size=128,
                       num_workers=4,
                       shuffle=False ,
                   )





def estimate_the_dataset(model, testloader,need_softmax=False):
    
    test_loss = 0
    correct = 0
    total = 0

    criterion =torch. nn.CrossEntropyLoss()
    model .eval()
    collect = []
    
    
    with torch.no_grad():
        
        
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if batch_idx==0:
                print ("====",inputs.min(),inputs.max())
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if need_softmax:
                outputs= F.softmax(outputs,dim=-1)
            
#             print ("target",targets.max(),targets.min(),inputs.max(),inputs.min())
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            
            pred_match_idx =predicted.eq(targets)
            
            collect.append(pred_match_idx.cpu().numpy())
#             
# #             print ("----",pred_match_idx[:10],"----")
#             targets_error = targets[~ pred_match_idx]
#             
#             
#             collect.append(targets_error.cpu().numpy())
#              
            
    ret =   np. concatenate(collect,axis=0)
    
    acc = 100.*correct/total

    #ret = np.unique(ret,return_counts=True)
#     print (ret)
    return ret ,acc


    acc = 100.*correct/total
    return acc 

    
'''
mnist
https://drive.google.com/file/d/1rUzzcvG7R55TvJVpdOaqgV3OB0kRznWA/view?usp=sharing
cifar
https://drive.google.com/file/d/1zK66FYCGu5zxbIcAx-f8PRPY-cQ_sBrE/view?usp=sharing
cifar vgg
https://drive.google.com/file/d/1Ys3-0QuxN6tbzcAh_nlNHJcrX6pVc-r2/view?usp=sharing
'''
import os 
#os.makedirs("./model",exist_ok=True)
root_dir= os.path.expanduser("~/.cache/")
 
acc_collect={}
name="mnist"
print (f"====================={name}====================")
if not os.path.isfile("uniq_mnist.npy"):
    file_id="1rUzzcvG7R55TvJVpdOaqgV3OB0kRznWA"
    net = models. ConvnetMnist()
    dtutil.download_file_from_google_drive(file_id=file_id, root=root_dir, filename=f"torch_best_{name}_{file_id}.pth")
    net.load_state_dict(torch.load(f"{root_dir}/torch_best_{name}_{file_id}.pth")["net"])
    net= net.to(device)
    unq ,acc = estimate_the_dataset(net,test_dataloader )
    acc_collect["mnist_with_acc"]= acc 
    acc_collect["mnist_with_unq"]= unq 
 
    print (unq.shape)
    np.save("uniq_mnist.npy",unq)
# print (acc_collect)
acc_collect={}
# 
# 
# 
name="cifar"
print (f"====================={name}====================")
if not os.path.isfile("uniq_cifar.npy"):
    file_id="1zK66FYCGu5zxbIcAx-f8PRPY-cQ_sBrE"
    net = models. ConvnetCifar()
    dtutil.download_file_from_google_drive(file_id=file_id, root=root_dir, filename=f"torch_best_{name}_{file_id}.pth")
    net.load_state_dict(torch.load(f"{root_dir}/torch_best_{name}_{file_id}.pth")["net"])
    net= net.to(device)
    unq ,acc = estimate_the_dataset(net,test_dataloader_cifar )
    acc_collect["cifar_with_acc"]= acc 
    acc_collect["cifar_with_unq"]= unq 


    print (unq.shape)
    np.save("uniq_cifar.npy",unq)

# print (acc_collect)
acc_collect={}

name="cifar"
print (f"====================={name} vgg====================")
if not os.path.isfile("uniq_cifar_vgg.npy"):
    file_id="1Ys3-0QuxN6tbzcAh_nlNHJcrX6pVc-r2"
    net = models. VGG16()
    dtutil.download_file_from_google_drive(file_id=file_id, root=root_dir, filename=f"torch_best_{name}_{file_id}.pth")
    net.load_state_dict(torch.load(f"{root_dir}/torch_best_{name}_{file_id}.pth"))
    net= net.to(device)
    unq ,acc = estimate_the_dataset(net,test_dataloader_cifar )
    acc_collect["vgg_with_acc"]= acc 
    acc_collect["vgg_with_unq"]= unq 


    print (unq.shape)
    np.save("uniq_cifar_vgg.npy",unq)
# print (acc_collect)
acc_collect={}