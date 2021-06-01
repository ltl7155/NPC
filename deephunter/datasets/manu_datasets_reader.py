
import os 
import torch 
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from  torchvision.datasets  import utils as dtutil
'''
RQ3 : Sensitivity with Defect Detection

We follow the similar design in [23]: 
    first, we randomly select 1,000 benign samples (denoted as 𝑠𝑏 ) from the test data. 
    Then, we construct several test suites by replacing a small number of samples of 𝑠𝑏 (i.e., 1%, 2%, 3%, 5%, 7%, 10%) 
        with the same number of errors such that they have the same size. 
    To be specific, for natural errors, we select a certain number of inputs from test data in each class, that are predicted incorrectly. 
    For adversarial examples, we adopt the PGD attack [28] to generate a large number of adversarial examples, some of which are then randomly selected in each class. 
    
    For natural errors and adversarial examples, we generate 7 test suites including different numbers of errors, respectively.
     To reduce the randomness, we repeat the process 5 times and calculate the average results.


'''
def get_dataloader(file_id_or_local_path):
    
    
    local_path =file_id_or_local_path if  "/" in file_id_or_local_path else ""

    return DatasetAdv(file_id_or_local_path=file_id_or_local_path)
    


class DatasetAdv(Dataset):
    
    @staticmethod
    def donwload_(file_id,root=None):
        if root is None :
            root = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),"../../","data/data_files")
        root = os.path.expanduser(root)
        os.makedirs(root,exist_ok=True)
        dtutil.download_file_from_google_drive(file_id=file_id, 
                                               root=root, 
                                               filename=f"adv_data_{file_id}.pkl")
        return os.path.join(os.path.abspath(root),f"adv_data_{file_id}.pkl")
    #x["layer"],x["regularization_weight"],x["epsilon"]
    
    def __init__(self,file_id_or_local_path):
        # if not os.path.isfile(local_path):
        local_path=self.donwload_(file_id=file_id_or_local_path)
        
        data=np.load(local_path,allow_pickle=True)
        d0=data[0]
        d1=data[1]
#         print ("uo:"*8,type(data), type(d0),type(d1),"d0d1,",list(d0.keys()), list(d1.keys()))
        self.inputx=  d0["inputs"]
        self.targets=  d0["targets"]
        
        self.data={}
        self.data_exp={}
        
        #print (d1.keys())
        if "epsilon" in d1 :
            get_key =lambda x:"{}:{}:{}:{}".format(x["attack"],x["layer"],x["regularization_weight"],x["epsilon"])
        else:
            get_key =lambda x:"{}:{}:{}:{}".format(x["attack"],x["layer"],x["regularization_weight"],x["confidence"])
#         get_value = lambda x:x["adversaries"]
#         
#         get_other = lambda x:x["adversaries"]=None
         
        self.keys = [get_key(y)  for y in data[1:]  ]
        
        self.data.update({get_key(y):y    for y in data[1:]  })                                   
#         self.data_exp.update({get_key(y):get_other(y)   for y in d1  })                                   

        
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self,index):
        
        key  = self.keys[index]
        
        data= self.data[key]
        label = self.targets 
        your_adv = data["adversaries"] if torch.is_tensor(data["adversaries"] ) else torch.from_numpy(data["adversaries"] )
        your_adv = your_adv.float()
        if self.targets is not None :
            your_lbl = self.targets if torch.is_tensor(self.targets ) else torch.from_numpy(self.targets ) 
        else :
            your_lbl = data["targets"]
        return({
            "key":key,
             "your_data": data["inputs"] if self.inputx is None else  self.inputx,
            "your_adv":data["adversaries"] if torch.is_tensor(data["adversaries"] ) else torch.from_numpy(data["adversaries"] ),
             "your_label":your_lbl , #self.targets if torch.is_tensor(self.targets ) else torch.from_numpy(self.targets )  
            })
        
#         del data["adversaries"]
        
        return data


if __name__=="__main__":
    p="1hc_aj908k7_Zs2L4TsaWYENG-GwtdJe2"
    p="16j5VUzf1aATdFTS3I46X1Dq1Tg-X5HDO"
    import torch 
    dt = DatasetAdv(file_id_or_local_path=p)
    dl = torch.utils.data.DataLoader(dt,batch_size=1,num_workers=0)
    
    result_dict={}
    for idx,data_dict in enumerate(dl):
        
        print (type(data_dict))
        print (data_dict.keys())
        
        data= data_dict["your_data"]
        key= data_dict["key"]
        adv = data_dict["your_adv"]
        lbl = data_dict["your_label"]
        
        assert len(key)==1 ,"we use batch_size==1, if your want >1,pls be care the mapto result_dict"
        
        print (type(data),type(lbl),type(adv))
        print ("-->",data.shape,lbl.shape, key)
        
        result_dict[key[0]]= 0.000
        
    
    save_score_method(result_dict,file_id=p)
#     import torch 
#     
#     import torchvision
#     import torchvision.transforms as transforms
#     data_dir="~/.torch"
#     device =torch.device("cpu")
#     num_per_class=10
#     for i in  range(10):
#         dataset = torchvision.datasets.CIFAR10(root=data_dir, 
#                                                train=False, 
#                                                download=True,
#                                                transform=transforms.Compose([
#                                                    transforms.ToTensor()
#                                                ]))
#     
#         class_distribution = torch.ones(len(np.unique(dataset.targets))) * num_per_class
#         inputs, targets = generate_batch(dataset, class_distribution, device)
#         
#         print (inputs.shape,inputs.mean(),inputs.std())
# #         print (targets.shape,targets.mean(),targets.min())
#         print ("=====\n=====")
    
