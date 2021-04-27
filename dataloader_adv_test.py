
import os 
import torch 
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from  torchvision.datasets  import utils as dtutil


class DatasetAdv(Dataset):
    
    @staticmethod
    def donwload_(file_id, root="./"):
        os.makedirs(root, exist_ok=True)
        dtutil.download_file_from_google_drive(file_id=file_id, 
                                               root=root, 
                                               filename=f"adv_data_{file_id}.pkl")
        return os.path.join(os.path.abspath(root),f"adv_data_{file_id}.pkl")
    #x["layer"],x["regularization_weight"],x["epsilon"]
    def __init__(self, file_id_or_local_path):
        local_path =file_id_or_local_path if  "/" in file_id_or_local_path else ""
        if not os.path.isfile(local_path):
            local_path=self.donwload_(file_id=file_id_or_local_path)
        
        data=np.load(local_path, allow_pickle=True)
#         print(data)
#         print ("uo:",len(data))
        d0=data[0]
        d1=data[1]
        
        self.inputx=  d0["inputs"]
        self.targets=  d0["targets"]
        
        self.data={}
        self.data_exp={}
        
        print (d1.keys())
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
        
        
        if torch.is_tensor(data["adversaries"]):
            return({
            "key":key,
            "your_data":self.inputx,
            "your_adv":data["adversaries"],
            "your_label":self.targets
        
        })

        
        return({
            "key":key,
            "your_data":self.inputx,
            "your_adv":torch.from_numpy(data["adversaries"]),
            "your_label":torch.from_numpy(self.targets)
        
        })
        
        
        #         del data["adversaries"]
        
        return data
    
def save_score_method(result_dict:dict,file_id,save_dir="./",need_check=True):
    import os 
    assert type(result_dict)==dict ,f"the result_dict should be dict,but {type(result_dict)}"
    
    if need_check:
        print ("start check","===="*8)
        key_list= []
        dt = DatasetAdv(file_id_or_local_path=file_id)
        dl = torch.utils.data.DataLoader(dt,batch_size=100,num_workers=0)
        for data in  dl :
            key_list.extend(data["key"])
        
        assert len(key_list)==len(result_dict), f"missing some key from origin {len(key_list)}!={len(result_dict)}"
        assert set(key_list) == set(result_dict.keys()), f"missing some key from origin { set(key_list) }:::: {set(result_dict.keys())} ::: "
        
    save_filename= "result_score_{}.npy".format(file_id)
    save_filename=os.path.join(save_dir,save_filename)
    np.save(save_filename,result_dict) 


if __name__=="__main__":
    p="1hc_aj908k7_Zs2L4TsaWYENG-GwtdJe2"
    p="1Gm926_p5_bvhgfDdlQmmV9lUCmjsF5Ft"
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
    
