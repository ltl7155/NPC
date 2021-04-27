
import os 
import torch 
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from  torchvision.datasets  import utils as dtutil


class DatasetAdv(Dataset):
    
    @staticmethod
    def donwload_(file_id,root="./data_files/"):
        root = os.path.expanduser(root)
        os.makedirs(root,exist_ok=True)
        dtutil.download_file_from_google_drive(file_id=file_id, 
                                               root=root, 
                                               filename=f"adv_data_{file_id}.pkl")
        return os.path.join(os.path.abspath(root),f"adv_data_{file_id}.pkl")
    #x["layer"],x["regularization_weight"],x["epsilon"]
    def __init__(self,file_id_or_local_path):
        local_path =file_id_or_local_path if  "/" in file_id_or_local_path else ""
        print ("local_path....",local_path)
        if not os.path.isfile(local_path):
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
        return({
            "key":key,
#             "your_data":data["inputs"],
            "your_adv":data["adversaries"] if torch.is_tensor(data["adversaries"] ) else torch.from_numpy(data["adversaries"] ),
            "your_label":data["targets"],#self.targets if torch.is_tensor(self.targets ) else torch.from_numpy(self.targets )  
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

    import torch 
    import torchvision.utils     as vt
    from  torchvision.datasets  import utils as dtutil
    import  torchvision
    import json
    import  _MANU_utils_rand as mm
    device =torch.device("cuda")

    '''
    from models import *
    name="mnist"
    file_id="1rUzzcvG7R55TvJVpdOaqgV3OB0kRznWA"
    sadl_convnet_mnist= ConvnetMnist()
    dtutil.download_file_from_google_drive(file_id=file_id, root="./pretrained_models/mnist/", filename=f"torch_best_{name}_{file_id}.pth")
    sadl_convnet_mnist.load_state_dict(torch.load(f"./pretrained_models/mnist/torch_best_{name}_{file_id}.pth")["net"])
    sadl_convnet_mnist=sadl_convnet_mnist.to(device)
    model= sadl_convnet_mnist .eval()
    '''


    p="1hc_aj908k7_Zs2L4TsaWYENG-GwtdJe2"
    p="16j5VUzf1aATdFTS3I46X1Dq1Tg-X5HDO"
    p="1NsLZ6-qJxDnF4aMFYweVqb0zfJ9sSTqm"
    #p="1PnRcb06teIiSUZNcPnb2kd_pLtaRcyhq"
    p="1Cr6GYJHqj0K58QToM7UdtspePOXPTtrz"
    p="1jC4ST7DifndeKzsQEAzCvvItKU-AvBZ8"
    #p="1cVFnWDM-XJNWtY7DAKjAD-RQVQ_D1gtn"
    #p1="/home/malei/wj_code/sadl_torch/nc_diversity_attacks_SADL_manu/assets/manu_results_cifar-1000_adv_ConvnetCifar_2020-12-19.pkl"
    #p2="/home/malei/wj_code/sadl_torch/nc_diversity_attacks_SADL_manu/assets/manu_results_cifar-1000_nature_ConvnetCifar_2020-12-19.pkl"
    #model = mm.get_net("cifar").to(device)

    #p1="/home/malei/wj_code/sadl_torch/nc_diversity_attacks_SADL_manu/assets/manu_results_mnist-1000_adv_ConvnetMnist_2020-12-19.pkl"
    #p2="/home/malei/wj_code/sadl_torch/nc_diversity_attacks_SADL_manu/assets/manu_results_mnist-1000_nature_ConvnetMnist_2020-12-19.pkl"
    #model = mm.get_net("mnist").to(device)
#     p="1wMFTzhs2HEoFM3flRcP98yhl6Ab1x4Hi"


    #p1="/home/malei/wj_code/sadl_torch/nc_diversity_attacks_SADL_manu/assets/manu_results_svhn_alexnet-1000_adv_AlexNet_2020-12-19.pkl"
    #p2="/home/malei/wj_code/sadl_torch/nc_diversity_attacks_SADL_manu/assets/manu_results_svhn_alexnet-1000_nature_AlexNet_2020-12-19.pkl"
    #model = mm.get_net("svhn").to(device)


    p1="/home/malei/wj_code/sadl_torch/nc_diversity_attacks_SADL_manu/assets/manu_results_cifar_vgg-1000_adv_VGG16_2020-12-19.pkl"
    p2="/home/malei/wj_code/sadl_torch/nc_diversity_attacks_SADL_manu/assets/manu_results_cifar_vgg-1000_nature_VGG16_2020-12-19.pkl"
    #p1="/tmp/cifar.nature.npz"
    #p2="/tmp/cifar.adv.npz"
    #model = mm.get_net("cifar").to(device)

    #p1="/tmp/mnist.nature.npz"
    #p2="/tmp/mnist.adv.npz"
    #model = mm.get_net("mnist").to(device)

    p1="/tmp/cifar_vgg.nature.npz"
    p2="/tmp/cifar_vgg.adv.npz"
    model = mm.get_net("cifar_vgg").to(device)

    for p,x_n in zip([p1,p2],["na","adv"]):
        print ("-----"*8)
        dt = DatasetAdv(file_id_or_local_path=p)
        dl = torch.utils.data.DataLoader(dt,batch_size=1,num_workers=0)
        result_dict={}
        for idx,data_dict in enumerate(dl):
            
            #print (type(data_dict))
            #print (data_dict.keys())
            
            data= data_dict["your_data"]
            key= data_dict["key"]
            adv = data_dict["your_adv"]
            lbl = data_dict["your_label"]
            lbl_str = lbl.tolist()
            with open(f"{idx}_save.json","w") as f :
                json.dump(lbl_str,f)
            data=torch.squeeze(data,dim=0)
            vt.save_image(data,f"{idx}_ori_{x_n}_v3.jpg",nrow=20)

            adv=torch.squeeze(adv,dim=0)
            vt.save_image(adv,f"{idx}_{x_n}_v3.jpg",nrow=20)

            lbl=torch.squeeze(lbl,dim=0)
            #print (lbl.shape,data.shape,adv.shape)# = data_dict["your_label"]
            dataset1= torch.utils.data.TensorDataset(data,adv,lbl)
            datadl  = torch.utils.data.DataLoader(dataset1,batch_size=1000)
            with torch.no_grad():
                all_f,all_r=[],[]
                pic_all ,pic_all_f= [] ,[]
                for data_one in datadl:
                    data_fake = data_one[1].to(device).float()
                    data_real = data_one[0].to(device).float()
                    gt=data_one[-1]
                    gt = gt.squeeze_(dim=-1)
                    #print ("gt",gt.shape,"data_fake:",data_fake.shape,"data_real:",data_real.shape)#print (torch.max(data_fake),torch.min(data_fake))

                    read_pred=  model.forward(data_real)
                    fake_pred=  model.forward(data_fake)

                    read_pred = read_pred .argmax(dim=-1).cpu()
                    fake_pred = fake_pred .argmax(dim=-1).cpu()
                    #print ( torch.all(torch.eq(fake_pred,gt)) )
                    #print (gt)
        
                    #print (data_fake.shape,data_real.shape,gt.shape,"gt....",read_pred.shape,fake_pred.shape)

                    #acc_real = torch.eq(read_pred,gt).sum()#.tolist()/float(read_pred.shape[0])
                    #acc_fake = torch.eq(fake_pred,gt).sum()#.tolist()/float(fake_pred.shape[0])
                    real_c= torch.eq(read_pred,gt).sum()/float(read_pred.shape[0])
                    fake_c= torch.eq(fake_pred,gt).sum()/float(read_pred.shape[0])
                    print ("real_c",real_c,"fakle",fake_c)


                #all_f_pred,all_f_gt  =  [x for x,y in all_f] ,[y for x,y in all_f]
                #all_f_pred,all_f_gt   = torch.cat (all_f_pred,dim=0), torch.cat(all_f_gt,dim=0)
                #acc_count= torch.eq(all_f_pred,all_f_gt).sum()
                #print (all_f_pred[-10:],"all_f_pred...")
                #total_c = all_f_pred.shape[0]
                #acc = acc_count/float(all_f_pred.shape[0])
                #print ("----"*8)
                #print ("pred_true_c",acc_count, "total_c ", total_c , "acc%", acc )
                


                #pic_all=torch.cat(pic_all,dim=0)
                #torchvision.utils.save_image(pic_all,"/tmp/real.jpg")
                #pic_all_f=torch.cat(pic_all_f,dim=0)
                #torchvision.utils.save_image(pic_all_f,"/tmp/fake.jpg")
                #print ("acc_fake",sum(all_f)/float(len(all_f)),"acc_real", sum(all_r)/float(len(all_r)) )


