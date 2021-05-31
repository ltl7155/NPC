import torch.nn as nn 
import torch.nn.functional as F 
import tqdm 
import torch 
import numpy as np 
import torch.utils.data as Data

from . import utils_nn  as deephunter_util 

device = torch.device("cuda")

class TorchModel(nn.Module):
    def __init__(self,net,layer_names=None ):
        super(TorchModel, self).__init__()
        if "TorchModel" in  str(type(net)):
            self.net = net.net 
        else :
            self.net  =net 
        
#         print (self.net)
        assert hasattr(self.net,"to") ,"hereis a model"
        self.layer_names = layer_names
        self.layer_dict = deephunter_util.get_model_layers(self.net,include_relu=True)
        print (self.layer_dict.keys())
        self.net.eval()

    
    # def forward(self,x,clear_method=lambda x:x.detach().cpu().numpy()):
    #
        # outputs= {}
        # for layer_name in self.layer_names :
            # o=deephunter_util.get_layer_output_v2(self.net, x , layer_name,include_relu=True,clear_method=clear_method)
            #
            # outputs.update(o)
            #
        # return outputs
    
    def predict_classes(self, dataset, batch_size=16, verbose=1,clear_method=lambda x:x.detach().cpu().numpy() ): 
        # traceback.print_stack()
        # print ("your input",type(dataset),"dataset...")
        # print (dataset.shape)
        dataset = deephunter_util.batch_data(dataset)
        # print ("your input",type(dataset),"dataset...")
        
        logits_out= deephunter_util.predict_batch(
            model=self.net,
            data= dataset, 
            fetch_func=lambda x:x[0].to(device),
            )
        # print ("logits", torch.is_tensor(logits), type(logits))
        #
        # print (logits.shape,"logits.shape",)
        #

        pred_class = logits_out .argmax(dim=-1)
        pred_class = clear_method(pred_class)
        print("\tfinished!")
        assert pred_class.dtype== np.int64, pred_class.dtype
        assert len(pred_class.shape)== 1, pred_class.shape
        return pred_class
    
    def predict(self,dataset ,batch_size=16, verbose=1 ,clear_method=lambda x:x.detach().cpu().numpy()): 
        raise Exception("should not be used anymore")
        dataset = deephunter_util.tensor2loader(dataset)
        
        logits_out,trace_out= deephunter_util.get_layer_output_batch(
            model=self.net,
            data= dataset,
            fetch_func=lambda x:x[0].to(device),
            include_relu=True,
            clear_method=lambda x:x.detach().cpu() )

        assert type(trace_out)==dict ,"your network forward should be logits,{\"layer1\":lay1_out.... }"
        
        ret=[v.numpy() for k,v in trace_out.items() if self.layer_names is None or k in self.layer_names]
        
        for i in  range(len(ret)):
            one_ret= ret[i]
            if len(one_ret.shape)>4:
                raise Exception("unimplementation")
            if len(one_ret.shape)==4:
                ret[i]=np.transpose(one_ret,(0,2,3,1))
            if len(one_ret.shape)==3:
                ret[i]=np.transpose(one_ret,(0,2,1))

                
        print("\tfinished!")
        assert self.layer_names is None or  len(ret)==len(self.layer_names), f"{ len(ret)}=={len(self.layer_names)},trace_out={len(trace_out)}?"
     
        for one_ret in ret :
            print ("predict.one_ret-->",type(one_ret))
            print ("predict.one_ret-->",len(one_ret))
            print ("predict.one_ret-->",one_ret.shape)
     
        return ret #if self.layer_names is None or len(self.layer_names)>1 else ret[0]
    
    def predict_v2(self,dataset ,batch_size=16, verbose=1 ,clear_method=lambda x:x.detach().cpu().numpy()): 
        dataset = deephunter_util.tensor2loader(dataset)
        
        logits_out,trace_out= deephunter_util.get_layer_output_batch_v2(
            model=self.net,
            data= dataset,
            fetch_func=lambda x:x[0].to(device),
            include_relu=True,
            clear_method=lambda x:x.detach().cpu() )

        assert type(trace_out)==dict ,"your network forward should be logits,{\"layer1\":lay1_out.... }"
        
        ret=[v.numpy() for k,v in trace_out.items() if self.layer_names is None or k in self.layer_names]
        
        for i in  range(len(ret)):
            one_ret= ret[i]
            assert one_ret.ndim<=3,f"this is a reduce result should be, epxect one_ret.ndim<=3, but get {one_red.shape}"
            print ("predict.one_ret-->",type(one_ret))
            print ("predict.one_ret-->",len(one_ret))
            print ("predict.one_ret-->",one_ret.shape)
     
        print("\tfinished!")
        assert self.layer_names is None or  len(ret)==len(self.layer_names), f"{ len(ret)}=={len(self.layer_names)},trace_out={len(trace_out)}?"
        return ret #if self.layer_names is None or len(self.layer_names)>1 else ret[0]
    




if __name__=="__main__":
    import traceback
    import sys 
    sys.path.append("../../")
    
    class Net(nn.Module):
        def __init__(self,num_classes=10):
            super(Net, self).__init__()
            self.features=nn.Sequential(
                 nn.Conv2d(3, 32, kernel_size=3,padding=1),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(32, 32, kernel_size=3,padding=1),
                 nn.ReLU(inplace=True),
                 nn.MaxPool2d(2),
                 nn.Conv2d(32, 64, kernel_size=3,padding=1),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(64, 64, kernel_size=3,padding=1),
                 nn.ReLU(inplace=True),
                 nn.MaxPool2d(2),
                 nn.Conv2d(64, 128, kernel_size=3,padding=1),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(128, 128, kernel_size=3,padding=1),
                 nn.ReLU(inplace=True),
                 nn.MaxPool2d(2) )
    
            self.classifier  =nn.Sequential(
                nn.Dropout(),
                nn.Linear(128 * 4 * 4, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, num_classes),
                )
            
            data= self.features 
        def forward(self, x):
            x= self.features(x)
            x1_flatt=nn.Flatten()(x)
            
            x_out= self.classifier(x1_flatt)
            
            return F.log_softmax(x_out, dim=-1) 

    
    import torch.utils.data
    num_class=10
    total_data=torch.randn(128,3,32,32)
    total_lbl = torch.randint(0,num_class,(128,))
    device=torch.device("cuda")
    
    dataset=torch.utils.data.TensorDataset(total_data.to(device),total_lbl.to(device))
    
    # dl =torch.utils.data.DataLoader(dataset,batch_size=16,num_workers=0)
    # for idx,(img,lbl) in enumerate(dl):
    #     print (img.shape,lbl.shape)
    # 
    model  =Net()
    model  =model.to(device)
    
    print (type(model) )
    # tempmodel = TorchModel(net=model,layer_names=["ReLU.3"])
    tempmodel = TorchModel(net=model)
    
    cls_ret = tempmodel.predict_classes(dataset)
    
    print (type(cls_ret))
    cls_trace  = tempmodel.predict(dataset)
    
    print (type(cls_trace))
    
    for k,v in enumerate(cls_trace):
        print (k,"::::", type(v))
        print (len(v))
        print (type(v[0]))
    
    # 
    # import torch 
    # x=torch.randn(4,3,32,32)
    #  
    # ret,trace_out = model(x)
    #  
    # for k,v in trace_out.items():
    #     print (k,":",v.shape)
