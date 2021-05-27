import torch.nn as nn 
import torch.nn.functional as F 
import tqdm 
import torch 
import numpy as np 
import torch.utils.data as Data


# def extract_outputs(model, data, module):
#     outputs = []      
#     def hook(module, input, output):
#         outputs.append(output)    
#     handle = module.register_forward_hook(hook)     
#     model(data)
#     handle.remove()
#     return torch.stack(outputs)
# def step_through_model(model, prefix=''):
#     for name, module in model.named_children():
#         path = '{}/{}'.format(prefix, name)
#         if (isinstance(module, nn.ReLU)):
# #             or isinstance(module, nn.Conv2d)
# #             or isinstance(module, nn.Relu)): # test for dataset
#             yield (path, name, module)
#         else:
#             yield from step_through_model(module, path)
# 
# def get_model_layers(model):
#     layer_dict = {}
#     idx=1
#     for (path, name, module) in step_through_model(model):
#         path_prefix = path.split("-")[0]
#         path_prefix = path_prefix.replace("relu","activation")
#         
#         layer_dict[path_prefix+path + '-' + str(idx)] = module
#         idx += 1
#     return layer_dict 


    
    
def get_model_modules(model, layer_name=None):
    layer_dict = {}
    idx=0
    for name, module in model.named_children():
        if (not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.BatchNorm2d)
            and not isinstance(module, nn.Dropout)
#             and not isinstance(module, nn.ReLU)
            and (layer_name is None or layer_name in name)):
            layer_dict[name + '-' + str(idx)] = module
            idx += 1
        else:
            for name_2, module_2 in module.named_children():
                for name_3, module_3 in module_2.named_children():
                    if (not isinstance(module_3, nn.Sequential)
                        and not isinstance(module_3, nn.BatchNorm2d)
                        and not isinstance(module, nn.Dropout)
#                         and not isinstance(module, nn.ReLU)
                        and 'shortcut' not in name_3
                        and (layer_name is None or layer_name in name_3)):
                        layer_dict[name_3 + '-' + str(idx)] = module_3
                        idx += 1    
                        
    return layer_dict

def step_through_model(model, prefix=''):
    for name, module in model.named_children():
        path = '{}/{}'.format(prefix, name)
#         if (isinstance(module, nn.Conv1d)
#             or isinstance(module, nn.Conv2d)
#             or isinstance(module, nn.ReLU)
#             or isinstance(module, nn.Linear)): # test for dataset
        if ( isinstance(module, nn.ReLU)): # test for dataset
            yield (path, name, module)
        else:
            yield from step_through_model(module, path)

def get_model_layers(model, cross_section_size=0,):
    layer_dict = {}
    i = 0
    for (path, name, module) in step_through_model(model):
        layer_dict[str(i) + path] = module
        i += 1
    if cross_section_size > 0:
        target_layers = list(layer_dict)[0::cross_section_size] 
        layer_dict = { target_layer: layer_dict[target_layer] for target_layer in target_layers }
    return layer_dict 

def get_layer_output_value(model, data, layer_name=None,clear_method=lambda x:x.detach().cpu().numpy()):   
    output_sizes = {}
    hooks = []  
    
    layer_dict_ori = get_model_layers(model)
    if layer_name is not None :
        layer_dict ={k:v for k,v in layer_dict_ori.items() if k==layer_name}
    else:
        layer_dict = layer_dict_ori
         
    def hook(module, input, output):
        module_idx = len(output_sizes)
        m_key = list(layer_dict)[module_idx]
        if output.ndim>=4:
            output=output.transpose(1,2).transpose(2,3,)

        output_sizes[m_key] = clear_method(output)#list(output.size()[1:])      
    
    for name, module in layer_dict.items():
        hooks.append(module.register_forward_hook(hook))
    
    try:
        #model(data[:1])  
        model(data)  
    finally:
        for h in hooks:
            h.remove() 
            
    return output_sizes
# def get_layer_output_sizes(model, data, layer_name=None):   
#     output_sizes = {}
#     hooks = []  
#     
#     layer_dict_ori = get_model_layers(model)
#     if layer_name is not None :
#         layer_dict ={k:v for k,v in layer_dict_ori.items() if k==layer_name}
#     else:
#         layer_dict = layer_dict_ori
#          
#     def hook(module, input, output):
#         module_idx = len(output_sizes)
#         m_key = list(layer_dict)[module_idx]
#         output_sizes[m_key] = list(output.size()[1:])      
#     
#     for name, module in layer_dict.items():
#         hooks.append(module.register_forward_hook(hook))
#     
#     try:
#         model(data[:1])  
#     finally:
#         for h in hooks:
#             h.remove() 
#             
#     return output_sizes


class TorchModel(nn.Module):
    def __init__(self,net,layer_names=[] ):
        super(TorchModel, self).__init__()
        if "TorchModel" in  str(type(net)):
            self.net = net.net 
        else :
            self.net  =net 
        
#         print (self.net)
        assert hasattr(self.net,"to") ,"hereis a model"
        self.layer_names = layer_names
        self.layer_dict = get_model_layers(self.net)
        print (self.layer_dict.keys())
        self.net.eval()
#         print (self.layer_dict.keys())
#         
#         print (dir(self.net))
#         exit()
    
    def forward(self,x,clear_method=lambda x:x.detach().cpu().numpy()):

        outputs= {}
        for layer_name in self.layer_names :
            o=get_layer_output_value(self.net, x , layer_name,clear_method=clear_method)
            
            outputs.update(o)

        return outputs
    
    def predict_classes(self, dataset, batch_size=16, verbose=1,clear_method=lambda x:x.detach().cpu().numpy() ): 
        dataloader = None 
        print("getting classes....")
        if type(dataset)==np.ndarray:
            if dataset.ndim<4:

                dataset=np.expand_dims(dataset,axis=-1)
                assert dataset.ndim==4 ,f"ndim=4, but {dataset.shape}"
                assert np.max(dataset)<=1,f"max-min--> {np.max(dataset)} --->min {np.min(dataset)}"
                #raise Exception(f"shoudle be [bchw], but get {dataset.shape}")
            if type(dataset)==np.ndarray :
                print ("---"*8,dataset.shape,"---")
                dataset = torch.from_numpy(dataset).transpose(2,3).transpose(1,2)
                dataset =torch.utils.data.TensorDataset(dataset)
                dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,num_workers=0,drop_last=False)
                if verbose>0 :
                    dataloader = tqdm.tqdm (enumerate(dataloader ))
                else:
                    dataloader = enumerate(dataloader )
        else:
            print("Setting dataloader...")
#             print(dataset)
            dataloader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
            if verbose>0 :
                    dataloader = tqdm.tqdm (enumerate(dataloader ))
            else:
                dataloader = enumerate(dataloader )

        with torch .no_grad() :
            if dataloader is not None :
                logits_out_list= []
                for idx, data  in dataloader :
                    if type(data)==list:
                        data= data[0]
                    data =data .cuda()
#                     print(data.size())
                    logits_out_one = self.net.forward(data)
                    logits_out_list .append( logits_out_one )
                logits_out = torch.cat(logits_out_list,dim=0)
            else :
                logits_out = self.net.forward(dataset)
        #logits_out = torch.softmax(logits_out)
        pred_class = logits_out .argmax(-1)
        pred_class = clear_method(pred_class)
        print("\tfinished!")
        assert pred_class.dtype== np.int64, pred_class.dtype
        assert len(pred_class.shape)== 1, pred_class.shape
        return pred_class;
    
    def predict(self,dataset ,batch_size=16, verbose=1 ,clear_method=lambda x:x.detach().cpu().numpy()): 
        #init 
        dataloader = None 
        print("getting feature maps....")
        collect_trace_out = {}
        for ly in self.layer_names:
            collect_trace_out[ly]=[] 
        logits_list = [] 
        if type(dataset)==np.ndarray:
            if dataset.ndim<4:
                dataset=np.expand_dims(dataset,axis=-1)
                assert dataset.ndim==4 ,f"ndim=4, but {dataset.shape}"
                assert np.max(dataset)<=1,f"max-min--> {np.max(dataset)} --->min {np.min(dataset)}"
                #raise Exception("shoudle be [bchw]")

            if type(dataset)==np.ndarray :
                dataset = torch.from_numpy(dataset).transpose(2,3).transpose(1,2)
                dataset =torch.utils.data.TensorDataset(dataset)
                dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,num_workers=0,drop_last=False)
                if verbose>0 :
                    dataloader = tqdm.tqdm (enumerate(dataloader ))
                else:
                    dataloader = enumerate(dataloader )
        
        else:
            print("Setting dataloader...")
#             print(dataset)
            dataloader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
            if verbose>0 :
                dataloader = tqdm.tqdm (enumerate(dataloader ))
            else:
                dataloader = enumerate(dataloader )


        with torch .no_grad() :
            if dataloader is not None :
                logits_out_list= {}
                for idx,data  in dataloader :
                    if type(data)==list:
                        data= data[0]
                    data =data . cuda()
                    logits_out_one = self.forward( data)
                    
                    for k,v in  logits_out_one.items():
                        if k in logits_out_list:
                            vv = logits_out_list[k] 
                            v  = vv+[v]
                        else :
                            v =  [v]
#                     print(logits_out_one)
                        logits_out_list.update({k:v})
                for k,v in logits_out_list.items():
                    v=np.concatenate(v)
                    logits_out_list[k]=v
                trace_out =  logits_out_list
            else :
                trace_out = self.forward(dataset)

        assert type(trace_out)==dict ,"your network forward should be logits,{\"layer1\":lay1_out.... }"

        ret=[v for k,v in trace_out.items() if k in self.layer_names]
            
        print("\tfinished!")
        assert  len(ret)==len(self.layer_names)
     
        return ret if len(self.layer_names)>1 else ret[0]
    

if __name__=="__main__":
    
    
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
    
    dataset=torch.utils.data.TensorDataset(total_data,total_lbl)
    
    # dl =torch.utils.data.DataLoader(dataset,batch_size=16,num_workers=0)
    # for idx,(img,lbl) in enumerate(dl):
    #     print (img.shape,lbl.shape)
    # 
    model  =Net()
    print (type(model) )
    tempmodel = TorchModel(net=model,layer_names=["ReLU.3"])
    
    cls_ret = tempmodel.predict_classes(dataset)
    
    print (type(cls_ret))
    cls_ret,cls_trace  = tempmodel.predict(dataset)
    
    print (type(cls_trace), cls_trace.keys())
    
    for k,v in cls_trace.items():
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
