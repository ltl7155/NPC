import torch 
import torch.nn as nn

import numpy as np

from tqdm import tqdm

import os 
import traceback

device=torch.device("cuda")

# def get_model_modules(model, layer_name=None):
    # layer_dict = {}
    # idx=0
    # for name, module in model.named_children():
        # if (not isinstance(module, nn.Sequential)
            # and not isinstance(module, nn.BatchNorm2d)
            # and not isinstance(module, nn.Dropout)
            # and not isinstance(module, nn.ReLU)
            # and (layer_name is None or layer_name in name)):
            # layer_dict[name + '-' + str(idx)] = module
            # idx += 1
        # else:
            # for name_2, module_2 in module.named_children():
                # for name_3, module_3 in module_2.named_children():
                    # if (not isinstance(module_3, nn.Sequential)
                        # and not isinstance(module_3, nn.BatchNorm2d)
                        # and not isinstance(module, nn.Dropout)
                        # and not isinstance(module, nn.ReLU)
                        # and 'shortcut' not in name_3
                        # and (layer_name is None or layer_name in name_3)):
                        # layer_dict[name_3 + '-' + str(idx)] = module_3
                        # idx += 1    
                        #
    # return layer_dict



def step_through_model(model, prefix='',**kwargs):
    for name, module in model.named_children():
        path = '{}/{}'.format(prefix, name)
        include_relu = kwargs.get("include_relu",False)
        # print ("->include_relu",include_relu)
        if include_relu :
            if (
                # isinstance(module, nn.Conv1d)
                # or isinstance(module, nn.Conv2d)
                # or isinstance(module, nn.Linear)
                isinstance(module, nn.ReLU)
                ):
                yield (path, name, module)
            else:
                yield from step_through_model(module, path,**kwargs)

        else:
            if (isinstance(module, nn.Conv1d)
                or isinstance(module, nn.Conv2d)
                or isinstance(module, nn.Linear)
                ): # test for dataset
                yield (path, name, module)
            else:
                yield from step_through_model(module, path,**kwargs)


def get_model_layers(model, cross_section_size=0,**kwargs):
    layer_dict = {}
    i = 0
    for (path, name, module) in step_through_model(model,**kwargs):
        layer_dict[str(i) + path] = module
        i += 1
    if cross_section_size > 0:
        target_layers = list(layer_dict)[0::cross_section_size] 
        layer_dict = { target_layer: layer_dict[target_layer] for target_layer in target_layers }
    return layer_dict 


def get_layer_output(model, data, layer_name=None,layer_dict=None,
                     clear_methods=lambda x:x.detach().cpu(),**kwargs):   
    output_dict = {}
    hooks = []  
    if layer_dict is None :
        layer_dict = get_model_layers(model,**kwargs)
 
    def hook(module, input, output):
        module_idx = len(output_dict)
        m_key = list(layer_dict)[module_idx]
        output_dict[m_key] =output.cpu()#clear_methods( output)#list(output.size()[1:])      
    
    for name, module in layer_dict.items():
        hooks.append(module.register_forward_hook(hook))
    
    try:
        # print ("* get_layer_output",data.shape,data.mean(),data.std())
        logits = model(data)  
    finally:
        for h in hooks:
            h.remove() 
            
    return logits , output_dict

def default_reduct(x):
    if x.ndim <4 : ## only for conv layer
        return x 
    stride=list(range(x.ndim))
    mean_stride = stride[2:]

    if torch.is_tensor(x):    
        return torch.mean(x,dim=mean_stride )
    return np.mean(x,axis=mean_stride)

def get_layer_output_v2(model, data, layer_name=None,layer_dict=None,
                        clear_method=lambda x:x.detach().cpu(),
                        reduce_method=default_reduct,**kwargs):   
    raise Exception("donot use")
def get_layer_output_reduce(model, data, layer_name=None,layer_dict=None,
                        clear_method=lambda x:x.detach().cpu(),
                        reduce_method=default_reduct,**kwargs):   

    output_sizes = {}
    hooks = []  

    if layer_dict is None :
        layer_dict = get_model_layers(model,**kwargs)
    if layer_name is not None :
        layer_dict ={k:v for k,v in layer_dict.items() if k==layer_name}
    # else:
        # layer_dict = layer_dict_ori
         
    def hook(module, input, output):
        module_idx = len(output_sizes)
        m_key = list(layer_dict)[module_idx]
        # if output.ndim>=4:
            # output=output.transpose(1,2).transpose(2,3,)
        # print ("hook from ",output.shape)
        output = reduce_method(output)
        # print ("hook to ",output.shape)
        output_sizes[m_key] = output.cpu()
    
    for name, module in layer_dict.items():
        hooks.append(module.register_forward_hook(hook))
    
    try:
        #model(data[:1])  
        logits =model(data)  
    finally:
        for h in hooks:
            h.remove() 
            
    return logits, output_sizes
# def get_layer_output(model, data, layer_name=None,layer_dict=None):   
    # output_dict = {}
    # hooks = []  
    # if layer_dict is None :
        # layer_dict = get_model_layers(model)
        #
    # def hook(module, input, output):
        # module_idx = len(output_dict)
        # m_key = list(layer_dict)[module_idx]
        # output_dict[m_key] = output#list(output.size()[1:])      
        #
    # for name, module in layer_dict.items():
        # hooks.append(module.register_forward_hook(hook))
        #
    # try:
        # logits = model(data)  
    # finally:
        # for h in hooks:
            # h.remove() 
            #
    # return logits , output_dict




def clear_methods_safe(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x 







# def get_model_modules(model, layer_name=None):
    # layer_dict = {}
    # idx=0
    # for name, module in model.named_children():
        # if (not isinstance(module, nn.Sequential)
            # and not isinstance(module, nn.BatchNorm2d)
            # and not isinstance(module, nn.Dropout)
            # and not isinstance(module, nn.ReLU)
            # and (layer_name is None or layer_name in name)):
            # layer_dict[name + '-' + str(idx)] = module
            # idx += 1
        # else:
            # for name_2, module_2 in module.named_children():
                # for name_3, module_3 in module_2.named_children():
                    # if (not isinstance(module_3, nn.Sequential)
                        # and not isinstance(module_3, nn.BatchNorm2d)
                        # and not isinstance(module, nn.Dropout)
                        # and not isinstance(module, nn.ReLU)
                        # and 'shortcut' not in name_3
                        # and (layer_name is None or layer_name in name_3)):
                        # layer_dict[name_3 + '-' + str(idx)] = module_3
                        # idx += 1    
                        #
    # return layer_dict





def get_layer_output_sizes(model, data, layer_name=None):   
    output_sizes = {}
    hooks = []  
    
    layer_dict = get_model_layers(model)
 
    def hook(module, input, output):
        module_idx = len(output_sizes)
        m_key = list(layer_dict)[module_idx]
        output_sizes[m_key] = list(output.size()[1:])      
    
    for name, module in layer_dict.items():
        hooks.append(module.register_forward_hook(hook))
    
    try:
        model(data[:1])  
    finally:
        for h in hooks:
            h.remove() 
            
    return output_sizes


def get_init_dict(model, data, init_value=False, layer_name=None,
                  fetch_func=lambda x:x[0].to(device)): 

    if not torch.is_tensor(data) :
        x_data =  next(iter(data))
        x_data = fetch_func(x_data)
    else :
        x_data = data 
    
    
    assert  torch.is_tensor(x_data),type(x_data)
    assert  x_data.ndim==4, f"epxect bcwh, but get {x_data.shape}" 
    x_data= x_data[:1] ## just only one 
    
    _,output_value= get_layer_output(model, x_data, layer_name)       
    output_sizes = {k:v.shape[1:] for k,v in  output_value .items() } 
    
    model_layer_dict = {}  
    for layer, output_size in output_sizes.items():
        for index in range(np.prod(output_size)):
            # since we only care about post-activation outputs
            model_layer_dict[(layer, index)] = init_value
    return model_layer_dict

# def get_init_maxtrix(model, data, init_value=False, layer_name=None,
                  # fetch_func=lambda x:x[0].to(device)): 
                  #
    # if not torch.is_tensor(data) :
        # x_data =  next(iter(data))
        # x_data = fetch_func(x_data)
    # else :
        # x_data = data 
        #
        #
    # assert  torch.is_tensor(x_data),type(x_data)
    # assert  x_data.ndim==4, f"epxect bcwh, but get {x_data.shape}" 
    # x_data= x_data[:1] ## just only one 
    #
    # _,output_value= get_layer_output(model, x_data, layer_name)       
    # output_sizes = {k:v.shape[1:] for k,v in  output_value .items() } 
    # print (output_sizes)
    # for k,v in output_value.items():
        # print (k,"->",v.shape)
        #
    # model_layer_dict = {}  
    # for layer, output_size in output_sizes.items():
        # for index in range(np.prod(output_size)):
            # # since we only care about post-activation outputs
            # model_layer_dict[(layer, index)] = init_value
    # return model_layer_dict





# def extract_outputs(model, data, module, force_relu=True):
    # outputs = []      
    # def hook(module, input, output):
        # if force_relu:
            # outputs.append(torch.relu(output))   
        # else:
            # outputs.append(output)
    # handle = module.register_forward_hook(hook)     
    # logits = model(data)
    # handle.remove()
    # return logits , torch.stack(outputs)
def extract_outputs(model, data, module,force_relu=True,clear_method=lambda x:x.detach().cpu() ):
    outputs = []      
    def hook(module, input, output):
        if force_relu:
            outputs.append(clear_method(torch.relu(output)) )  
        else:
            outputs.append(clear_method(output))
    handle = module.register_forward_hook(hook)     
    logits = model(data)
    handle.remove()
    return logits , torch.cat (outputs,dim=0)





def scale(out, rmax=1, rmin=0):
    output_std = (out - out.min()) / (out.max() - out.min())
    output_scaled = output_std * (rmax - rmin) + rmin
    return output_scaled



import tqdm 
def batch_data(data_tensor_or_not):
    def is_torch(x):
        return torch.is_tensor(x)
    def is_np(x):
        return type(x)==np.ndarray
    def to_torch(x):
        if not is_torch(x):
            return torch.from_numpy(x)
        return x 
    def is_torch_dataloader(x):
        return "DataLoader" in str(type(x))
    
    if is_torch_dataloader(data_tensor_or_not):
        return data_tensor_or_not
    
    dataset =data_tensor_or_not 
    if type(data_tensor_or_not) in [list,tuple] and ( is_torch(data_tensor_or_not[0]) or is_np(data_tensor_or_not[0]) ):
        
        dataset = [to_torch(x) for x in data_tensor_or_not]
        dataset = torch.utils.data.TensorDataset(*dataset)
    elif is_torch(data_tensor_or_not):
        dataset = [data_tensor_or_not]
        dataset = torch.utils.data.TensorDataset(*dataset)
    elif is_np(data_tensor_or_not):
        dataset = [to_torch(data_tensor_or_not)]
        dataset = torch.utils.data.TensorDataset(*dataset)
    
    batch_size = os.environ.get("batch_size",None)
    if batch_size is None :
        batch_size = len(data_tensor_or_not)
    batch_size = int(batch_size)
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=  batch_size,
        )

def tensor2loader(data_tensor_or_not):
    return batch_data(data_tensor_or_not)

# def tensor2loader(data_tensor_or_not):
    # batch_size = os.environ.get("batch_size",None)
    # if batch_size is None :
        # batch_size = len(data_tensor_or_not)
    # batch_size = int(batch_size)
    #
    #
    # return torch.utils.data.DataLoader(
        # data_tensor_or_not,
        # batch_size=  batch_size,
        # # pin_memory=True,
        # )
# def batch_data(data_tensor_or_not):
    # if torch.is_tensor(data_tensor_or_not):
        # data_tensor_or_not= torch.utils.data.TensorDataset(data_tensor_or_not),
        # return tensor2loader(data_tensor_or_not)  
    # return data_tensor_or_not

# def debug_mem(s=""):
    # print("=="*8,s)
    # t = torch.cuda.get_device_properties(0).total_memory
    # r = torch.cuda.memory_reserved(0) 
    # a = torch.cuda.memory_allocated(0)
    # f = r-a  # free inside reserved    
    # print (f)
    #
def collect_fn_grad(batch,clear_method=lambda x:x.detach().cpu(),**kwargs):
    return collect_fn(batch=batch,clear_method=clear_method,**kwargs)

def collect_fn (batch,clear_method=lambda x:x.detach().cpu(),**kwargs):
    
    def collect_fn_dict(logits_out_one,clear_method=lambda x:x.detach().cpu(), **kwargs):
        logits_out_list= {}
        
        for dict_item in logits_out_one:
            for k,v in  dict_item.items():
                if k in logits_out_list:
                    vv = logits_out_list[k] 
                    v1  = vv+[v]
                else :
                    v1 =  [v]
                logits_out_list.update({k:v1})
            
        for k,v in logits_out_list.items():
            if len(v)>0:
                if torch.is_tensor(v[0]):
                    # verbose=[x.shape for x in v ]
                    verbose= torch.cat(v)
                else :
                    verbose=np.concatenate(v)
            logits_out_list[k]= clear_method(verbose)
        return logits_out_list
    
    def collect_fn_list(logits_out_one,clear_method=lambda x:x.detach().cpu(), **kwargs):
        if len(logits_out_one)>0:
            if torch.is_tensor(logits_out_one[0]):
                verbose= torch.cat(logits_out_one)
            else :
                verbose=np.concatenate(logits_out_one)
            logits_out_one= clear_method(verbose)
            
        return logits_out_one

    if len(batch)<=0:
        return batch
    if type(batch[0])==dict :
        return collect_fn_dict(logits_out_one=batch, clear_method=clear_method)
    if torch.is_tensor(batch[0]) or type(batch[0])==np.ndarray :
        return collect_fn_list(logits_out_one=batch, clear_method=clear_method)
    raise Exception(f"unkown input,{type(batch)}.. {type(batch[0])} ")



def get_layer_output_batch(model, data, layer_name=None,fetch_func= lambda x:x[0],**kwargs ):   
    
    
    dataloader  = batch_data(data)
    
    collect_list1 = []
    collect_list2 = []
    
    layer_dict = get_model_layers(model,**kwargs)

    with torch.no_grad():
    
        for  one_data in  dataloader:
            one_data = fetch_func(one_data)
            # print (one_data.shape)
            ret1,ret2 = get_layer_output(model=model,data=one_data, layer_name=layer_name,layer_dict=layer_dict,**kwargs)
            # if type(ret2)==dict:
                # for k,v in ret2.items():
                    # print (v.shape,"-->",k)
                    #
            collect_list1.append(ret1)
            collect_list2.append(ret2)
    
    collect_list1 = collect_fn(collect_list1,**kwargs)
    collect_list2 = collect_fn(collect_list2,**kwargs)

    return collect_list1 , collect_list2


def get_layer_output_batch_v2(model, data, layer_name=None,fetch_func= lambda x:x[0],**kwargs ):   
    return get_layer_output_batch_withreduce(
        model = model ,
        data=data, 
        layer_name=layer_name,
        fetch_func=fetch_func,
        **kwargs
        )
def get_layer_output_batch_withreduce(model, data, layer_name=None,fetch_func= lambda x:x[0],**kwargs ):   

    dataloader  = batch_data(data)
    
    collect_list1 = []
    collect_list2 = []
    
    layer_dict = get_model_layers(model,**kwargs)

    with torch.no_grad():
        for  one_data in  dataloader:
            one_data = fetch_func(one_data)
            # print (one_data.shape)
            ret1,ret2 = get_layer_output_reduce(model=model,data=one_data, layer_name=layer_name,layer_dict=layer_dict,**kwargs)
            #
            collect_list1.append(ret1)
            collect_list2.append(ret2)
    
    collect_list1 = collect_fn(collect_list1,**kwargs)
    collect_list2 = collect_fn(collect_list2,**kwargs)

    return collect_list1 , collect_list2


def extract_outputs_batch(model, data, module, force_relu=True,
                          fetch_func=lambda x:x[0],
                          clear_method=lambda x:x.detach().cpu(),
                          ):
    dataloader  = batch_data(data)
    
    collect_list1 = []
    collect_list2 = []
    # print ("* extract_outputs_batch","start 1 ")
    with torch.no_grad():
        for  one_data in  dataloader:
            # print ("* extract_outputs_batch","start 1.1 ")
            one_data = fetch_func(one_data)
            # print (one_data.shape)
            ret1,ret2 = extract_outputs(model, data=one_data, module=module, force_relu=force_relu)
            # if type(ret2)==dict:
                # for k,v in ret2.items():
                    # print (v.shape,"-->",k)
                    
            collect_list1.append(ret1)
            collect_list2.append(ret2)
    
    # print ("* extract_outputs_batch","start 2 ")
    collect_list1 = collect_fn(collect_list1,clear_method=clear_method)
    # print ("* extract_outputs_batch","start 2 ")
    collect_list2 = collect_fn(collect_list2,clear_method=clear_method)
    # print ("* extract_outputs_batch","start 3 ")
    
    # print ("collect_list1,2",collect_list1.shape,collect_list2.shape)
    # if type(collect_list2)==dict:
        # for k,v in collect_list2.items():
            # print (v.shape,"-!->",k)
#
    # print ("collect_list2",type(collect_list2),)
    # if torch.is_tensor(collect_list2):
        # print (collect_list2.shape,"collect_list2.shape")

    return collect_list1 , collect_list2

def predict_batch(model, data, fetch_func= lambda x:x[0],**kwargs ):   
    
    dataloader  = batch_data(data)
    
    collect_list1 = []
    # collect_list2 = []
    
    with torch.no_grad():
    
        for  one_data in  dataloader:
            one_data = fetch_func(one_data)
            one_data = one_data .to(device)

            ret1 =model.forward(one_data)
            
            collect_list1.append(ret1)
    
    collect_list1 = collect_fn(collect_list1,**kwargs)

    # collect_list1.squeeze_(dim=0)

    return collect_list1
def predict_batch_enable_grad(model, data, fetch_func= lambda x:x[0],**kwargs ):   
    
    dataloader  = batch_data(data)
    
    collect_list1 = []
    # collect_list2 = []
    
    for  one_data in  dataloader:
        one_data = fetch_func(one_data)
        one_data = one_data .to(device)
        ret1 =model.forward(one_data)
        
        collect_list1.append(ret1)
    
    collect_list1 = collect_fn_grade(collect_list1,**kwargs)

    # collect_list1.squeeze_(dim=0)

    return collect_list1


# if __name__=="__main__":
    # import torchvision 
    # import os 
    # os.environ["batch_size"]="128"
    # device=torch.device("cuda")
    #
    # model = torchvision.models.alexnet()
    # model = model.to(device)
    #
    # data = torch.randn(400,3,224,224)
    # data = data.to(device)
    # from tqdm import tqdm 
    #
    # layer_dict = get_model_layers(model) 
    # for layer, module in tqdm(layer_dict.items()): 
        # _,v1 = extract_outputs_batch(model, data, module)
        # print(layer,"v1",v1.shape,"v1")
        # # outputs = torch.squeeze(torch.sum(v1, dim=1))
        # # # print (type(outputs),"-->")
        # # print (outputs.shape)
    # # # extract_outputs
    # print ("========"*10)
    # print ("get_layer_output_batch")
    # # import os 
    # # os.environ["batch_size"]="128"
    # # device=torch.device("cuda")
    # #
    # # import torchvision 
    # # device=torch.device("cuda")
    # #
    # # model = torchvision.models.alexnet()
    # # model = model.to(device)
    #
    # data = torch.randn(400,3,224,224)
    # data = data.to(device)
    # from tqdm import tqdm 
    #
    # v0,v1 = get_layer_output_batch(model, data)
    # # print (type(v0),type(v1))
    # print ("v0",v0.shape)
    # for k,v in v1.items():
        # print (k,":",v.shape)
