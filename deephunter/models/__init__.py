import os 
import torch 
import traceback

'''
1,mnist
2,cifar
3,cifar_vgg
4,svhn
5,vgg16_bn

'''



file_id_list={
    # "covnet_mnist":"1rUzzcvG7R55TvJVpdOaqgV3OB0kRznWA",
    # "covnet_cifar10":"1zK66FYCGu5zxbIcAx-f8PRPY-cQ_sBrE",
    # "vgg_cifar10":"1Ys3-0QuxN6tbzcAh_nlNHJcrX6pVc-r2",
    # "vgg_imagenet":"17SVIR3TD36Z7lZDgfhJ9gPSf9I8IQNj5", #target_classes = self.classes[:10]
    # # "vgg_imagenet":"14sp16y9GwZTqIUpXX_xldUi4tdSon_bM",# target_classes = [5, 93, 186, 282, 322, 409, 454, 527, 659, 817]
    # "alexnet_svhn":"1RrwV1O0fciEXvdFSJipZLoM0hFtpT9k8",

    "covnet_mnist":("1rUzzcvG7R55TvJVpdOaqgV3OB0kRznWA","mnist_mixup_acc_99.28_ckpt.pth"),
    "covnet_cifar10":("1zK66FYCGu5zxbIcAx-f8PRPY-cQ_sBrE","cifar_mixup_acc_90.36_ckpt.pth"),
    "vgg_cifar10":("1Ys3-0QuxN6tbzcAh_nlNHJcrX6pVc-r2","vgg_seed32_dropout.pkl"),
    "vgg_imagenet":("17SVIR3TD36Z7lZDgfhJ9gPSf9I8IQNj5","vgg16_bn_lr0.0001_49_imagenet_train_layer-1_withDataAugment.pkl"), #target_classes = self.classes[:10]
    # "vgg_imagenet":"14sp16y9GwZTqIUpXX_xldUi4tdSon_bM",# target_classes = [5, 93, 186, 282, 322, 409, 454, 527, 659, 817]
    
    "alexnet_svhn":("1RrwV1O0fciEXvdFSJipZLoM0hFtpT9k8","alexnet_lr0.0001_39.pkl"),

#compatable


    }
map_dict = {
    "convmnist":"covnet_mnist",
    "mnist":"covnet_mnist",
    "convcifar10":"covnet_cifar10",
    "vgg":"vgg_cifar10",
    "vgg16_bn":"vgg_imagenet",
    "alexnet":"alexnet_svhn",
    "svhn":"alexnet_svhn",
    "cifar_vgg":"vgg_cifar10",
    "cifar":"covnet_cifar10",
    "imagenettianli_vgg16bn":"vgg_imagenet",
    "imagenet_vgg16bn":"vgg_imagenet",
    }
name_list= list(file_id_list.keys())


default_trained_dir=os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../",
    "data/trained_models/")

def get_masked_net(name="",
            dt_name=None,
            device=torch.device("cuda"),
            use_cache=True,
            cache_dir=default_trained_dir,
            prefix_lambda=lambda x:f"torch_best_{x[0]}_{x[1]}.pth",
            ):
    
    return get_net(
        name=name,
        dt_name=dt_name,
        device=device,
        use_cache=use_cache,
        cache_dir=cache_dir,
        prefix_lambda=prefix_lambda,
        is_masked=True,
        )
    
    
def get_net(name="",
            dt_name=None,
            device=torch.device("cuda"),
            use_cache=True,
            cache_dir=default_trained_dir,
            prefix_lambda=lambda x:f"torch_best_{x[0]}_{x[1]}.pth",
            is_masked=False,
            ):
    
    name = map_dict.get(name,name)
        
    assert  name in name_list ,        print (f"expect the name contained in {name_list}, but get {name}" )
    
    dt_name =None if "_" not in  name else name.split("_")[-1]  
    name =None if "_" not in  name else name.split("_")[0]  

    if is_masked:
        net = _get_masked_net(name=name,dt_name=dt_name)
    else:
        net = _get_net(name=name,dt_name=dt_name)
        
    assert net is not None , f"name =={name},{dt_name} net none"


    if use_cache:
        import torchvision.datasets.utils as dtutil
        final_name = f"{name}_{dt_name}"
        (file_id,filename) = file_id_list[final_name]
        #filename = prefix_lambda((final_name,file_id))
        dtutil.download_file_from_google_drive(file_id=file_id, root=cache_dir, filename=filename)
        net.load_state_dict(torch.load(f"{cache_dir}/{filename}"))

        setattr(net,"fid",file_id)
        # except Exception as ex :
            # traceback.print_exc()
            # print (f"{cache_dir}/torch_best_{final_name}_{file_id}.pth")
            # print (net)
    net= net.to(device)
    net.eval()
    
    # net.feature_list = types.MethodType(feature_list,net)

    
    return net 




def _get_net(name="",dt_name="cifar10"):
    net =None 
    if name=="alexnet":
        from .alexnet import AlexNet
        net = AlexNet()
    if name=="covnet":
        if dt_name=="cifar10":
            from .covnet_cifar10 import ConvnetCifar
            net = ConvnetCifar()
        if dt_name=="mnist":
            from .covnet_mnist import ConvnetMnist
            net = ConvnetMnist()

    if name=="vgg":
        if dt_name=="cifar10":
            from  .vgg_cifar10 import VGG16 
            net = VGG16(num_classes=10)
        if dt_name=="imagenet":
            from .vgg_imagenet import vgg16_bn
            
            net = vgg16_bn(num_classes=10)

    return net 



def _get_masked_net(name="",dt_name="cifar10"):
    net =None 
    if name=="alexnet":
        from .masked.mask_AlexNet_SVHN import mask_AlexNet as AlexNet
        net = AlexNet()
    if name=="covnet":
        if dt_name=="cifar10":
            from .masked.mask_sa_models import mask_ConvnetCifar as ConvnetCifar
            net = ConvnetCifar()
        if dt_name=="mnist":
            from .masked.mask_sa_models import mask_ConvnetMnist as ConvnetMnist
            net = ConvnetMnist()

    if name=="vgg":
        if dt_name=="cifar10":
            from .masked.mask_VGG_16 import mask_VGG16 as VGG16
            net = VGG16(num_classes=10)
        if dt_name=="imagenet":
            from .masked.mask_vgg_imagenet import mask_vgg16_bn as vgg16_bn
            net = vgg16_bn(num_classes=10)

    return net 

