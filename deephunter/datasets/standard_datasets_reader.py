import torchvision.transforms as transforms 
import torch 
import torchvision

def get_dataset(dataset_name="mnist",root=None,
                 is_train=False,
                 is_download=True,):
    '''
    support four datasets' reader, ie: mnist, cifar10, svhn and imagenet(10classes)
    '''
    if root is None :
        root = os.path.dirname(os.path.abspath(__file__))
        root = os.path.join(root,"../..")
        
    dataset=  _get_dataset(dataset_name=dataset_name,root=root,
                 is_train=is_train,
                 is_download=is_download,)
    return dataset 

def _get_dataset(dataset_name="mnist",
                 root=".",
                 is_train=False,
                 is_download=True,):
    
    dataset=None 
    
    if dataset_name == "mnist":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
         ])
        dataset = torchvision.datasets.MNIST(
            root=os.path.join(root,'./data/data_files/datasets/'), 
            train=is_train, 
            download=is_download, 
            transform=transform_test)

    if dataset_name == "cifar10":
        transform_test = transforms.Compose([
                transforms.ToTensor(),
        ])
        
        dataset = torchvision.datasets.CIFAR10(
                root = os.path.join(root,'./data/data_files/datasets/cifar-10'),
                train = is_train,
                transform = transform_test,
                download = is_download)
    if dataset_name == "imagenet":
        '''
        this is a intranet path  
        for any imagenet dataset's issues, please contact us by github 
        '''
        if is_train:
            valdir =os.path.join(root, "./data/data_files/datasets/10Class_imagenet/train/")
        else:
            valdir =os.path.join(root, "./data/data_files/datasets/10Class_imagenet/val/")
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        dataset = test_dataset = imagenet10Folder (
            valdir,
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ]))
    if dataset_name == "SVHN":
        transform_test = transforms.Compose([
                transforms.ToTensor(),
        ])
        
        dataset = torchvision.datasets.SVHN(
                root = os.path.join(root,'./data/data_files/datasets/SVHN'),
                split="train",
                transform = transform_test,
                download = is_download)
    return dataset 