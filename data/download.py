

"""
download the weights from drive.google.com. 
warning:
  if you cannot access the google.com ,please use a VPN or contact us to help you. 
  if you are out of quato, please wait to next day or contact us to help too. 
"""

import torchvision.datasets.utils as dtutil
from tqdm import tqdm 
import os 

file_id_list={
    "covnet_mnist":("1rUzzcvG7R55TvJVpdOaqgV3OB0kRznWA","mnist_mixup_acc_99.28_ckpt.pth"),
    "covnet_cifar10":("1zK66FYCGu5zxbIcAx-f8PRPY-cQ_sBrE","cifar_mixup_acc_90.36_ckpt.pth"),
    "vgg_cifar10":("1Ys3-0QuxN6tbzcAh_nlNHJcrX6pVc-r2","vgg_seed32_dropout.pkl"),
    "vgg_imagenet":("17SVIR3TD36Z7lZDgfhJ9gPSf9I8IQNj5","vgg16_bn_lr0.0001_49_imagenet_train_layer-1_withDataAugment.pkl"), #target_classes = self.classes[:10]
    # "vgg_imagenet":"14sp16y9GwZTqIUpXX_xldUi4tdSon_bM",# target_classes = [5, 93, 186, 282, 322, 409, 454, 527, 659, 817]
    
    "alexnet_svhn":("1RrwV1O0fciEXvdFSJipZLoM0hFtpT9k8","alexnet_lr0.0001_39.pkl"),
    }

##
# 
file_id_list_datasets ={
    "datasets":(
         "1vmESmVgWiYe5vYarZIPN7oz-6pcovvMG", "dataset_mnist_cifar10_svhn_10classimagenet.tar"),
    }

def download_func(file_id,
                  final_name,
                  cache_dir="./",
                  extract=False,
                  ):
    # if cache_dir is None :
        # cache_dir ="./"# os.path.dirname(final_name)
        
    dtutil.download_file_from_google_drive(file_id=file_id, root=cache_dir, filename=final_name)
    # net.load_state_dict(torch.load(f"{cache_dir}/torch_best_{final_name}_{file_id}.pth"))
    if extract :
        from_path = os.path.join(cache_dir,final_name)
        to_path = os.path.join(cache_dir)
        dtutil.extract_archive(from_path, to_path, remove_finished=True)

import argparse 

parser = argparse.ArgumentParser(description=' download')
parser.add_argument('--download_weight', action='store_true')
parser.add_argument('--download_datasets', action='store_true')
parser.add_argument('--file_id',"-fid", type=str, default=None)
parser.add_argument('--save_name', type=str, default="./trained_models")
args = parser.parse_args()



if __name__=="__main__":
    
    if args.download_weight :
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"./trained_models")
        # cache_dir= os.path.expanduser("./trained_models")
        for name,(file_id,file_save_name) in tqdm(file_id_list.items()):
            download_func(file_id=file_id,final_name=file_save_name,cache_dir=cache_dir)
            print (f"finished download of {name}")
    if args.download_datasets :
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"./data_files")
        # cache_dir= os.path.expanduser("./trained_models")
        for name,(file_id,file_save_name) in tqdm(file_id_list_datasets.items()):
            download_func(file_id=file_id,final_name=file_save_name,cache_dir=cache_dir,extract=True)
            print (f"finished download of {name}")
    
    if args.file_id and args.save_name :
        download_func(file_id=args.file_id,final_name=args.save_name)
    
    