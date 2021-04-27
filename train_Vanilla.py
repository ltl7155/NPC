import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import sys
sys.path.append("./models/")

import pickle
import random
from imagenet10Folder import imagenet10Folder 
import torch.backends.cudnn as cudnn
from VGG_16 import VGG16
from vgg import vgg16_bn
import torchvision.models as torch_models
#basic setting


def train(model):
    if args.dataset == "cifar10":
        transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
    #             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
                transforms.ToTensor(),
    #             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_data = torchvision.datasets.CIFAR10(
                root = './data/cifar-10',
                train = True,
                transform = transform_train,
                download = True
        )
        test_data = torchvision.datasets.CIFAR10(
                root = './data/cifar-10',
                train = False,
                transform = transform_test,
                download = True
        )
        train_loader = Data.DataLoader(dataset=train_data, batch_size=args.batchsize, shuffle=True)
        test_loader = Data.DataLoader(dataset=test_data, batch_size=args.batchsize, shuffle=False)
        
    elif args.dataset == "cifar100":
        transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
    #             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
                transforms.ToTensor(),
    #             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_data = torchvision.datasets.CIFAR100(
                root = './data/cifar-100',
                train = True,
                transform = transform_train,
                download = True
        )
        test_data = torchvision.datasets.CIFAR100(
                root = './data/cifar-100',
                train = False,
                transform = transform_test,
                download = True
        )
        train_loader = Data.DataLoader(dataset=train_data, batch_size=args.batchsize, shuffle=True)
        test_loader = Data.DataLoader(dataset=test_data, batch_size=args.batchsize, shuffle=False)
        
    elif args.dataset == "imagenet":
        train_dir = "/mnt/dataset/Image__ILSVRC2012/ILSVRC2012_img_train/train/"
        val_dir = "/mnt/mfs/litl/ICSE_CriticalPath/data/ILSVRC2012_img_val/"
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        train_dataset = imagenet10Folder(
            train_dir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
#                 transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ]))
        test_dataset = imagenet10Folder(
            val_dir,
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ]))
        train_loader = Data.DataLoader(dataset=train_dataset, batch_size=args.batchsize, shuffle=True)
        test_loader = Data.DataLoader(dataset=test_dataset, batch_size=args.batchsize, shuffle=False)
        
    elif args.dataset == "SVHN":
        if args.noDataAug:
            transform_train = transforms.Compose([
                    transforms.ToTensor(),
        #             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            transform_train = transforms.Compose([

                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
        #             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        transform_test = transforms.Compose([
                transforms.ToTensor(),
    #             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_data = torchvision.datasets.SVHN(
                root = './data/SVHN',
                split = 'train',
                transform = transform_train,
                download = True
        )
        test_data = torchvision.datasets.SVHN(
                root = './data/SVHN',
                split = 'test',
                transform = transform_test,
                download = True
        )
        train_loader = Data.DataLoader(dataset=train_data, batch_size=args.batchsize, shuffle=True)
        test_loader = Data.DataLoader(dataset=test_data, batch_size=args.batchsize, shuffle=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if args.overfit:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
        
    loss_func = nn.CrossEntropyLoss()

    top0 = AverageMeter()
    losses = AverageMeter()
    
#     log_dict = {'Loss': losses.val, 'top0_prec': top0.val}
#     set_tensorboard(log_dict, 0, logger_train)
    
    model.train()
    for epoch in range(args.epoch):
        adjust_learning_rate(optimizer, epoch)
        for step, (x, y) in enumerate(train_loader):
            train_x = x.cuda()
            train_y = y.cuda()

            logits = model(train_x)
            total_loss = loss_func(logits, train_y)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            train_output = logits
            train_loss = loss_func(train_output, train_y)
            _, train_pred_y = train_output.max(1)
            Accuracy = float(train_pred_y.eq(train_y).sum().item()) / float(train_y.size(0)) * 100.0
            if (step+1) % 10 == 0:
                print("epoch " + str(epoch) + " step" + str(step), 
                      '| train loss: %.4f' % train_loss.item(), '| train accuracy: %.2f%%' % Accuracy)
            top0.update(Accuracy, train_x.size(0))
            losses.update(train_loss.item(), train_x.size(0))

        print('epoch ' + str(epoch))
        total = 0
        correct = 0
        model.eval()
        with torch.no_grad():
            for test_step,(val_x, val_y) in enumerate(test_loader):
                val_x = val_x.cuda()
                val_y = val_y.cuda()
                val_output = model(val_x)
                _,val_pred_y = val_output.max(1)
                correct += val_pred_y.eq(val_y).sum().item()
                total += val_y.size(0)
        result = float(correct) * 100.0 / float(total)
        print('val accuracy: %.2f%%' % result) 
        model.train()    
        save_epochs = [round(args.epoch/3), round(args.epoch*2/3), args.epoch-1]
        
        if epoch in save_epochs:
            if args.noDataAug:
                suffix = str(epoch) + "_noDataAug" + args.suffix
            else: 
                suffix = str(epoch) + "_" + args.dataset + args.suffix
            print('saving model...')
            save_model_path = args.save_model_path + args.model_type + "_lr" + str(args.lr) + "_" + suffix +  "_train_layer" + str(args.train_layer) + "_withDataAugment.pkl"
            torch.save(model.state_dict(), save_model_path)
        
        log_dict = {'top0_prec': result}
#         set_tensorboard(log_dict, epoch, logger_val)
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (args.decay_factor ** (epoch // args.epoch_lr_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
                
if __name__ == "__main__":
    #arguments
    parser = argparse.ArgumentParser(description='model interpretation')
#     parser.add_argument('--load_model_path', default="./tes.pkl", help='load model path')
    parser.add_argument('--model_type', default="vgg", help='model type')
    parser.add_argument('--save_model_path', default="./trained_models/", help='save model path')
#     parser.add_argument('--train_data_path', default="./data/cifar-10", help='training dataset path')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--batchsize', type=int, default=64, help='training batch size')
    parser.add_argument('--epoch', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--decay_factor', type=int, default=0.3)
    parser.add_argument('--epoch_lr_step', type=int, default=30)
    parser.add_argument('--train_layer', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=32, help='training batch size')
    parser.add_argument('--suffix', type=str, default="", help='training batch size')
    parser.add_argument('--dataset', type=str, default="cifar100")
    parser.add_argument('--gpu', type=str, default="3")
    parser.add_argument('--noDataAug', action='store_true')
    parser.add_argument('--overfit', action='store_true')
#     parser.add_argument('--dropout', action='store_true')
#     parser.add_argument('--init', action='store_true')
#     parser.add_argument('--version2', action='store_true')
    args = parser.parse_args()
    #print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
#     if args.seed is not None:
#         random.seed(args.seed)
#         torch.manual_seed(args.seed)
#         torch.cuda.manual_seed(args.seed)
#         cudnn.deterministic = True
    
    if "vgg" in args.model_type:
        if args.dataset == "imagenet":
            net = vgg16_bn(num_classes=10).cuda()
        elif args.dataset == "cifar100":
            net = VGG16(num_classes=100).cuda()
        else:
            net = VGG16(num_classes=10).cuda()
    elif args.model_type == "alexnet":
        if args.overfit:
            from AlexNet_SVHN_noDropout import AlexNet
            net = AlexNet(num_classes=10).cuda()
        else:
            if args.dataset == "imagenet":
                net = torch_models.alexnet(num_classes=10).cuda()
            else:
                from AlexNet_SVHN import AlexNet
                net = AlexNet(num_classes=10).cuda()
    #torch.save(net.state_dict(), args.load_model_path)
#     print(net)           
    train(net)