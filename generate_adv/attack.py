#--------------------------------------------------------------------------
# Attack methods including FGSM, Step-LL, IFGSM, PGD and MIFGSM in PyTorch
#--------------------------------------------------------------------------

import torch 
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import numpy as np 
import pickle
import os
import argparse
from VGG_16_featuredict import *
# from ResNet18 import *
# from Inception_v2 import *
# import VGG
#GPUID = "6"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def get_bool(string):
    if(string == 'False'):
        return False
    else:
        return True
parser = argparse.ArgumentParser(description='attack implementation')
parser.add_argument('--attack', default='fgsm', help='attack type to be used(fgsm, ifgsm, step_ll, pgd....)')
parser.add_argument('--generate', type=get_bool, default=False, help='whether to generate adv examples as .p files')
# if use iterative attack , the droplast should be set as True when attacking ANP
parser.add_argument('--droplast', type=get_bool, default=False, help='whether to drop last batch in testloader')
parser.add_argument('--model', default='resnet', help='target model or model generate adv(resnet, vgg,...)')
parser.add_argument('--modelpath', default="../model_path/naive_param.pkl", help='target model path')
parser.add_argument('--dataroot', default="../data/train/mnist/", help='training data path')
parser.add_argument('--attack_batchsize', type=int, default=128, help='batchsize in Attack')
parser.add_argument('--attack_epsilon', type=float, default=8.0, help='epsilon in Attack')
parser.add_argument('--attack_alpha', type=float, default=2.0, help='alpha in Attack')
parser.add_argument('--attack_iter', type=int, default=10, help='iteration times in Attack')
parser.add_argument('--attack_momentum', type=float, default=1.0, help='momentum paramter in Attack')
parser.add_argument('--savepath', default="../save_path/test", help='saving path of clean and adv data')
parser.add_argument('--imgpath', default='../img_path/eps_0.031', help='images path')
parser.add_argument('--dataset', default='cifar10', help='dataset used for attacking')
args = parser.parse_args()
print(args.attack,args.modelpath)


class Attack():
    def __init__(self, dataroot, dataset, batch_size, target_model, criterion):
        self.dataroot = dataroot
        self.dataset = dataset
        self.batch_size = batch_size
        self.model = target_model
        self.criterion = criterion
        
    # root of MNIST/CIFAR-10 testset
    def return_data(self):
        if self.dataset == 'mnist':
            test_dataset = torchvision.datasets.MNIST(root=self.dataroot,train=False, transform=transforms.ToTensor())
        elif self.dataset == 'cifar10':
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            test_dataset = torchvision.datasets.CIFAR10(root=self.dataroot,train=False, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=self.batch_size,shuffle=False,drop_last=args.droplast)
        return test_loader
        
    def return_trainloader(self):
        if self.dataset == 'cifar10':
            train_dataset = torchvision.datasets.CIFAR10(root=self.dataroot,train=True, transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=self.batch_size,shuffle=True)
        return train_loader
        
    def fgsm(self,epsilon):
        data_loader = self.return_data()
        self.model.eval()

        correct = 0
        correct_cln = 0
        correct_adv = 0
        total = 0 
        for i, (images, labels) in enumerate(data_loader):
            x = Variable(images, requires_grad = True).cuda()
            y_true = Variable(labels, requires_grad = False).cuda()
            x.retain_grad()
            h,_ = self.model(x)
            _, predictions = torch.max(h,1)
            correct_cln += (predictions == y_true).sum()
            loss = self.criterion(h, y_true)
            self.model.zero_grad()
            if x.grad is not None:
                x.grad.data.fill_(0)
            loss.backward()
            
            #FGSM
            #x.grad.sign_()   # change the grad with sign ?
            x_adv = x.detach() + epsilon * torch.sign(x.grad)
            x_adv = torch.clamp(x_adv,0,1)
            
            h_adv,_ = self.model(x_adv)
            _, predictions_adv = torch.max(h_adv,1)
            correct_adv += (predictions_adv == y_true).sum()
            if i == 0:
                test_data_cln = x.data.detach()
                test_data_adv = x_adv.data
                test_label = labels
                test_label_adv = predictions_adv
            else:
                test_data_cln = torch.cat([test_data_cln, x.data.detach()],0)
                test_data_adv = torch.cat([test_data_adv, x_adv.data.detach()],0)
                test_label = torch.cat([test_label, labels],0)
                test_label_adv = torch.cat([test_label_adv, predictions_adv],0)

            #print(test_data_cln.size(),test_data_adv.size(),test_label.size())

            correct += (predictions_adv == predictions).sum()
            total += len(predictions)
        
        self.model.train()
        #print("Error Rate is ", float(total-correct)*100/total)
        print("Before FGSM the accuracy is", float(100*correct_cln)/total)
        print("After FGSM the accuracy is", float(100*correct_adv)/total)

        return test_data_cln, test_data_adv, test_label, test_label_adv

    def i_fgsm(self,epsilon,alpha,iteration):
        test_loader = self.return_data()
        self.model.eval()

        correct = 0
        correct_cln = 0
        correct_adv = 0
        total = 0
        for i,(images,labels) in enumerate(test_loader):
            x = Variable(images, requires_grad = True).cuda()
            x.retain_grad()
            y_true = Variable(labels, requires_grad = False).cuda()
            x_adv = Variable(x.data, requires_grad=True).cuda()
            x_adv.retain_grad()

            h,_ = self.model(x)
            _, predictions = torch.max(h,1)
            correct_cln += (predictions == y_true).sum()

            for j in range(0,iteration):
                h_adv,_ = self.model(x_adv)

                loss = self.criterion(h_adv, y_true)
                self.model.zero_grad()
                if x_adv.grad is not None:
                    x_adv.grad.data.fill_(0)
                loss.backward()
                
                #I-FGSM
                #x_adv.grad.sign_()   # change the grad with sign ?
                if not args.attack == 'bpda_all':
                    print(type(x_adv.grad),x_adv.grad.size())
                x_adv = x_adv.detach() + alpha * torch.sign(x_adv.grad)
                # according to the paper of Kurakin:
                x_adv = torch.where(x_adv > x+epsilon, x+epsilon, x_adv)
                x_adv = torch.clamp(x_adv, 0, 1)
                x_adv = torch.where(x_adv < x-epsilon, x-epsilon, x_adv)
                x_adv = torch.clamp(x_adv, 0, 1)
                x_adv = Variable(x_adv.data, requires_grad=True).cuda()

            h_adv,_ = self.model(x_adv)
            _, predictions_adv = torch.max(h_adv,1)
            correct_adv += (predictions_adv == y_true).sum()

            #print(x.data.size(),x_adv.data.size(),labels.size())
            if i == 0:
                test_data_cln = x.data.detach()
                test_data_adv = x_adv.data
                test_label = labels
                test_label_adv = predictions_adv
            else:
                test_data_cln = torch.cat([test_data_cln, x.data.detach()],0)
                test_data_adv = torch.cat([test_data_adv, x_adv.data.detach()],0)
                test_label = torch.cat([test_label, labels],0)
                test_label_adv = torch.cat([test_label_adv, predictions_adv],0)

            #print(test_data_cln.size(),test_data_adv.size(),test_label.size())

            correct += (predictions == predictions_adv).sum()
            total += len(predictions)
        
        self.model.train()
        #print("Error Rate is ",float(total-correct)*100/total)
        print("Before I-FGSM the accuracy is",float(100*correct_cln)/total)
        print("After I-FGSM the accuracy is",float(100*correct_adv)/total)

        return test_data_cln, test_data_adv, test_label, test_label_adv 

    def PGD(self,epsilon,alpha,iteration):
        test_loader = self.return_data()
        self.model.eval()
        correct = 0
        correct_cln = 0
        correct_adv = 0
        total = 0
        for i,(images,labels) in enumerate(test_loader):
            print("index", i)
            x = Variable(images, requires_grad = True).cuda()
            x.retain_grad()
            y_true = Variable(labels, requires_grad = False).cuda()
            h,_ = self.model(x)
            _, predictions = torch.max(h,1)
            correct_cln += (predictions == y_true).sum()
            x_rand = x.detach()
            # PGD
            x_rand = x_rand + torch.zeros_like(x_rand).uniform_(-epsilon,epsilon)
            x_rand = torch.clamp(x_rand,0,1)
            x_adv = Variable(x_rand.data, requires_grad=True).cuda()
            for j in range(0,iteration):
                #print('batch = {}, iter = {}'.format(i,j))
                h_adv,_ = self.model(x_adv)
                loss = self.criterion(h_adv, y_true)
                self.model.zero_grad()
                if x_adv.grad is not None:
                    x_adv.grad.data.fill_(0)
                loss.backward()
                
                #x_adv.grad.sign_()   # change the grad with sign ?
                #print(type(x_adv.grad),x_adv.grad.size())
                x_adv = x_adv.detach() + alpha * torch.sign(x_adv.grad)
                # according to the paper of Kurakin:
                x_adv = torch.where(x_adv > x+epsilon, x+epsilon, x_adv)
                x_adv = torch.clamp(x_adv, 0, 1)
                x_adv = torch.where(x_adv < x-epsilon, x-epsilon, x_adv)
                x_adv = torch.clamp(x_adv, 0, 1)
                x_adv = Variable(x_adv.data, requires_grad=True).cuda()
            #x_adv = transforms.ToPILImage('RGB')(x_adv.detach().cpu().squeeze(0))
            #x_adv = transforms.ToTensor()(x_adv).unsqueeze(0).cuda()
            h_adv,_ = self.model(x_adv)
            _, predictions_adv = torch.max(h_adv,1)
            correct_adv += (predictions_adv == y_true).sum()
            #print(x.data.size(),x_adv.data.size(),labels.size())
            if i == 0:
                test_data_cln = x.data.detach()
                test_data_adv = x_adv.data
                test_label = labels
                test_label_adv = predictions_adv
            else:
                test_data_cln = torch.cat([test_data_cln, x.data.detach()],0)
                test_data_adv = torch.cat([test_data_adv, x_adv.data.detach()],0)
                test_label = torch.cat([test_label, labels],0)
                test_label_adv = torch.cat([test_label_adv, predictions_adv],0)
            #print(test_data_cln.size(),test_data_adv.size(),test_label.size())
            correct += (predictions == predictions_adv).sum()
            total += len(predictions)
        
        self.model.train()

        #print("Error Rate is ",float(total-correct)*100/total)
        print("Before PGD the accuracy is",float(100*correct_cln)/total)
        print("After PGD the accuracy is",float(100*correct_adv)/total)
        return test_data_cln, test_data_adv, test_label, test_label_adv 
        
    def step_ll(self,epsilon):
        data_loader = self.return_data()
        self.model.eval()

        correct = 0
        correct_cln = 0
        correct_adv = 0
        total = 0 
        for i, (images, labels) in enumerate(data_loader):
            x = Variable(images, requires_grad = True).cuda()
            x.retain_grad()
            y_true = Variable(labels, requires_grad = False).cuda()

            h,_ = self.model(x)
            _, predictions = torch.max(h,1)
            # Step-LL
            _, predictions_ll = torch.min(h,1)
            correct_cln += (predictions == y_true).sum()
            loss = self.criterion(h, predictions_ll)
            self.model.zero_grad()
            if x.grad is not None:
                x.grad.data.fill_(0)
            loss.backward()
            
            #x.grad.sign_()   # change the grad with sign ?
            x_adv = x.detach() - epsilon * torch.sign(x.grad)
            x_adv = torch.clamp(x_adv,0,1)
            
            h_adv,_ = self.model(x_adv)
            _, predictions_adv = torch.max(h_adv,1)
            correct_adv += (predictions_adv == y_true).sum()
            #print(x.data.size(),x_adv.data.size(),labels.size())
            if i == 0:
                test_data_cln = x.data.detach()
                test_data_adv = x_adv.data
                test_label = labels
                test_label_adv = predictions_adv
            else:
                test_data_cln = torch.cat([test_data_cln, x.data.detach()],0)
                test_data_adv = torch.cat([test_data_adv, x_adv.data.detach()],0)
                test_label = torch.cat([test_label, labels],0)
                test_label_adv = torch.cat([test_label_adv, predictions_adv],0)

            #print(test_data_cln.size(),test_data_adv.size(),test_label.size())

            correct += (predictions_adv == predictions).sum()
            total += len(predictions)
        
        self.model.train()
        
        #print("Error Rate is ", float(total-correct)*100/total)
        print("Before Step-ll the accuracy is", float(100*correct_cln)/total)
        print("After Step-ll the accuracy is", float(100*correct_adv)/total)

        return test_data_cln, test_data_adv, test_label, test_label_adv
        
    def momentum_ifgsm(self,epsilon,alpha,iteration,attack_momentum):
        test_loader = self.return_data()
        self.model.eval()

        correct = 0
        correct_cln = 0
        correct_adv = 0
        total = 0
        for i,(images,labels) in enumerate(test_loader):
            x = Variable(images, requires_grad = True).cuda()#测试集的一批图     
            x.retain_grad()
            y_true = Variable(labels, requires_grad = False).cuda()#一批图片对应的label
            x_adv = Variable(x.data, requires_grad=True).cuda()#获取这批图片的副
            x_grad = torch.zeros(x.size()).cuda()#制造全0的与x大小一致的矩阵

            h,_ = self.model(x)#正常输出
            _, predictions = torch.max(h,1)#正常输出的分类结          
            correct_cln += (predictions == y_true).sum()#          
            _, predictions_ll = torch.min(h,1)#least-likely    
            #predictions_ll = torch.Tensor([0] * labels.size()[0])
            #predictions_ll = predictions_ll.long().cuda()
            for j in range(0,iteration):#迭代攻击，x_adv初始为原始数
                h_adv,_ = self.model(x_adv)#对抗样本输出

                loss = self.criterion(h_adv, predictions_ll)
                self.model.zero_grad()
                if x_adv.grad is not None:
                    x_adv.grad.data.fill_(0)
                loss.backward()#获取对抗样本到y_true
                #I-FGSM
                #x_adv.grad.sign_()   # change the grad with sign ?
                #print(type(x_adv.grad),x_adv.grad.size())

                # in Boosting attack
                # alpha = epsilon / iteration

                # calc |x|p except dim=0
                norm = x_adv.grad
                for k in range(1,4):
                    norm = torch.norm(norm,p=1,dim=k).unsqueeze(dim=k)#经过该步骤，x_adv的梯度转化为一个批*1*1*1的tensor
                #即后面三维度的数值总和
                # Momentum on gradient noise
                noise = attack_momentum * x_grad + x_adv.grad / norm#给后面三个维度归
                x_grad = noise

                x_adv = x_adv.detach() - alpha * torch.sign(noise)
                # according to the paper of Kurakin:
                x_adv = torch.where(x_adv > x+epsilon, x+epsilon, x_adv)
                x_adv = torch.clamp(x_adv, 0, 1)
                x_adv = torch.where(x_adv < x-epsilon, x-epsilon, x_adv)
                x_adv = torch.clamp(x_adv, 0, 1)
                x_adv = Variable(x_adv.data, requires_grad=True).cuda()

            h_adv,_ = self.model(x_adv)
            _, predictions_adv = torch.max(h_adv,1)
            correct_adv += (predictions_adv == y_true).sum()

            #print(x.data.size(),x_adv.data.size(),labels.size())
            if i == 0:
                test_data_cln = x.data.detach()
                test_data_adv = x_adv.data
                test_label = labels
                test_label_adv = predictions_adv
            else:
                test_data_cln = torch.cat([test_data_cln, x.data.detach()],0)
                test_data_adv = torch.cat([test_data_adv, x_adv.data.detach()],0)
                test_label = torch.cat([test_label, labels],0)
                test_label_adv = torch.cat([test_label_adv, predictions_adv],0)

            #print(test_data_cln.size(),test_data_adv.size(),test_label.size())

            correct += (predictions == predictions_adv).sum()
            total += len(predictions)
        
        self.model.train()
        #print("Error Rate is ",float(total-correct)*100/total)
        print("Before Momentum IFGSM the accuracy is",float(100*correct_cln)/total)
        print("After Momentum IFGSM the accuracy is",float(100*correct_adv)/total)
        print(test_data_adv.size())
        return test_data_cln, test_data_adv, test_label, test_label_adv 
        

def save_img(imgpath,test_data_cln, test_data_adv, test_label, test_label_adv):
    #save adversarial example
    if Path(imgpath).exists()==False:
        Path(imgpath).mkdir(parents=True)
    toImg = transforms.ToPILImage()
    image = test_data_cln.cpu()
    image_adv = test_data_adv.cpu()
    label = test_label.cpu()
    label_adv = test_label_adv.cpu()
    tot = len(image)
    batch = 10
    for i in range(0, batch):
        #print(image[i].size())
        im = toImg(image[i])
        #im.show()
        im.save(Path(imgpath)/Path('{}_label_{}_cln.jpg'.format(i,test_label[i])))
        im = toImg(image_adv[i])
        #im.show()
        im.save(Path(imgpath)/Path('{}_label_{}_adv.jpg'.format(i,test_label_adv[i])))

def display(test_data_cln, test_data_adv, test_label, test_label_adv):
    # display a batch adv
    toPil = transforms.ToPILImage()
    curr = test_data_cln.cpu()
    curr_adv = test_data_adv.cpu()
    label = test_label.cpu()
    label_adv = test_label_adv.cpu()
    disp_batch = 10
    for a in range(disp_batch):
        b = a + disp_batch 
        plt.figure()
        plt.subplot(121)
        plt.title('Original Label: {}'.format(label[a].cpu().numpy()),loc ='left')
        plt.imshow(toPil(curr[a].cpu().clone().squeeze()))
        plt.subplot(122)
        plt.title('Adv Label : {}'.format(label_adv[a].cpu().numpy()),loc ='left')
        plt.imshow(toPil(curr_adv[a].cpu().clone().squeeze()))
        plt.show()


def save_data_label(path, test_data_cln, test_data_adv, test_label, test_label_adv, eps):
    if os.path.exists(path) == False:
        os.makedirs(path)
    with open(path+'test_data_cln.pkl','wb') as f:
        pickle.dump(test_data_cln.cpu(), f, pickle.HIGHEST_PROTOCOL)

    with open(path+'test_adv(eps_{:.3f}).pkl'.format(eps),'wb') as f:
        pickle.dump(test_data_adv.cpu(), f, pickle.HIGHEST_PROTOCOL)

    with open(path+'test_label.pkl','wb') as f:
        pickle.dump(test_label.cpu(), f, pickle.HIGHEST_PROTOCOL)
    
    with open(path+'label_adv(eps_{:.3f}).pkl'.format(eps),'wb') as f:
        pickle.dump(test_label_adv.cpu(), f, pickle.HIGHEST_PROTOCOL)

def main():
    if args.model == 'vgg':
        print('attack vgg model')
        model = VGG16()
    elif(args.model=='inception'):
        print('attack inception model')
        model = Inception_v2()
    elif(args.model=='resnet'):
        print('attack resnet model')
        model = ResNet18()
    elif(args.model=='ANP_VGG'):
        model=VGG.VGG16(enable_lat=False,epsilon=0, pro_num=1)
    model.cuda()
    model.load_state_dict(torch.load((args.modelpath)))
    # if cifar then normalize epsilon from [0,255] to [0,1]
    if(args.model=='inception'):
        if torch.cuda.device_count() > 1:
            print(torch.cuda.device_count()," GPUs")
            totdev=torch.cuda.device_count()
            model = nn.DataParallel(model)
        else:
            totdev=1
            print("1 GPU")
    if args.dataset == 'mnist':
        eps = args.attack_epsilon
        alpha = args.attack_alpha
    else:
        eps = args.attack_epsilon / 255.0
        alpha = args.attack_alpha / 255.0
    
    #eps = args.attack_epsilon

    attack = Attack(dataroot = args.dataroot,
                    dataset  = args.dataset,
                    batch_size = args.attack_batchsize,
                    target_model = model,
                    criterion = nn.CrossEntropyLoss()
                    )
    
    if os.path.exists(args.savepath) == False:
        os.makedirs(args.savepath)
    if args.attack == 'fgsm':
        test_data_cln, test_data_adv, test_label, test_label_adv = attack.fgsm(eps)
    elif args.attack == 'ifgsm':
        test_data_cln, test_data_adv, test_label, test_label_adv = attack.i_fgsm(eps,alpha,args.attack_iter)
    elif args.attack == 'stepll':
        test_data_cln, test_data_adv, test_label, test_label_adv = attack.step_ll(eps)
    elif args.attack == 'pgd':
        test_data_cln, test_data_adv, test_label, test_label_adv = attack.PGD(eps,alpha,args.attack_iter)
    elif args.attack == 'momentum_ifgsm':
        test_data_cln, test_data_adv, test_label, test_label_adv = attack.momentum_ifgsm(eps,alpha,args.attack_iter,args.attack_momentum)
    print(test_data_adv.size(),test_label.size(), type(test_data_adv))
    
    if args.generate:
        save_data_label(args.savepath, test_data_cln, test_data_adv,test_label, test_label_adv, eps)

if __name__ == "__main__":
    main()