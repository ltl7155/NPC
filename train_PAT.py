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
from AlexNet_SVHN import AlexNet
import pickle
from collections import Counter
import time


def PGD_attack(model, epsilon, k, a, x_nat, y, loss_func):
    model.eval()
    x_rand = x_nat.detach()
    x_rand = x_rand + torch.zeros_like(x_rand).uniform_(-epsilon, epsilon)
    x_adv = Variable(x_rand.data, requires_grad=True).cuda()

    for j in range(0, k):
        h_adv = model(x_adv)
        loss = loss_func(h_adv, y)
        model.zero_grad()
        if (x_adv.grad is not None):
            x_adv.grad.data.fill_(0)
        loss.backward()

        x_adv = x_adv.detach() + a * torch.sign(x_adv.grad)
        x_adv = torch.where(x_adv > x_nat + epsilon, x_nat + epsilon, x_adv)
        x_adv = torch.clamp(x_adv, 0, 1)
        x_adv = torch.where(x_adv < x_nat - epsilon, x_nat - epsilon, x_adv)
        x_adv = torch.clamp(x_adv, 0, 1)
        x_adv = Variable(x_adv.data, requires_grad=True).cuda()

    model.train()
    return x_adv


def train(model):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])
    train_data = torchvision.datasets.SVHN(
                root = './data/SVHN',
                split = 'train',
                transform = transform_train,
                download = False
        )
    test_data = torchvision.datasets.SVHN(
            root = './data/SVHN',
            split = 'test',
            transform = transform_test,
            download = False
    )
    train_loader = Data.DataLoader(dataset=train_data, batch_size=args.batchsize, shuffle=True)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=args.batchsize, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_func = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epoch):
        for step, (x, y) in enumerate(train_loader):
            train_x = Variable(x, requires_grad=False).cuda()
            train_y = Variable(y, requires_grad=False).cuda()

            adv_x = PGD_attack(model=model, epsilon=8.0/255, k=10, a=1.0/255, x_nat=train_x, y=train_y, loss_func=loss_func)
            logits1 = model(adv_x)
            loss1 = loss_func(logits1, train_y)

            logits2 = model(train_x)
            loss2 = loss_func(logits2, train_y)

            '''
            if(epoch < 5):
                total_loss = loss2
            else:
                total_loss = 0.5 *(loss1 + loss2)
            '''
            
            total_loss = 0.5 *(loss1 + loss2)

#             print("total_loss:", total_loss)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            

        print('epoch ' + str(epoch))
        total = 0
        correct = 0
        model.eval()
        with torch.no_grad():
            for test_step, (val_x, val_y) in enumerate(test_loader):
                val_x = val_x.cuda()
                val_y = val_y.cuda()
                val_output = model(val_x)
                _, val_pred_y = val_output.max(1)
                correct += val_pred_y.eq(val_y).sum().item()
                total += val_y.size(0)
        result = float(correct) * 100.0 / float(total)
        print('val accuracy: %.2f%%' % result)
        model.train()
                

#             if (step + 1) % 10 == 0:
#                 model.eval()
#                 with torch.no_grad():
#                     train_output, _ = model(train_x)
#                     train_loss = loss_func(train_output, train_y)
#                     _, train_pred_y = train_output.max(1)
#                     Accuracy = float(train_pred_y.eq(train_y).sum().item()) / float(train_y.size(0)) * 100.0
#                     print('train loss: %.4f' % train_loss.item(), '| train accuracy: %.2f%%' % Accuracy)
#                 model.train()
                
        if (epoch+1)%5 == 0:
            save_model_path = args.save_model_path + "PAT_epoch" + str(epoch) + "_lr" + str(args.lr) + ".pkl"
            print('saving model...')
            torch.save(model.state_dict(), save_model_path)
            

if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser(description='model interpretation')
    parser.add_argument('--load_model_path', default="./trained_models/alexnet_lr0.0001_39.pkl",
                        help='load model path')
    parser.add_argument('--save_model_path', default="./trained_models/PAT/", 
                        help='save model path')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate')
    parser.add_argument('--batchsize', type=int, default=200, help='training batch size')
    parser.add_argument('--epoch', type=int, default=60, help='number of epochs to train for')
    parser.add_argument('--gpu', type=str, default="6", help='training batch size')
    args = parser.parse_args()
    # print(args)
   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    net = AlexNet(num_classes=10).cuda()
    # torch.save(net.state_dict(), args.load_model_path)

    # load model
    if os.path.exists(args.load_model_path):
        net.load_state_dict(torch.load(args.load_model_path))
        print('load model.')
    else:
        print("load failed.")

    train(net)