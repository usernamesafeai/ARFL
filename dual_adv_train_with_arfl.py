'''
@Author: ***** 
@Date: created on Dec 16, 2021, updated on Nov 6, 2022 and on Nov 16, 2023 
Some of the code are adapted from https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/demos/Adversairal%20Training%20(MNIST).ipynb
Input: learning rate lr, epsilon ep, fold number fn, mixed or adv or std type, percentage of mixed pt,
Output: saved model 
'''

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchattacks
from torchattacks import PGD, FGSM
from ImageDataset import Dataset
from PIL import ImageFile
from PIL import Image
import random
from utils import calculate_robustness
import argparse
import time

import torchvision.transforms as tf

pretransform = tf.Compose([
            tf.RandomVerticalFlip(),
            tf.RandomAffine(45, translate=(0,0), scale=None, shear=45, resample=Image.BILINEAR, fillcolor=0),
            ])

ImageFile.LOAD_TRUNCATED_IMAGES = True
print("PyTorch", torch.__version__)
print("Torchvision", torchvision.__version__)
print("Torchattacks", torchattacks.__version__)
print("Numpy", np.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_img(images, fname, flag):
    A = images.cpu().detach().numpy()
    A = A[0,0,:]
    A = A * 255
    im = Image.fromarray(A)
    im = im.convert('L')
    im.save('data/atk/' + fname[0][:-4] + flag + '.jpeg')

def train_adv_net(num_epochs, num_train, percentage, batch_size, epsilon, lr, foldnumber, gamma, rfl, train_loader, val_loader, model):
    model.train()
    max_acc = 0.0
    min_loss = 1e8 
    best_model = model
    atk = PGD(model, eps=epsilon, alpha=0.1, steps=7)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):

        current_loss = 0.0
        total_batch = num_train // batch_size
        correct = 0
        total = 0

        for i, (batch_images, batch_labels, fnames) in enumerate(train_loader):
            batch_labels = torch.tensor(batch_labels).to(device)
            if percentage > 0:
                if random.random() < percentage: 
                    X = batch_images.cuda()
                else:
                    X = atk(batch_images, batch_labels).cuda()
            else:
                X = atk(batch_images, batch_labels).cuda()
            Y = batch_labels.cuda()

            pred = model(X)

            if rfl:
                #Robust learning
                representation = model.avgpool(X)
                robustness = calculate_robustness(Y, representation)
                cost = loss(pre, Y) - gamma * robustness
                #end robust learning
            else:
                cost = loss(pre, Y)

            current_loss += cost.item()
            _, predicted = torch.max(pred.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels.cuda()).sum()

            cost.backward()
            optimizer.step()

            if (i+1) % 200 == 0:
                print(str(pred) + '-' + str(batch_labels) + '-' + str(cost.item()))
                print('Epoch [%d/%d], lter [%d/%d], Loss: %.4f'
                     %(epoch+1, num_epochs, i+1, total_batch, cost.item()))

        epoch_acc = float(correct) / total
        print('Loss: {:.4f} Acc: {:.4f}'.format(current_loss, epoch_acc))
        correct = 0
        total = 0

        current_loss = 0.0
        for images, labels, fnames in val_loader:
            labels = torch.tensor(labels).to(device)
            if random.random() < percentage: 
                images = images.cuda()
            else:
                images = atk(images, labels).cuda()
            outputs = model(images)


            cost = loss(outputs, labels.cuda())
            current_loss += cost.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.cuda()).sum()
        acc = float(correct) / total
        if current_loss < min_loss:
            min_loss = current_loss 
            best_model = model
            print('better model is found with loss = ' + str(min_loss))

        print('Validation accuracy: %.2f %%' % (100 * float(correct) / total))
        modelsettings = '_E' + str(epoch) + '_lr' + str(lr) + '_percentage' + str(percentage) + '_epsilon' + str(epsilon) + '_batchsize' + str(batch_size) + '_foldnumber' + str(foldnumber) + '_rfl' + str(rfl) + '_gamma' + str(gamma)
        torch.save(model, './checkpoints/vgg16adv_layer26' + modelsettings + '.pth')

    model = best_model
    modelsettings = '_lr' + str(lr) + '_percentage' + str(percentage) + '_epsilon' + str(epsilon) + '_batchsize' + str(batch_size) + '_foldnumber' + str(foldnumber) + '_rfl' + str(rfl) + '_gamma' + str(gamma)
    torch.save(model, './checkpoints/vgg16adv_layer26_' + modelsettings + '.pth')

    print('---------------Finished Adv training---------------------')

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--fn', type=int, default=1,
                        help='fold number')
    parser.add_argument('--batchsize', type=int, default=2, metavar='N',
                        help='input batch size for training [default: 2]')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train [default: 100]')
    parser.add_argument('--epsilon', type=float, default=0.01, metavar='N',
                        help='scale of purturbation [default: 0.01]')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate [default: 1e-4]')
    parser.add_argument('--percentage', type=float, default=0.5,
                        help='percentage of standard data [default: 0.5]')
    parser.add_argument('--rfl', type=bool, default=False,
                        help='Whether to use robust feature learning [default: False]')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='weight of robustness term [default: 0.1]')
 

    args = parser.parse_args()
    foldnumber = args.fn
    batch_size = args.batchsize
    num_epochs = args.epochs
    epsilon = args.epsilon
    lr = args.lr
    percentage = args.percentage
    rfl = args.rfl
    gamma = args.gamma
    print('Fold number = ' + str(foldnumber)) 
    print('epsilon = ' +  str(epsilon))
    print('percentage = ' + str(percentage))
    print('epochs = ' + str(num_epochs) )
    print('learning rate = ' + str(lr))
    print('batch_size = ' + str(batch_size))
    print('use robust feature learning = ' + str(rfl))
    print('gamma = ' + str(gamma))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.vgg16(pretrained=True)
    num_classes = 2
    model.classifier[6] = nn.Linear(4096,num_classes)
    print(model)
    model = model.to(device)

    count_param = 0
    for param in model.parameters():
        if count_param < 26 :
            param.requires_grad = False
        else:
            param.requires_grad = True 
        count_param += 1

    print('---------------Finished loading normal model---------------------')

    labelpath = 'data/CMMD/pidside2label.json'
    labelpath_generated = 'data/CMMD/pidside2label_generated.json'
    datapath = 'data/CMMD_trainvaltest_allv1'

    train_datapath = datapath + '/train' + str(foldnumber) + '/'
    test_datapath = datapath + '/test' + str(foldnumber) + '/'
    test_datapath_generated = datapath + '/test_generated/'
    val_datapath = datapath + '/val' + str(foldnumber) + '/'

    train_set = Dataset(train_datapath, labelpath, transform = pretransform)
    test_set = Dataset(test_datapath, labelpath)
    test_set_generated = Dataset(test_datapath_generated, labelpath_generated)
    val_set = Dataset(val_datapath, labelpath)

    train_loader  = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                             batch_size=batch_size,
                                             shuffle=False)

    test_loader_generated = torch.utils.data.DataLoader(dataset=test_set_generated,
                                             batch_size=batch_size,
                                             shuffle=False)

    val_loader = torch.utils.data.DataLoader(dataset=val_set,
                                             batch_size=batch_size,
                                             shuffle=False)

    num_train = len(train_set)
    train_adv_net(num_epochs, num_train, percentage, batch_size, epsilon, lr, foldnumber, gamma, rfl, train_loader, val_loader, model)

    end_time = time.time()
    time_diff = round(end_time - start_time)
    minutes = time_diff // 60
    hours = minutes // 60
    minutes = minutes % 60
    seconds = time_diff % 60
    print('total training time + testing time -- hours:minutes:seconds = ' + str(hours) + ':' + str(minutes) + ':' + str(seconds))
if __name__=='__main__':
    main()
