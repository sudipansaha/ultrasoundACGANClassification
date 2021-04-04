"""""""""
ACGAN code adapted for breast ultrasound classification

"""""""""
from __future__ import print_function
import argparse
import os
import random
import numpy as np
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import folderPersonalizedForAcgan
import scipy.io as sio
import numpy as np
from sklearn.metrics import roc_auc_score
import cv2


import model



#command to run: %python synthImageSaver.py --dataset folder --dataroot ./dataset/set1Training/ --outf ./checkpoints/set1/ --niter 200 --cuda


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | mnist | folder')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=40, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./checkpoints/', help='folder to output images and model checkpoints')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass


##Defining manual seeds
manualSeed=40
torch.manual_seed(manualSeed) #set seed (manually) for generate random numbers with CPU
random.seed(manualSeed)
np.random.seed(manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


if opt.dataset in ['folder']:
    # folder dataset
    dataset = folderPersonalizedForAcgan.ImageFolder(root=opt.dataroot,
                                   transform=transforms.Compose([
                                   transforms.Resize((opt.imageSize,opt.imageSize)), ##since image original sizes are not square, take (size,size)
                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomVerticalFlip(),
                                   transforms.RandomRotation(7),
                                   transforms.RandomRotation(15),
                                   #transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0, hue=0),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    
nc = 3   ##number of channels of the image
nb_label = 2 ##number of different classes in the dataset



assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)

    

netG = model.netG(nz, ngf, nc)

if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = model.netD(ndf, nc, nb_label)

if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

s_criterion = nn.BCELoss()
c_criterion = nn.NLLLoss()

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
s_label = torch.FloatTensor(opt.batchSize)
c_label = torch.LongTensor(opt.batchSize)

real_label = 1
fake_label = 0

if opt.cuda:
    netD.cuda()
    netG.cuda()
    s_criterion.cuda()
    c_criterion.cuda()
    input, s_label = input.cuda(), s_label.cuda()
    c_label = c_label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

input = Variable(input)
s_label = Variable(s_label)
c_label = Variable(c_label)
noise = Variable(noise)
fixed_noise = Variable(fixed_noise)
fixed_noise_ = np.random.normal(0, 1, (opt.batchSize, nz))
random_label = np.random.randint(0, nb_label, opt.batchSize)
print('fixed label:{}'.format(random_label))
random_onehot = np.zeros((opt.batchSize, nb_label))
random_onehot[np.arange(opt.batchSize), random_label] = 1
fixed_noise_[np.arange(opt.batchSize), :nb_label] = random_onehot[np.arange(opt.batchSize)]


fixed_noise_ = (torch.from_numpy(fixed_noise_))
fixed_noise_ = fixed_noise_.resize_(opt.batchSize, nz, 1, 1)
fixed_noise.copy_(fixed_noise_)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

def test(predict, labels):
    correct = 0
    pred = predict.max(1)[1]
    correct = pred.eq(labels).cpu().sum()
    return correct, len(labels)

for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        ###########################
        # (1) Update D network
        ###########################
        # train with real
        netD.zero_grad()
        img, label = data
        batch_size = img.size(0)
        input.resize_(img.size()).copy_(img)
        s_label.resize_(batch_size).fill_(real_label)
        c_label.resize_(batch_size).copy_(label)
        s_output, c_output = netD(input)
        source_errD_real = s_criterion(s_output, s_label)
        class_errD_real = c_criterion(c_output, c_label)
        errD_real = source_errD_real + class_errD_real
        errD_real.backward()
        D_x = s_output.mean()
        
        correct, length = test(c_output, c_label)

        # train with fake
        noise.resize_(batch_size, nz, 1, 1)
        noise.normal_(0, 1)

        label = np.random.randint(0, nb_label, batch_size)
        noise_ = np.random.normal(0, 1, (batch_size, nz))
        label_onehot = np.zeros((batch_size, nb_label))
        label_onehot[np.arange(batch_size), label] = 1
        noise_[np.arange(batch_size), :nb_label] = label_onehot[np.arange(batch_size)]
        
        noise_ = (torch.from_numpy(noise_))
        noise_ = noise_.resize_(batch_size, nz, 1, 1)
        noise.copy_(noise_)

        c_label.resize_(batch_size).copy_(torch.from_numpy(label))

        fake = netG(noise)
        
        if epoch==(opt.niter-1):
            for synthImageSaverIter in range(opt.batchSize):
                fakeDetached = fake.detach().cpu().numpy()
                fakeImageCurrentIter = np.squeeze(fakeDetached[synthImageSaverIter,0,:,:])
                fakeImageCurrentIter = (fakeImageCurrentIter-np.amin(fakeImageCurrentIter))/(np.amax(fakeImageCurrentIter)-np.amin(fakeImageCurrentIter))
                if c_label[synthImageSaverIter]==1:
                    cv2.imwrite('output/syntheticImages/synthImageMal'+str(synthImageSaverIter)+'.png',fakeImageCurrentIter*255)
                if c_label[synthImageSaverIter]==0:
                    cv2.imwrite('output/syntheticImages/synthImageBen'+str(synthImageSaverIter)+'.png',fakeImageCurrentIter*255)
        s_label.fill_(fake_label)
        s_output,c_output = netD(fake.detach())
        source_errD_fake = s_criterion(s_output, s_label)
        class_errD_fake = c_criterion(c_output, c_label)
        errD_fake = source_errD_fake + class_errD_fake

        ##The errD backpropgation has been changed from the original code. In first few epochs, only class_errD_fake is used to backpropagate
        #errD_fake.backward()
        if epoch < (opt.niter/10):
            class_errD_fake.backward()
        else:
            source_errD_fake.backward(retain_graph=True)
            class_errD_fake.backward()
        D_G_z1 = s_output.mean()
        errD = source_errD_real + source_errD_fake
        optimizerD.step()

        ###########################
        # (2) Update G network
        ###########################
        netG.zero_grad()
        s_label.fill_(real_label)  # fake labels are real for generator cost
        s_output,c_output = netD(fake)
        source_errG = s_criterion(s_output, s_label)
        class_errG = c_criterion(c_output, c_label)
        
        errG = source_errG + class_errG
        errG.backward()
        D_G_z2 = s_output.mean()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)) (before/after D update): %.4f / %.4f, Class Prediction Accuracy: %.4f / %.4f = %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD, errG, D_x, D_G_z1, D_G_z2,
                 correct, length, 100.* correct / length))
#        if i % 100 == 0:
#            vutils.save_image(img,
#                    '%s/real_samples.png' % opt.outf)
#            #fake = netG(fixed_cat)
#            fake = netG(fixed_noise)
#            vutils.save_image(fake,
#                    '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch))

    # do checkpointing
    if (epoch+1)%100==0:
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch+1))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch+1))
        
        
        
