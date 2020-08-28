import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchcontrib.optim import SWA

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2

from dataloader import KcityLane
from albumentations import *
from models.lanenet import LaneNet
from loss.loss import *
from loss.loss2 import *
from loss.lovasz import *

#basic setting
num_epochs = 50
batch_size = 7
vis_result= True
validation_ratio = 0
startlr = 1e-4

#model load
model = LaneNet().cuda()
model = torch.load("./ckpt/recent.pth")
model_name = model.__class__.__name__

#dataset load
aug = Compose([
             #HorizontalFlip(),
             RandomBrightnessContrast(brightness_limit=(-0.3,0.4),p=0.5),
             Cutout(num_holes=5, max_h_size=40, max_w_size=40,p=0.1),
             Resize(368,640,p=1)
           ], p=1)


#train_dataset = TUSimple("/home/yo0n/바탕화면/TUsimple", transform= aug)
train_dataset = dataset = KcityLane(image_path = '/home/yo0n/workspace2/KCITY/kcity/images',
                    label_path = '/home/yo0n/workspace2/KCITY/kcity/lane_test',
                    transform = aug)
dataiter,dataset_len = len(train_dataset)//batch_size,len(train_dataset)
train_len = int(dataset_len*(1-validation_ratio))
train_dataset, validate_dataset = torch.utils.data.random_split(train_dataset, [train_len, dataset_len-train_len])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=0)
#valid_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=1,shuffle=True, num_workers=0)
print("trainset lenght :: ",len(train_dataset))
print_iter = 10

#loss
binary_criterion  = FocalLoss(gamma=2.1)
#binary_criterion  = LossBinary()
cluster_criterion = Clustering()

#optim setting
#optimizer = optim.RMSprop(model.parameters(), lr=startlr, weight_decay=5e-4, momentum=0.9)
optimizer = optim.AdamW(model.parameters(), lr=startlr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4, amsgrad=False)
scheduler = optim.lr_scheduler.CyclicLR(optimizer,base_lr=startlr, max_lr=startlr*3, step_size_up=2000, mode='triangular2' , gamma=0.9994,cycle_momentum=False )
#opt = SWA(optimizer, swa_start=10, swa_freq=5, swa_lr=0.05)

global_iter = 0
global_losses = list()
for epoch in range(num_epochs):
    losses = list()
    binary_losses  = list()
    cluster_losses = list()
    #print("optim lr : ",optimizer.param_groups[0]['lr'])
    for iteration,sample in enumerate(train_loader):
        global_iter+=1

        img, binary_label , instance_label = sample #torch.Size([4, 368, 640, 3]) torch.Size([4, 368, 640]) torch.Size([4, 368, 640])
        img = img.permute(0,3,1,2).float()/255.
        binary_label = binary_label.view(-1,1,368,640).float()
        instance_label = instance_label.view(-1,1,368,640).float()
        img,binary_label,instance_label = img.cuda(),binary_label.cuda(),instance_label.cuda()

        binary_pred, instance_pred = model(img)

        #binary_loss     = lovasz_hinge(binary_pred, binary_label.squeeze() )
        binary_loss     = binary_criterion(binary_pred, binary_label )
        clustering_loss = cluster_criterion(instance_pred, binary_label, instance_label)

        loss =   binary_loss + 10*clustering_loss
        losses.append(loss.item())
        global_losses.append(loss.item())
        binary_losses.append(binary_loss.item())
        cluster_losses.append(clustering_loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(iteration%print_iter == 0 and False):
            print(str(epoch)," :: ",str(iteration), "/",dataiter,"\n  loss     :: ",loss.item())
            print("  avg loss :: ",sum(losses)/len(losses))
            print("  + avg binary loss   :: ",sum(binary_losses)/len(binary_losses))
            print("  + avg instance loss :: ",sum(cluster_losses)/len(cluster_losses))

            binary_pred = np.argmax(binary_pred[0].detach().cpu().numpy(), axis=0)
            #binary_pred = binary_pred[0].detach().cpu().squeeze().numpy()
            instance_pred = instance_pred[0].detach().cpu().numpy()
            img_0 = img[0].permute(1,2,0).squeeze().cpu().numpy()
            instance_label0 = instance_label[0].permute(1,2,0).squeeze().cpu().numpy()

            plt.subplot(3,1,1)
            plt.imshow(img_0)
            plt.imshow(instance_label0,alpha=0.3)
            plt.subplot(3,1,2)
            plt.imshow(binary_pred)
            plt.subplot(3,1,3)
            plt.imshow(instance_pred[0])
            plt.show(block=False)
            plt.pause(3)
            plt.close()

    scheduler.step()
    if(epoch %5 ==0):
        #torch.save(model,"./ckpt/"+str(epoch)+".pth")
        torch.save(model,"./ckpt/recent.pth")
        print(" ==> GLOBAL  avg loss :: ",sum(global_losses)/len(global_losses))
