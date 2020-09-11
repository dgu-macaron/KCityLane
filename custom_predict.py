import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchcontrib.optim import SWA

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.cluster import MeanShift,DBSCAN
import hdbscan
import cv2,io,random,os
import argparse
import csaps

from models.lanenet import LaneNet

#basic setting
batch_size = 1

#model load
model = torch.load("./ckpt/recent_line.pth").eval()
model_name = model.__class__.__name__

hdbscan_cluster = hdbscan.HDBSCAN(min_cluster_size=100,allow_single_cluster=False)
frame_array = list()

#img = plt.imread('/home/yo0n/workspace2/KCITY/kcity/images/1597891859696844752.jpg')
images_folders = ['/home/yo0n/workspace2/KCITY/kcity/images/'+i for i in os.listdir('/home/yo0n/workspace2/KCITY/kcity/images') if i!=".DS_Store"]
img = plt.imread(images_folders[random.randint(0,len(images_folders))])
#img = np.pad(img, ((8,8), (0,0), (0, 0)), 'constant')
#img = cv2.resize(img, dsize=(640,368), interpolation=cv2.INTER_AREA)
img = torch.tensor(img).unsqueeze(0)
img = img.permute(0,3,1,2).float()/255.
img,model = img.cuda(), model.cuda() #img.cuda(), model.cuda()

start =time.time()
binary_pred, instance_pred = model(img)
end = time.time()

binary_pred = np.argmax(F.softmax(binary_pred).squeeze().detach().cpu().numpy(), axis=0)
instance_final = np.zeros_like(binary_pred)
embedding_dim = int(instance_pred.shape[1])
instance_pred = instance_pred.squeeze().detach().cpu().numpy() #(embedding dim , 368, 640)
binary_mask = np.stack([binary_pred for i in range(instance_pred.shape[0])], axis=0)
mask = instance_pred * binary_mask #(2, 368, 640)
lanes = instance_pred[binary_mask > 0].reshape(embedding_dim,-1).transpose(1,0)

start_clustering = time.time()
#clustering = DBSCAN(eps=0.25, min_samples=8,n_jobs=-1).fit(lanes)
#cluster_labels = MeanShift(n_jobs=4, min_bin_freq=1000 ,bin_seeding=True, max_iter=10).fit(lanes).labels_
cluster_labels = hdbscan_cluster.fit_predict(lanes)
end_clustering = time.time()

instance_final[binary_pred > 0] = cluster_labels + 1
#(4, 368, 640) (4, 368, 640) (42646,) (21323,)

img = img.squeeze().permute(1,2,0).squeeze().cpu().numpy()

print("infer time : ",end-start)
print("clustering time : ",end_clustering-start_clustering)

img = img#+ np.array([0.485, 0.456, 0.406])
mask = np.stack([instance_final for i in range(3)], axis=-1)
lanes = np.unique(instance_final).shape[0]

for i in range(1,lanes):
    laneCoord = np.where(instance_final==i)
    print(i,len(laneCoord[0]), laneCoord[0].shape)

plt.title(str(lanes))
plt.imshow(img)
plt.imshow(instance_final, alpha=0.5)
plt.show()
