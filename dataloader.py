from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import os,json
import numpy as np
import cv2
from scipy.ndimage.morphology import grey_dilation
from scipy.interpolate import CubicSpline
#from utils.util import *
from skimage import filters
from utils.image import *

warnings.filterwarnings("ignore")

class TUSimple(Dataset):
    def __init__(self, path, transform = None):
        self.path = path
        self.LINE_SIZE = 15
        self.PAD = 120
        self.transform = transform
        sub    = [i for i in os.listdir(self.path) if i!=".DS_Store"]
        labels = [self.path + "/" + i for i in sub if i[-4:]=="json"]
        images_root_path = self.path + "/clips"
        images = list()
        self.labels = dict()
        images_folders = [self.path+"/clips/"+i for i in os.listdir(images_root_path) if i!=".DS_Store"]
        for imgs_folder in images_folders:
            for i in os.listdir(imgs_folder):
                if("DS" in i):
                    continue

                tmp_path = imgs_folder + "/" +i
                lst_of_imgs = [imgs_folder + "/" + i+"/"+j for j in os.listdir(tmp_path) if j=="20.jpg"]
                images += lst_of_imgs

        self.images = images
        for label_path in labels:
            with open(label_path,"r") as f:
                for i in f.readlines():
                    todict = json.loads(i[:-1])
                    label_img_name = todict['raw_file']
                    self.labels[label_img_name] = todict

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        key_ind = image_path.split("/").index("clips")
        key_path = os.path.join( *image_path.split("/")[key_ind:])
        abs_path = self.path +"/"+os.path.join( *image_path.split("/")[key_ind:])

        label = self.labels[key_path]
        lanes_w = np.array(label['lanes'])
        lanes_h = np.array(label['h_samples'])
        lane_cnt = lanes_w.shape[0]

        image = plt.imread(image_path) #(720, 1280, 3)
        image=np.pad(image, ((self.PAD,self.PAD), (0,0), (0, 0)), 'constant')
        image = cv2.resize(image, dsize=(640,480), interpolation=cv2.INTER_AREA)
        hmap = np.zeros(image.shape[:2])

        lane_pair = list()
        point = 0
        for i in range(lane_cnt):
            mask = (lanes_w[i,:] * lanes_h) > 0
            xs = (lanes_w[i,:][mask]) /1280. * 640.
            ys = (lanes_h[mask]+self.PAD) /960. * 480.
            ys = np.clip(ys, 0, 639)
            for j in range(xs.shape[0]):
                try:
                    hmap[int(ys[j]), int(xs[j])] = 1
                except:
                    print(ys)
                    print(xs)
                if(j<xs.shape[0]-1):
                    cv2.line(hmap, (int(xs[j]), int(ys[j])), (int(xs[j+1]), int(ys[j+1])), (i+1, i+1, i+1),  self.LINE_SIZE//2)
                #hmap = draw_umich_gaussian(hmap, [int(xs[j]), int(ys[j])], 10)
                point+=1

        instance = hmap

        if self.transform:
            augmented = self.transform(image=image, masks=[instance])

            image = augmented['image']
            instance = augmented['masks'][0]


        binary = np.where(instance>0, 1, 0)

        show = False
        if show:
            plt.imshow(image)
            plt.imshow(hmap, alpha=0.5)
            """
            plt.subplot(4,1,1)
            plt.imshow(image)
            plt.subplot(4,1,2)
            plt.imshow(image)
            plt.imshow(hmap, alpha=0.5)
            plt.subplot(4,1,3)
            plt.imshow(instance)
            plt.subplot(4,1,4)
            plt.imshow(binary)
            """
            plt.show()

        return image, binary, instance

class KcityLane(Dataset):
    def __init__(self, image_path,label_path, transform = None):
        self.transform = transform
        self.image_root_path = image_path
        self.label_root_path = label_path

        self.images_path = [self.image_root_path+"/"+i for i in os.listdir(self.image_root_path)]
        self.labels_path = [self.label_root_path+"/"+i for i in os.listdir(self.label_root_path) if i!=".DS_Store"]

        self.classMap = {'lane' : 1, 'bus':2, 'center':3}

        self.LINE_SIZE = 10

    def __len__(self):
        return len(self.labels_path)

    def __getitem__(self, idx):
        label_path = self.labels_path[idx]
        key_ind = label_path.split('/')[-1][:-5]
        #print(key_ind)
        #if len(key_ind)==10 and key_ind[-1]=='0':
        #    key_ind = key_ind[:-1]
        image_path = self.image_root_path +"/"+ key_ind + ".jpg"

        with open(label_path) as f:
            polys = json.load(f)['shapes']

        image = plt.imread(image_path) #(720, 1280, 3)
        image = np.pad(image, ((8,8), (0,0), (0, 0)), 'constant')
        #image = cv2.resize(image, dsize=(640,368), interpolation=cv2.INTER_AREA)
        mask_center  = np.zeros(image.shape[:2])
        mask_bus  = np.zeros(image.shape[:2])
        mask_lane = np.zeros(image.shape[:2])

        centers = dict()
        buses = dict()
        count = 1
        for poly in polys:
            try:
                cls = self.classMap[poly['label']]
            except:
                continue
            pts = np.array(poly['points'])
            cnt = pts[0]
            group = poly['group_id']
            if cls==3:
                try:
                    centers[group].append(cnt)
                except:
                    centers[group] = list()
                    centers[group].append(cnt)
            elif cls==2:
                try:
                    buses[group].append(cnt)
                except:
                    buses[group] = list()
                    buses[group].append(cnt)
            elif cls ==1 :
                cv2.fillPoly(mask_lane, np.int32([pts])+8 , count)
                count+=1

        for i in centers:
            try:
                cnt0, cnt1 = centers[i][0]
                cnt0 += 4
                cnt1 += 4
                cv2.line(mask_center, (int(cnt0[0]), int(cnt0[1])), (int(cnt1[0]), int(cnt1[1])), (i+1, i+1, i+1),  self.LINE_SIZE//2)
            except:
                pass

        for i in mask_bus:
            try:
                cnt0, cnt1 = centers[i]
                cnt0 += 4
                cnt1 += 4
                cv2.line(mask_bus, (int(cnt0[0]), int(cnt0[1])), (int(cnt1[0]), int(cnt1[1])), (i+1, i+1, i+1),  self.LINE_SIZE//2)
            except:
                pass

        #mask_binary = np.where(mask_entry_instance>0, 1, 0)
        if self.transform:
            augmented = self.transform(image=image, masks=[mask_lane,mask_center,mask_bus])
            image = augmented['image']
            lane = augmented['masks'][0]
            center = augmented['masks'][1]
            mask = augmented['masks'][2]
        #print(image.shape, mask.shape)

        show = False
        if show:
            plt.figure(figsize=(15,6))
            plt.subplot(3,3,1)
            plt.imshow(image)
            plt.imshow(np.where(lane>0, 1, 0), alpha=0.5)
            plt.subplot(3,3,2)
            plt.imshow(image)
            plt.imshow(lane, alpha=0.5)
            plt.subplot(3,3,3)
            plt.imshow(image)
            plt.imshow(center, alpha=0.5)
            plt.show()
            #print(torch.unique(instance))

        #return image, binary, instance
        return image, np.where(lane>0, 1, 0), lane

class KcityLane2(Dataset):
    def __init__(self, image_path,label_path, transform = None):
        self.transform = transform
        self.image_root_path = image_path
        self.label_root_path = label_path

        self.images_path = [self.image_root_path+"/"+i for i in os.listdir(self.image_root_path)]
        self.labels_path = [self.label_root_path+"/"+i for i in os.listdir(self.label_root_path) if i!=".DS_Store"]

        self.classMap = {'lane' : 1, 'lane0':2, 'center':3}

        self.LINE_SIZE = 15

    def __len__(self):
        return len(self.labels_path)

    def __getitem__(self, idx):
        label_path = self.labels_path[idx]
        key_ind = label_path.split('/')[-1][:-5]
        #print(key_ind)
        #if len(key_ind)==10 and key_ind[-1]=='0':
        #    key_ind = key_ind[:-1]
        image_path = self.image_root_path +"/"+ key_ind + ".jpg"

        with open(label_path) as f:
            polys = json.load(f)['shapes']

        image = plt.imread(image_path) #(480, 640, 3)
        #image = np.pad(image, ((8,8), (0,0), (0, 0)), 'constant')
        #image = cv2.resize(image, dsize=(640,368), interpolation=cv2.INTER_AREA)
        mask_center  = np.zeros(image.shape[:2])
        mask_bus  = np.zeros(image.shape[:2])
        mask_lane = np.zeros(image.shape[:2])

        centers = dict()
        lanes = dict()
        count = 1
        for poly in polys:
            pts = np.array(poly['points'])
            cnt0, cnt1 = pts[:,0],pts[:,1]
            #cnt0 += 4
            #cnt1 += 4
            ptnCnt = cnt0.shape[0]
            for ind in range(ptnCnt-1):
                cv2.line(mask_lane, (int(cnt0[ind]), int(cnt1[ind])), (int(cnt0[ind+1]), int(cnt1[ind+1])), (count+1, count+1, count+1),  self.LINE_SIZE//2)
            count+=1

        #mask_binary = np.where(mask_entry_instance>0, 1, 0)
        if self.transform:
            augmented = self.transform(image=image, masks=[mask_lane])
            image = augmented['image']
            lane = augmented['masks'][0]
            #center = augmented['masks'][1]
            #mask = augmented['masks'][2]
        #print(image.shape, mask.shape)

        show = False
        if show:
            plt.figure(figsize=(15,6))
            plt.imshow(image)
            plt.imshow(lane, alpha=0.5)
            plt.show()
            #print(torch.unique(instance))

        #return image, binary, instance
        return image, np.where(lane>0, 1, 0), lane

class KcityPark(Dataset):
    def __init__(self, image_path,label_path, transform = None):
        self.transform = transform
        self.image_root_path = image_path
        self.label_root_path = label_path

        self.images_path = [self.image_root_path+"/"+i for i in os.listdir(self.image_root_path)]
        self.labels_path = [self.label_root_path+"/"+i for i in os.listdir(self.label_root_path) if i!=".DS_Store"]

        self.classMap = {'binary' : 1, 'entry':2, 'area':3}

        self.LINE_SIZE = 10

    def __len__(self):
        return len(self.labels_path)

    def __getitem__(self, idx):
        label_path = self.labels_path[idx]
        key_ind = label_path.split('/')[-1][:-5]
        #if len(key_ind)==10 and key_ind[-1]=='0':
        #    key_ind = key_ind[:-1]
        image_path = self.image_root_path +"/"+ key_ind + ".jpg"

        with open(label_path) as f:
            polys = json.load(f)['shapes']

        image = plt.imread(image_path) #(720, 1280, 3)
        image = np.pad(image, ((8,8), (0,0), (0, 0)), 'constant')
        #image = cv2.resize(image, dsize=(640,368), interpolation=cv2.INTER_AREA)
        mask_binary  = np.zeros(image.shape[:2])
        mask_area_instance = np.zeros(image.shape[:2])
        mask_entry = np.zeros(image.shape[:2])
        mask_entry_instance = np.zeros(image.shape[:2])

        groups = dict()
        for poly in polys:
            cls = self.classMap[poly['label']]
            if cls==2:
                pts = np.array(poly['points'])
                cnt = pts[0]
                group = poly['group_id']
                try:
                    groups[group].append(cnt)
                except:
                    groups[group] = list()
                    groups[group].append(cnt)

        group = 1
        for poly in polys:
            cls = self.classMap[poly['label']]
            pts = np.array(poly['points'])
            if cls ==1 :
                cv2.fillPoly(mask_binary, np.int32([pts])+8 , 1)
            elif cls ==3:
                cv2.fillPoly(mask_area_instance, np.int32([pts])+8, group)
                group += 1

        for i in groups:
            try:
                cnt0, cnt1 = groups[i]
                cnt0 += 8
                cnt1 += 8
                cv2.line(mask_entry_instance, (int(cnt0[0]), int(cnt0[1])), (int(cnt1[0]), int(cnt1[1])), (i+1, i+1, i+1),  self.LINE_SIZE//2)
            except:
                pass

        mask_binary = np.where(mask_entry_instance>0, 1, 0)
        if self.transform:
            augmented = self.transform(image=image, masks=[mask_binary,mask_entry_instance,mask_area_instance])
            image = augmented['image']
            binary = augmented['masks'][0]
            instance = augmented['masks'][1]
            area_instance = augmented['masks'][2]
        #print(image.shape, mask.shape)

        show = True
        if show:
            plt.subplot(2,2,1)
            plt.imshow(image)
            plt.imshow(np.where(area_instance>0, 1, 0), alpha=0.5)
            plt.subplot(2,2,2)
            plt.imshow(image)
            plt.imshow(area_instance, alpha=0.5)
            plt.show()
            #print(torch.unique(instance))

        #return image, binary, instance
        return image, np.where(area_instance>0, 1, 0), area_instance

class Kookmin(Dataset):
    def __init__(self, path="/home/yo0n/바탕화면/kookmin_Lane", transform = None):
        self.path = path
        self.transform = transform
        self.image_root_path = self.path + "/images"
        self.label_root_path = self.path + "/labels"

        self.images_path = [self.image_root_path+"/"+i for i in os.listdir(self.image_root_path)]
        self.labels_path = [self.label_root_path+"/"+i for i in os.listdir(self.label_root_path) if i!=".DS_Store"]

        self.classMap = {'lane' : 1, 'stop':2}

    def __len__(self):
        return len(self.labels_path)

    def __getitem__(self, idx):
        label_path = self.labels_path[idx]
        key_ind = label_path.split('/')[-1][:-5]
        #if len(key_ind)==10 and key_ind[-1]=='0':
        #    key_ind = key_ind[:-1]
        image_path = self.path+"/images/" + key_ind + ".jpg"

        with open(label_path) as f:
            polys = json.load(f)['shapes']

        image = plt.imread(image_path) #(720, 1280, 3)
        #image = np.pad(image, ((8,8), (0,0), (0, 0)), 'constant')
        #image = cv2.resize(image, dsize=(640,368), interpolation=cv2.INTER_AREA)
        mask  = np.zeros(image.shape[:2])

        for poly in polys:
            cls = self.classMap[poly['label']]
            pts = np.array(poly['points'])
            cv2.fillPoly(mask, np.int32([pts]) , 1)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        #print(image.shape, mask.shape)

        mask2 = np.zeros((2, mask.shape[1], mask.shape[2]))
        mask2[1,:,:] = mask
        mask2[0,:,:][mask2[1,:,:]!=1] = 1


        show = False
        if show:
            plt.subplot(2,2,1)
            plt.imshow(image)
            plt.subplot(2,2,2)
            plt.imshow(mask)
            plt.show()

        return image, mask2

class LaneMerge(Dataset):
    def __init__(self, aug):
        self.dataset0 = KcityLane2(image_path = '/home/yo0n/workspace2/KCITY/kcity/images',
                            label_path = '/home/yo0n/workspace2/KCITY/kcity/label_lane',
                            transform = aug)
        self.dataset1 = TUSimple(path='/home/yo0n/바탕화면/TUsimple' ,transform=aug)

    def __len__(self):
        return len(self.dataset0) + len(self.dataset1)

    def __getitem__(self, idx):
        if(idx < len(self.dataset0)):
            return self.dataset0[idx]
        elif(idx >= len(self.dataset0)):
            return self.dataset1[idx-len(self.dataset0)]
        else:
            print("idx error : ",idx)

def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)

        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)

        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)
    plt.show()

if __name__=="__main__":
    import random
    from albumentations import *

    aug = Compose([
                 #HorizontalFlip(),
                 OneOf([
                    MotionBlur(p=0.2),
                    MedianBlur(blur_limit=3, p=0.1),
                    ISONoise(p=0.3),
                    Blur(blur_limit=3, p=0.1),], p=0.35),
                 RandomBrightnessContrast(brightness_limit=(-0.3,0.4),p=0.5),
                 ShiftScaleRotate(shift_limit=0.0125, scale_limit=0.1, rotate_limit=15,border_mode=cv2.BORDER_CONSTANT, p=0.2),
                 OneOf([HueSaturationValue(hue_shift_limit=20, sat_shift_limit=0.1, val_shift_limit=0.1, p=0.3),
                        ToSepia(p=0.08),], p=0.3),
                 OneOf([RandomShadow(p=0.2),
                        RandomRain(blur_value=2,p=0.4),
                        RandomFog(fog_coef_lower=0.1,fog_coef_upper=0.2, alpha_coef=0.25, p=0.2),], p=0.25),
                 Cutout(num_holes=5, max_h_size=40, max_w_size=40,p=0.1),
                 #Resize(368,640,p=1)
                ], p=1)


    random.seed(a=None)
    """
    dataset = TUSimple("/home/yo0n/바탕화면/TUsimple", transform = aug)
    o = dataset[random.randint(0,len(dataset)-1)]
    print(o[0].shape, o[1].shape, o[2].shape) # (368, 640) (368, 640)

    lanes = int(o[2].max())
    plt.subplot(2,1,1)
    plt.imshow(o[0] + np.array([0.485, 0.456, 0.406]))
    plt.subplot(2,1,2)
    plt.imshow(o[2])
    plt.show()
    """
    """
    #aug = RandomBrightnessContrast(brightness_limit=(-0.3,0.0),p=1)
    aug = Normalize(std=(0., 0., 0),p=1)
    augmented = aug(image=o[0], mask=o[2])

    img_aug = augmented['image']
    img_aug = o[0]/255. - (0.485, 0.456, 0.406)
    mask_aug = augmented['mask']

    visualize(img_aug, mask_aug, original_image=o[0], original_mask=o[2])
    """

    """
    dataset = KcityLane2(image_path = '/home/yo0n/workspace2/KCITY/kcity/images',
                        label_path = '/home/yo0n/workspace2/KCITY/kcity/label_lane',
                        transform = aug)
    """
    dataset = LaneMerge(aug=aug)
    #o = dataset[1200]

    o = dataset[random.randint(0,len(dataset)-1)]
