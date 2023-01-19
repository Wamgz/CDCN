'''
Code of 'Searching Central Difference Convolutional Networks for Face Anti-Spoofing' 
By Zitong Yu & Zhuo Su, 2019

If you use the code, please cite:
@inproceedings{yu2020searching,
    title={Searching Central Difference Convolutional Networks for Face Anti-Spoofing},
    author={Yu, Zitong and Zhao, Chenxu and Wang, Zezheng and Qin, Yunxiao and Su, Zhuo and Li, Xiaobai and Zhou, Feng and Zhao, Guoying},
    booktitle= {CVPR},
    year = {2020}
}

Only for research purpose, and commercial use is not allowed.

MIT License
Copyright (c) 2020 
'''

from __future__ import print_function, division
import os
import torch
import pandas as pd
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pdb
import math
import os 
import imgaug.augmenters as iaa
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

#face_scale = 1.3  #default for test, for training , can be set from [1.2 to 1.5]
# data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
seq = iaa.Sequential([
    iaa.Add(value=(-40,40), per_channel=True), # Add color 
    iaa.GammaContrast(gamma=(0.5,1.5)) # GammaContrast with a gamma of 0.5 to 1.5
])

def crop_face_from_scene(image, bbox_path, scale):
    f=open(bbox_path,'r')
    lines=f.readlines()
    y1,x1,w,h=[float(ele) for ele in lines[:4]]
    f.close()
    y2=y1+w
    x2=x1+h

    y_mid=(y1+y2)/2.0
    x_mid=(x1+x2)/2.0
    h_img, w_img = image.shape[0], image.shape[1]
    #w_img,h_img=image.size
    w_scale=scale*w
    h_scale=scale*h
    y1=y_mid-w_scale/2.0
    x1=x_mid-h_scale/2.0
    y2=y_mid+w_scale/2.0
    x2=x_mid+h_scale/2.0
    y1=max(math.floor(y1),0)
    x1=max(math.floor(x1),0)
    y2=min(math.floor(y2),w_img)
    x2=min(math.floor(x2),h_img)

    #region=image[y1:y2,x1:x2]
    region=image[x1:x2,y1:y2]
    return region





# array
class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability = 0.5, sl = 0.01, sh = 0.05, r1 = 0.5, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, sample):
        img, map_x, spoofing_label = sample['image_x'], sample['map_x'],sample['spoofing_label']
        
        if random.uniform(0, 1) < self.probability:
            attempts = np.random.randint(1, 3)
            for attempt in range(attempts):
                area = img.shape[0] * img.shape[1]
           
                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1/self.r1)
    
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
    
                if w < img.shape[1] and h < img.shape[0]:
                    x1 = random.randint(0, img.shape[0] - h)
                    y1 = random.randint(0, img.shape[1] - w)

                    img[x1:x1+h, y1:y1+w, 0] = self.mean[0]
                    img[x1:x1+h, y1:y1+w, 1] = self.mean[1]
                    img[x1:x1+h, y1:y1+w, 2] = self.mean[2]
                    
        return {'image_x': img, 'map_x': map_x, 'spoofing_label': spoofing_label}


# Tensor
class Cutout(object):
    def __init__(self, length=50):
        self.length = length

    def __call__(self, sample):
        img, map_x, spoofing_label = sample['image_x'], sample['map_x'],sample['spoofing_label']
        h, w = img.shape[1], img.shape[2]    # Tensor [1][2],  nparray [0][1]
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        length_new = np.random.randint(1, self.length)
        
        y1 = np.clip(y - length_new // 2, 0, h)
        y2 = np.clip(y + length_new // 2, 0, h)
        x1 = np.clip(x - length_new // 2, 0, w)
        x2 = np.clip(x + length_new // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return {'image_x': img, 'map_x': map_x, 'spoofing_label': spoofing_label}


class Normaliztion(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __call__(self, sample):
        image_x, map_x, spoofing_label = sample['image_x'], sample['map_x'],sample['spoofing_label']
        new_image_x = (image_x - 127.5)/128     # [-1,1]
        new_map_x = map_x/255.0                 # [0,1]
        return {'image_x': new_image_x, 'map_x': new_map_x, 'spoofing_label': spoofing_label}



class RandomHorizontalFlip(object):
    """Horizontally flip the given Image randomly with a probability of 0.5."""
    def __call__(self, sample):
        image_x, map_x, spoofing_label = sample['image_x'], sample['map_x'],sample['spoofing_label']
        
        new_image_x = np.zeros((256, 256, 3))
        new_map_x = np.zeros((32, 32))

        p = random.random()
        if p < 0.5:
            #print('Flip')

            new_image_x = cv2.flip(image_x, 1)
            new_map_x = cv2.flip(map_x, 1)

                
            return {'image_x': new_image_x, 'map_x': new_map_x, 'spoofing_label': spoofing_label}
        else:
            #print('no Flip')
            return {'image_x': image_x, 'map_x': map_x, 'spoofing_label': spoofing_label}



class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        image_x, map_x, spoofing_label = sample['image_x'], sample['map_x'],sample['spoofing_label']
        
        # swap color axis because
        # numpy image: (batch_size) x H x W x C
        # torch image: (batch_size) x C X H X W
        image_x = image_x[:,:,::-1].transpose((2, 0, 1))
        image_x = np.array(image_x)
        
        map_x = np.array(map_x)
        
                        
        spoofing_label_np = np.array([0],dtype=np.long)
        spoofing_label_np[0] = spoofing_label
        
        
        return {'image_x': torch.from_numpy(image_x.astype(np.float)).float(), 'map_x': torch.from_numpy(map_x.astype(np.float)).float(), 'spoofing_label': torch.from_numpy(spoofing_label_np.astype(np.long)).long()}


class Spoofing_train(Dataset):

    def __init__(self, info_list, jpgs_dir, depth_maps_dir, bboxes_dir, transform=None):

        self.landmarks_frame = pd.read_csv(info_list, delimiter=',', header=None)
        self.jpgs_dir = jpgs_dir
        self.depth_maps_dir = depth_maps_dir
        self.bboxes_dir = bboxes_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)
    
    def __getitem__(self, idx):
        video_name = str(self.landmarks_frame.iloc[idx, 1])
        jpgs_path = os.path.join(self.jpgs_dir, video_name)
        depth_maps_path = os.path.join(self.depth_maps_dir, video_name)
        bboxs_path = os.path.join(self.bboxes_dir, video_name)
             
        image_x, map_x = self.get_single_image_x(jpgs_path, depth_maps_path, bboxs_path, video_name)

        spoofing_label = self.landmarks_frame.iloc[idx, 0]
        if spoofing_label == 1:
            spoofing_label = 1            # real
        else:
            spoofing_label = 0
            map_x = np.zeros((32, 32))    # fake

        
        sample = {'image_x': image_x, 'map_x': map_x, 'spoofing_label': spoofing_label}
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    # jpg_dir_path: /root/autodl-tmp/oulu_depth/Dev_jpgs/6_3_22_4
    # map_dir_path: /root/autodl-tmp/oulu_depth/Dev_depth/6_3_22_4
    # bbox_dir_path: /root/autodl-tmp/oulu_depth/Dev_bbox/6_3_22_4
    def get_single_image_x(self, jpg_dir_path, map_dir_path, bbox_dir_path, video_name):

        frames_total = len([name for name in os.listdir(map_dir_path) if os.path.isfile(os.path.join(map_dir_path, name))])
        # random choose 1 frame
        image_name = ''
        for i in range(500):
            image_id = np.random.randint(1, frames_total-1)
            image_name = "{:03}".format(image_id)
            bbox_path = os.path.join(bbox_dir_path, str(image_name)+'.dat')
            depth_map_path = os.path.join(map_dir_path, str(image_name)+'.jpg')
        
            # some .dat & map files have been missing  
            if os.path.exists(depth_map_path) and os.path.exists(bbox_path):
                depth_map_jpg = cv2.imread(depth_map_path, 0)
                if depth_map_jpg is not None:
                    break
        
        if image_name == '':
            print('error: ', depth_map_path, bbox_path)
            return np.zeros((256, 256, 3)), np.zeros((32, 32))
        
        # random scale from [1.2 to 1.5]
        face_scale = np.random.randint(12, 15)
        face_scale = face_scale/10.0
        
        # RGB
        jpg_path = os.path.join(jpg_dir_path, image_name + '.jpg')
        jpg_image = cv2.imread(jpg_path)

        # gray-map
        depth_map_path = os.path.join(map_dir_path, image_name + '.jpg')
        depth_map_image = cv2.imread(depth_map_path, 0)
        bbox_path = os.path.join(bbox_dir_path, image_name + '.dat')

        face_image_cropped = cv2.resize(crop_face_from_scene(jpg_image, bbox_path, face_scale), (256, 256))
        # data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
        image_x_aug = seq.augment_image(face_image_cropped) 

        depth_image_cropped = cv2.resize(crop_face_from_scene(depth_map_image, bbox_path, face_scale), (32, 32))
    
        return image_x_aug, depth_image_cropped




