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
#from skimage import io, transform
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pdb
import math
import os 


frames_total = 8    # each video 8 uniform samples
 
face_scale = 1.3  #default for test and val 
#face_scale = 1.1  #default for test and val

def crop_face_from_scene(image,face_name_full, scale):
    f=open(face_name_full,'r')
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




class Normaliztion_valtest(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __call__(self, sample):
        image_x, val_map_x, spoofing_label = sample['image_x'],sample['val_map_x'] ,sample['spoofing_label']
        new_image_x = (image_x - 127.5)/128     # [-1,1]
        return {'image_x': new_image_x, 'val_map_x': val_map_x , 'spoofing_label': spoofing_label}


class ToTensor_valtest(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        image_x, val_map_x, spoofing_label = sample['image_x'],sample['val_map_x'] ,sample['spoofing_label']
        
        # swap color axis because    BGR2RGB
        # numpy image: (batch_size) x T x H x W x C
        # torch image: (batch_size) x T x C X H X W
        image_x = image_x[:,:,:,::-1].transpose((0, 3, 1, 2))
        image_x = np.array(image_x)
        
        val_map_x = np.array(val_map_x)
                        
        spoofing_label_np = np.array([0],dtype=np.long)
        spoofing_label_np[0] = spoofing_label
        
        return {'image_x': torch.from_numpy(image_x.astype(np.float)).float(), 'val_map_x': torch.from_numpy(val_map_x.astype(np.float)).float(),'spoofing_label': torch.from_numpy(spoofing_label_np.astype(np.long)).long()} 


# jpg_dir_path: /root/autodl-tmp/oulu_depth/Dev_jpgs/6_3_22_4
# map_dir_path: /root/autodl-tmp/oulu_depth/Dev_depth/6_3_22_4
# bbox_dir_path: /root/autodl-tmp/oulu_depth/Dev_bbox/6_3_22_4
class Spoofing_valtest(Dataset):

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

        sample = {'image_x': image_x, 'val_map_x': map_x, 'spoofing_label': spoofing_label}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_single_image_x(self, jpg_dir_path, map_dir_path, bbox_dir_path, videoname):
        files_total = len([name for name in os.listdir(jpg_dir_path) if os.path.isfile(os.path.join(jpg_dir_path, name))])//3
        interval = files_total//10
        
        image_x = np.zeros((frames_total, 256, 256, 3))
        val_map_x = np.ones((frames_total, 32, 32))
        
        # random choose 1 frame
        image_name = ''
        for i in range(frames_total):
            image_id = i*interval + 1 
            flag = False
            for temp in range(50):
                image_name = "{:03}".format(image_id)
                bbox_path = os.path.join(bbox_dir_path, str(image_name)+'.dat')
                depth_map_path = os.path.join(map_dir_path, str(image_name)+'.jpg')
                
                if os.path.exists(bbox_path) and os.path.exists(depth_map_path)  :    # some scene.dat are missing
                    depth_map_jpg = cv2.imread(depth_map_path, 0)
                    if depth_map_jpg is not None:
                        break
                    else:
                        image_id +=1
                else:
                    image_id +=1
        
            # RGB
            jpg_path = os.path.join(jpg_dir_path, image_name + '.jpg')
            jpg_image = cv2.imread(jpg_path)
            
            # gray-map
            depth_map_path = os.path.join(map_dir_path, image_name + '.jpg')
            depth_map_image = cv2.imread(depth_map_path, 0)

            image_x[i,:,:,:] = cv2.resize(crop_face_from_scene(jpg_image, bbox_path, face_scale), (256, 256))
            # transform to binary mask --> threshold = 0 
            depth_image_cropped = cv2.resize(crop_face_from_scene(depth_map_image, bbox_path, face_scale), (32, 32))
            
            np.where(depth_image_cropped < 1, depth_image_cropped, 1)
            val_map_x[i,:,:] = depth_image_cropped
            
			
        return image_x, val_map_x

if __name__ == '__main__':
    path = '/root/autodl-tmp/oulu_depth/Dev_depth/3_2_24_5/065.jpg'
    print(os.path.exists(path))

        