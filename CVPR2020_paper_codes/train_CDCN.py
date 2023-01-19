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
import torch
import matplotlib.pyplot as plt
import argparse,os
import pandas as pd
import cv2
import numpy as np
import random
import math
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from models.CDCNs import Conv2d_cd, CDCN, CDCNpp

from Load_OULUNPU_train import Spoofing_train, Normaliztion, ToTensor, RandomHorizontalFlip, Cutout, RandomErasing
from Load_OULUNPU_valtest import Spoofing_valtest, Normaliztion_valtest, ToTensor_valtest
from data_fetchers import DataFetcher

import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import copy
import pdb
from torchvision.utils import make_grid
from utils import AvgrageMeter, accuracy, performances, plot_save_jpg, cal_heatmap, tensorboard_add_image_sample, feature_2_heat_map
from loss import contrast_depth_loss


# jpg路径
train_image_dir = '/root/autodl-tmp/oulu_depth/Train_jpgs/'        
val_image_dir = '/root/autodl-tmp/oulu_depth/Dev_jpgs/' 
test_image_dir = '/root/autodl-tmp/oulu_depth/Test_jpgs/'    

# depth map路径
train_depth_map_dir = '/root/autodl-tmp/oulu_depth/Train_depth/'   
val_depth_map_dir = '/root/autodl-tmp/oulu_depth/Dev_depth/'    
test_depth_map_dir = '/root/autodl-tmp/oulu_depth/Test_depth/'

# bounding box路径
train_bbox_dir = '/root/autodl-tmp/oulu_depth/Train_bbox/'   
val_bbox_dir = '/root/autodl-tmp/oulu_depth/Dev_bbox/'    
test_bbox_dir = '/root/autodl-tmp/oulu_depth/Test_bbox/'

train_list = '/root/autodl-tmp/oulu/Protocols/Protocol_1/Train.txt'
val_list = '/root/autodl-tmp/oulu/Protocols/Protocol_1/Dev.txt'
test_list =  '/root/autodl-tmp/oulu/Protocols/Protocol_1/Test.txt'

writer = SummaryWriter('/root/tf-logs')

# main function
def train_test():
    #os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (args.gpu)
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_dir = os.path.join(args.log, now)
    if not os.path.exists(os.path.join(args.log, now)):
        os.makedirs(log_dir)
    
    log_file = open(os.path.join(log_dir, 'log.txt'), 'w')
    
    echo_batches = args.echo_batches
    display(log_file, "Oulu-NPU start train & test at %s" % now)

    # load the network, load the pre-trained model in UCF101?
    if args.finetune:
        display(log_file, 'finetune!\n')
        model = CDCN()
        model = model.cuda()
        model.load_state_dict(torch.load('xxx.pkl'))
    else:
        display(log_file, 'train from scratch!\n')
        model = CDCNpp(basic_conv=Conv2d_cd, theta=0.7)
        model = model.cuda()


    ## 初始化模型所需工具
    lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)    
    criterion_absolute_loss = nn.MSELoss().cuda()
    criterion_contrastive_loss = contrast_depth_loss().cuda() 

    # load random 24-frame clip data every epoch
    train_data = Spoofing_train(info_list=train_list, jpgs_dir=train_image_dir, depth_maps_dir=train_depth_map_dir, bboxes_dir=train_bbox_dir, transform=transforms.Compose([RandomErasing(), RandomHorizontalFlip(),  ToTensor(), Cutout(), Normaliztion()]))
    dataloader_train = DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=16, pin_memory=True)
    if torch.cuda.is_available():
        dataloader_train = DataFetcher(dataloader_train)

    loss_factor = 1000
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        loss_absolute, loss_contra = AvgrageMeter(), AvgrageMeter()

        model.train()
        for i, sample_batched in enumerate(dataloader_train):
            # get the inputs(batch_size, 3, 256, 256)
            inputs, map_label, spoof_label = sample_batched['image_x'].cuda(), sample_batched['map_x'].cuda(), sample_batched['spoofing_label'].cuda() 
            
            optimizer.zero_grad()

            map_x, embedding, x_Block1, x_Block2, x_Block3, x_input = model(inputs)

            absolute_loss = loss_factor * criterion_absolute_loss(map_x, map_label)
            contrastive_loss = loss_factor * criterion_contrastive_loss(map_x, map_label)
            loss =  absolute_loss + contrastive_loss

            loss.backward()
            optimizer.step()

            n = inputs.size(0)
            loss_absolute.update(absolute_loss.data, n)
            loss_contra.update(contrastive_loss.data, n)
            
            if (i + 1) % echo_batches == 0:
                feature_2_heat_map(log_dir, x_input, x_Block1, x_Block2, x_Block3, map_x, epoch, i)
        scheduler.step()
        
        writer.add_scalar('data/absolute_loss', loss_absolute.avg, epoch)
        writer.add_scalar('data/contrastive_loss', loss_contra.avg, epoch)
        writer.add_scalar('data/total_loss', loss_absolute.avg + loss_contra.avg , epoch)
        # whole epoch average
        display(log_file, 'epoch:%d, Train:  Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f\n' % (epoch + 1, loss_absolute.avg, loss_contra.avg))

        if (epoch + 1) % args.epoch_save == 0:  # save every 5 epochs
            torch.save(model.state_dict(), os.path.join(log_dir, 'epoch%d.pkl' % (epoch + 1)))
            display(log_file, 'save model at epoch: %d' % (epoch + 1))
        
        if (epoch + 1) % args.epoch_test == 0:    # test every 5 epochs  
            val_map_score_list = test_model(model=model, optimizer=optimizer, info_list=val_list, image_dir=val_image_dir, depth_map_dir=val_depth_map_dir, bbox_dir=val_bbox_dir, transforms=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
            test_map_score_list = test_model(model=model, optimizer=optimizer, info_list=test_list, image_dir=test_image_dir, depth_map_dir=test_depth_map_dir, bbox_dir=test_bbox_dir, transforms=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
            
            #############################################################     
            #       performance measurement both val and test
            #############################################################     
            val_threshold, test_threshold, val_ACC, val_ACER, test_ACC, test_APCER, test_BPCER, test_ACER, test_ACER_test_threshold = performances(val_map_score_list, test_map_score_list)
            writer.add_scalar('data/val_acc', val_ACC, epoch)
            writer.add_scalar('data/val_acer', val_ACER, epoch)
            writer.add_scalar('data/test_acc', test_ACC, epoch)
            writer.add_scalar('data/test_apcer', test_APCER, epoch)
            writer.add_scalar('data/test_bpcer', test_BPCER, epoch)
            writer.add_scalar('data/test_acer', test_ACER, epoch)

            display(log_file, 'epoch:%d, Val:  val_threshold= %.4f, val_ACC= %.4f, val_ACER= %.4f' % (epoch + 1, val_threshold, val_ACC, val_ACER))
            display(log_file, 'epoch:%d, Test:  ACC= %.4f, APCER= %.4f, BPCER= %.4f, ACER= %.4f \n' % (epoch + 1, test_ACC, test_APCER, test_BPCER, test_ACER))
            
            # save the model until the next improvement
            if epoch % args.epoch_save == 0:
                torch.save(model.state_dict(), os.path.join('checkpoint', 'CDCN_epoch_%d.pkl' % (epoch + 1)))


    display(log_file, 'Finished Training')
    log_file.close()

###########################################
'''            val     test        '''
###########################################
def test_model(model, optimizer, info_list, image_dir, depth_map_dir, bbox_dir, transforms=None):
    map_score_list = []

    model.eval()
    with torch.no_grad():
        # val for threshold
        data = Spoofing_valtest(info_list, image_dir, depth_map_dir, bbox_dir, transforms)
        dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=16)
        
        for i, sample_batched in enumerate(dataloader):
            # get the inputs (1, frames, 3, 256, 256)
            inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
            val_maps = sample_batched['val_map_x'].cuda()   # binary map from PRNet

            optimizer.zero_grad()
            
            map_score = torch.tensor(0.0).cuda()
            for frame_t in range(inputs.shape[1]):
                map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(inputs[:,frame_t,:,:,:])
                
                score_norm = torch.sum(map_x)/torch.sum(val_maps[:,frame_t,:,:])
                map_score += score_norm
            map_score = map_score/inputs.shape[1]
            map_score_list.append((map_score, spoof_label[0][0]))

    return map_score_list

def display(log_file, msg):
    print(msg)
    log_file.write(msg + '\n')
    log_file.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--gpu', type=int, default=0, help='the gpu id used for predict')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')  
    parser.add_argument('--batchsize', type=int, default=24, help='initial batchsize')  
    parser.add_argument('--step_size', type=int, default=500, help='how many epochs lr decays once')  # 500 
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--echo_batches', type=int, default=40, help='how many batches display once')  # 50
    parser.add_argument('--epochs', type=int, default=3000, help='total training epochs')
    parser.add_argument('--log', type=str, default="CDCN_log", help='log and save model name')
    parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune other models')
    parser.add_argument('--epoch_test', type=int, default=20)
    parser.add_argument('--epoch_save', type=int, default=50)

    args = parser.parse_args()
    train_test()
