#!/usr/bin/env python
# encoding: utf-8
'''
@author: tianxiaomo
@license: (C) Apache.
@contact: huguanghao520@gmail.com
@software: PyCharm
@file: train.py
@time: 2019/10/8 9:43
@desc:
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader,Dataset

import adabound
from torchsummary import summary
import torchvision.models as models

from sys import argv
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
from easydict import EasyDict as edict
import os, csv, glob, cv2, math, argparse, datetime,logging,sys,random,ast

# 参数定义部分
def get_config():
    config = edict()
    config.train_dir = 'data'
    config.checkpoints_dir = 'checkpoints'
    config.val_dir = 'data'
    # hard nagative mining ratio, needed by loss layer
    config.param_hnm_ratio = 5

    # the number of image channels
    config.param_num_image_channel = 3

    # the number of output scales (loss branches)
    config.param_num_output_scales = 8

    # feature map size for each scale
    config.param_feature_map_size_list = [159, 159, 79, 79, 39, 19, 19, 19]

    # bbox lower bound for each scale
    config.param_bbox_small_list = [5, 15, 20, 40, 70, 110, 250, 400]
    assert len(config.param_bbox_small_list) == config.param_num_output_scales

    # bbox upper bound for each scale
    config.param_bbox_large_list = [15, 20, 40, 70, 110, 250, 400, 560]
    assert len(config.param_bbox_large_list) == config.param_num_output_scales

    # bbox gray lower bound for each scale
    config.param_bbox_small_gray_list = [math.floor(v * 0.9) for v in config.param_bbox_small_list]
    # bbox gray upper bound for each scale
    config.param_bbox_large_gray_list = [math.ceil(v * 1.1) for v in config.param_bbox_large_list]

    # the RF size of each scale used for normalization, here we use param_bbox_large_list for better regression
    config.param_receptive_field_list = config.param_bbox_large_list
    # RF stride for each scale
    config.param_receptive_field_stride = [4, 4, 8, 8, 16, 32, 32, 32]
    # the start location of the first RF of each scale
    config.param_receptive_field_center_start = [3, 3, 7, 7, 15, 31, 31, 31]

    config.param_normalization_constant = [i/(2.0*j) for i,j in zip(config.param_receptive_field_list,config.param_receptive_field_stride)]

    # the sum of the number of output channels, 2 channels for classification and 2 for bbox regression
    config.param_num_output_channels = 4

    config.num_workers = 1
    # config.batch_size = config.num_workers
    config.batch_size = 2
    config.gpus = '6'
    config.epochs = 4000
    config.lr = 5e-2
    config.optimizer = 'SGD'
    resume_epoch = 0
    # pre_weights = os.path.join(config.checkpoints_dir, 'ctpn_ep50_0.0075_0.0190_0.0264.pth.tar')
    config.pre_weights = None
    config.IMAGE_MEAN = [123.68, 116.779, 103.939]

    config.param_scheduler_step_list = [50, 100, 150]
    config.param_scheduler_factor = 0.1

    return config

# 其他方法
class LFCD_Unit:
    @classmethod
    def readtxt(cls,p):
        '''
        load annotation from the text file
        :param p:
        :return:
        '''
        text_polys = []
        text_tags = []
        if not os.path.exists(p):
            return np.array(text_polys, dtype=np.float32)
        with open(p, 'r',encoding='utf-8') as f:
            reader = csv.reader(f)
            for line in reader:
                label = line[-1]
                # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
                line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]

                x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
                text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                if label == '*' or label == '###':
                    text_tags.append(True)
                else:
                    text_tags.append(False)
            return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool)

    @classmethod
    def save_checkpoint(cls,log,checkpoints_dir,state, epoch, loss_cls, loss_regr, loss, ext='pth.tar'):
        if not os.path.exists(checkpoints_dir):
            os.mkdir(checkpoints_dir)
        check_path = os.path.join(checkpoints_dir,
                                  f'ctpn_ep{epoch:02d}_'
                                  f'{loss_cls:.4f}_{loss_regr:.4f}_{loss:.4f}.{ext}')
        torch.save(state, check_path)
        log.info('saving to {}'.format(check_path))

    @classmethod
    def get_date_str(cls):
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')

    @classmethod
    def init_logger(cls,log_file=None, log_path=None, log_level=logging.DEBUG, mode='w', stdout=True):
        """
        log_path: 日志文件的文件夹路径
        mode: 'a', append; 'w', 覆盖原文件写入.
        """
        fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
        if log_path is None:
            log_path = '~/temp/log/'
        if log_file is None:
            log_file = 'log_' + LFCD_Unit.get_date_str() + '.log'
        log_file = os.path.join(log_path, log_file)
        # 此处不能使用logging输出
        print('log file path:' + log_file)
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        logging.basicConfig(level=log_level,
                            format=fmt,
                            filename=os.path.abspath(log_file),
                            filemode=mode)

        if stdout:
            console = logging.StreamHandler(stream=sys.stdout)
            console.setLevel(log_level)
            formatter = logging.Formatter(fmt)
            console.setFormatter(formatter)
            logging.getLogger('').addHandler(console)

        return logging

    @classmethod
    def collate(cls,batch):
        r"""Puts each data field into a tensor with outer dimension batch size"""
        imgs,fea1,fea2,fea3,fea4,fea5,fea6,fea7,fea8 = [],[],[],[],[],[],[],[],[]
        mask1,mask2,mask3,mask4,mask5,mask6,mask7,mask8 = [],[],[],[],[],[],[],[]
        for img,fea,mask in batch:
            imgs.append(img)
            fea1.append(fea[0])
            fea2.append(fea[1])
            fea3.append(fea[2])
            fea4.append(fea[3])
            fea5.append(fea[4])
            fea6.append(fea[5])
            fea7.append(fea[6])
            fea8.append(fea[7])
            mask1.append(mask[0])
            mask2.append(mask[1])
            mask3.append(mask[2])
            mask4.append(mask[3])
            mask5.append(mask[4])
            mask6.append(mask[5])
            mask7.append(mask[6])
            mask8.append(mask[7])

        imgs = torch.stack(imgs, 0)
        fea_list = []
        fea_list.append(torch.stack(fea1,0))
        fea_list.append(torch.stack(fea2,0))
        fea_list.append(torch.stack(fea3,0))
        fea_list.append(torch.stack(fea4,0))
        fea_list.append(torch.stack(fea5,0))
        fea_list.append(torch.stack(fea6,0))
        fea_list.append(torch.stack(fea7,0))
        fea_list.append(torch.stack(fea8,0))
        mask_list = []
        mask_list.append(torch.stack(mask1,0))
        mask_list.append(torch.stack(mask2,0))
        mask_list.append(torch.stack(mask3,0))
        mask_list.append(torch.stack(mask4,0))
        mask_list.append(torch.stack(mask5,0))
        mask_list.append(torch.stack(mask6,0))
        mask_list.append(torch.stack(mask7,0))
        mask_list.append(torch.stack(mask8,0))
        return imgs,fea_list,mask_list

    @classmethod
    def get_arguments(cls,config):
        parser = argparse.ArgumentParser(description='Pytorch CTPN For TexT Detection')
        parser.add_argument('--num-workers', type=int, default=config.num_workers)
        parser.add_argument('--batch-size',type=int,default=config.batch_size)
        parser.add_argument('--train-dir', type=str, default=config.train_dir)
        parser.add_argument('--val-dir', type=str, default=config.val_dir)
        parser.add_argument('--pretrained-weights', type=str,default=config.pre_weights)
        parser.add_argument('--gpus',type=str,default=config.gpus)
        parser.add_argument('--opt', type=str, default=config.optimizer)
        parser.add_argument('--lr',type=float,default=config.lr)
        parser.add_argument('--val',type=ast.literal_eval,default=True)
        parser.add_argument('--train_enhance',type=ast.literal_eval,default=False)
        parser.add_argument('--val_enhance',type=ast.literal_eval,default=False)
        return parser.parse_args()

# 模型定义部分
class ResBlock(nn.Module):
    def __init__(self,channels):
        super(ResBlock, self).__init__()
        self.conv2dRelu = nn.Sequential(
            nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1),
            nn.ReLU6(channels),
            nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1),
        )
        self.relu = nn.ReLU6(channels)
    def forward(self,x):
        return self.relu(x + self.conv2dRelu(x))

class LFCDLossBranch(nn.Module):
    def __init__(self,in_channels,out_channels=64,num_classes=2):
        super(LFCDLossBranch, self).__init__()
        self.conv1x1relu = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0),
            nn.ReLU6(out_channels)
        )
        self.score =nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size=1,stride=1,padding=0),
            nn.ReLU6(out_channels),
            nn.Conv2d(out_channels,num_classes,kernel_size=1,stride=1)
        )
        self.locations =nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size=1,stride=1,padding=0),
            nn.ReLU6(out_channels),
            nn.Conv2d(out_channels,2,kernel_size=1,stride=1)
        )
    def forward(self,x):
        score = self.score(self.conv1x1relu(x))
        locations = self.locations(self.conv1x1relu(x))
        return score,locations

class LFCDNet(nn.Module):
    def __init__(self,num_classes = 2):
        super(LFCDNet, self).__init__()
        self.num_classes = num_classes
        self.priors = None
        self.c1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=2,padding=0),
            nn.ReLU6(64)
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,stride=2,padding=0),
            nn.ReLU6(64)
        )
        self.tinypart1 = nn.Sequential(
            ResBlock(64),
            ResBlock(64),
            ResBlock(64)
        )
        self.tinypart2 = ResBlock(64)
        self.c11 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,stride=2,padding=0),
            nn.ReLU6(64)
        )
        self.smallpart1 = ResBlock(64)
        self.smallpart2 = ResBlock(64)
        self.c16 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3,stride=2,padding=0),
            nn.ReLU6(128)
        )
        self.mediumpart = ResBlock(128)
        self.c19 = nn.Sequential(
            nn.Conv2d(128,128,kernel_size=3,stride=2,padding=0),
            nn.ReLU6(128)
        )
        self.largepart1 = ResBlock(128)
        self.largepart2 = ResBlock(128)
        self.largepart3 = ResBlock(128)


        self.lossbranch1 = LFCDLossBranch(64,num_classes = self.num_classes)
        self.lossbranch2 = LFCDLossBranch(64,num_classes = self.num_classes)
        self.lossbranch3 = LFCDLossBranch(64,num_classes = self.num_classes)
        self.lossbranch4 = LFCDLossBranch(64,num_classes = self.num_classes)
        self.lossbranch5 = LFCDLossBranch(128,num_classes = self.num_classes)
        self.lossbranch6 = LFCDLossBranch(128,num_classes = self.num_classes)
        self.lossbranch7 = LFCDLossBranch(128,num_classes = self.num_classes)
        self.lossbranch8 = LFCDLossBranch(128,num_classes = self.num_classes)

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(c1)

        c8 = self.tinypart1(c2)
        c10 = self.tinypart2(c8)

        c11 = self.c11(c10)
        c13 = self.smallpart1(c11)
        c15 = self.smallpart2(c13)

        c16 = self.c16(c15)
        c18 = self.mediumpart(c16)

        c19 = self.c19(c18)
        c21 = self.largepart1(c19)
        c23 = self.largepart2(c21)
        c25 = self.largepart3(c23)

        score1,loc1 = self.lossbranch1(c8)
        score2,loc2 = self.lossbranch2(c10)
        score3,loc3 = self.lossbranch3(c13)
        score4,loc4 = self.lossbranch4(c15)
        score5,loc5 = self.lossbranch5(c18)
        score6,loc6 = self.lossbranch6(c21)
        score7,loc7 = self.lossbranch7(c23)
        score8,loc8 = self.lossbranch8(c25)

        cls,loc = [],[]
        cls.append(score1)
        cls.append(score2)
        cls.append(score3)
        cls.append(score4)
        cls.append(score5)
        cls.append(score6)
        cls.append(score7)
        cls.append(score8)
        loc.append(loc1)
        loc.append(loc2)
        loc.append(loc3)
        loc.append(loc4)
        loc.append(loc5)
        loc.append(loc6)
        loc.append(loc7)
        loc.append(loc8)

        return (cls,loc)
