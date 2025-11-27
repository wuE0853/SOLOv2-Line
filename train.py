#!/usr/bin/env python
# -*- coding:utf-8 -*-
import pdb
import torch
import cv2
import datetime

from torch.nn.utils import clip_grad
from model.solov2 import SOLOv2
from configs import * 
from utils import timer
from data_loader.build_loader import make_data_loader
from val import val


def show_ann(img, boxes, masks):
    img = img.cpu.numpy().astype('uint8')
    for i in range(img.shape[0]):
        img_np = img[i].transpose(1, 2, 0)
        one_box = boxes[i].cpu().numpy()
        one_mask = masks[i]

        for k in range(one_box.shape[0]):
            cv2.rectangle(img_np, 
                          (int(one_box[k, 0]), int(one_box[k, 1])),
                          (int(one_box[k, 2]), int(one_box[k, 3])), 
                          (0, 255, 0), 1)
            
        print(f'\nimg shape: {img_np.shape}')

        cv2.imshow('aa', img_np)
        cv2.waitKey()

        print('masks: ', one_mask.shape)
        for k in range(one_mask.shape[0]):
            one = one_mask[k].astype('uint8') * 200
            cv2.imshow('bb', one)
            cv2.waitKey()

