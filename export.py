#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import numpy as np
import torch.jit
import utils.onnx_trans as onnx_trans

from model.solov2 import SOLOv2
from configs import *
import cv2

if __name__ == '__main__':
    cfg = Line_solov2_r50_light(mode='onnx')
    cfg.print_cfg()

    model = SOLOv2(cfg).cuda()
    state_dict = torch.load(cfg.val_weight)
    print(f'Detecting with {cfg.val_weight}.')

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    os.makedirs('./export_files', exist_ok=True)
    file_path = f'./export_files/{cfg.__class__.__name__}.pt'

    input_size = cfg.onnx_shape
    class_suffix = '-'.join(list(cfg.class_names))
    inp_names = [f'seg_{input_size[0]}_{input_size[1]}-{class_suffix}']

    input_img = cv2.imread('dataset/line_sep_d/train/244.jpg', cv2.IMREAD_COLOR)
    input_img = cv2.resize(input_img, input_size)
    input_img = torch.from_numpy(input_img).cuda()

    input_img = onnx_trans.base_input_trans(input_img)

    model = torch.jit.trace(model, input_img)
    model.save(file_path)
    print(f'Saved to {file_path}.\n')
