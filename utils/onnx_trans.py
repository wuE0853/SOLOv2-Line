#!usr/bin/env python
# -*- coding:utf-8 -*-
import pdb
import torch

def base_input_trans(x: torch.Tensor):
    # input as colored image
    x = x[:, :, (2, 1, 0)] # BGR to RGB
    x = x.float() / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0)
    return x