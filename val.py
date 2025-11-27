#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import pdb
import datetime
import json
import numpy as np
from model.solov2 import SOLOv2
from configs import *
from utils import timer
from utils.cocoeval import SelfEval
from data_loader.build_loader import make_data_loader
import pycocotools.mask as mask_util


def val(cfg, model = None):
    if model is None:
        model = SOLOv2(config).cuda()
        state_dict = torch.load(cfg.val_weight)
        if 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict, strict=True)
        print(f'Evaluating "{cfg.val_weight}"')