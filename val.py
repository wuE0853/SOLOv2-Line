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
    
    cfg.eval()
    model.eval()
    data_loader = make_data_loader(cfg)
    dataset = data_loader.dataset

    val_num = len(data_loader)
    print(f'Length of DataLoader: {val_num}.')
    timer.reset(reset_at=0)

    dt_id = 1 #cannot be the value that equal to  False(bool)
    json_result = []
    for i, (img, ori_shape, resize_shape, _) in enumerate(data_loader):
        timer.start(i)

        img = img.cuda().detach()

        with torch.no_grad(), timer.counter('forward'):
            seg_result = model(img, ori_shape=ori_shape, resize_shape=resize_shape, post_mode='val')[0]

        with timer.counter('metric'):
            if seg_result is not None:
                seg_pred = seg_result[0].cpu().numpy()
                cate_label = seg_result[1].cpu().numpy()
                cate_score = seg_result[2].cpu().numpy()

                for j in range(seg_pred.shape[0]):
                    data = dict()
                    cur_mask = seg_pred[j, ...]
                    data['image_id'] = dataset.ids[i]
                    data['score'] = float(cate_score[j])

                    # rle encoding(游程编码) to reduce the space cost
                    rle = mask_util.encode(np.array(cur_mask[:, :, np.newaxis], order='F'))[0]
                    rle['count'] = rle['count'].decode
                    data['segmentation'] = rle

                    if 'Coco' in dataset.__class__.__name__ or 'Line' in dataset.__class__.__name__:
                        data['categoty_id'] = dataset.cate_ids[cate_label[j] + 1]
                    # else:
                    # The need of NonCOCO dataset

                    dt_id += 1
                    json_result.append(data)
        
        timer.add_batch_time()

        if timer.started():
            t_batch, t_data, t_forward, t_metric = timer.get_times(['batch', 'data', 'forward', 'metric'])
            seconds = (val_num - i) * t_batch
            eta = str(datatime.timedelta(seconds)).split('.')

            print(f'\rstep: {i}/{val_num} | t_batch: {t_batch:.3f} | t_d: {t_data:.3f} | t_f: {t_forward:.3f} | t_m: {t_metric:.3f} |'
                  f'ETA: {eta}', end='')
            
            print('\n\n')

            file_folder = 'result/val'
            os.makedirs(file_folder, exist_ok=True)
            file_path = f'{file_folder}/{cfg.name()}.json'
            with open(file_path, 'w') as f:
                json.dump(json_result, f)
            print(f'val result dumped: {file_path}.\n')

            if 'Coco' in dataset.__class__.__name__ or 'Line' in dataset.__class__.__name__:
                coco_dt = dataset.coco.loadRes(file_path)
                segm_eval = SelfEval(dataset.coco, coco_dt)
            # else:
            # nonCOCO dataset and annotations should use other funcntion to get gt and dt

            segm_eval.evaluate()
            segm_eval.accumulate()
            segm_eval.summarize()


if __name__ == '__main__':
    cfg = Line_solov2_r50_light(mode='val')
    cfg.print_cfg()
    val(cfg)
