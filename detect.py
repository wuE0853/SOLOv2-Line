#!/usr/bin/env python
# -*- coding:uft-8 -*-
import os
import pdb
import shutil
import datetime

import cv2
import numpy as np

from model.solov2 import SOLOv2
from configs import *
from utils import timer
from scipy import ndimage
from data_loader.build_loader import make_data_loader

PALLETE = [
    # orange and green
    [32, 243, 119], [18, 204, 255], [0, 255, 162], [51, 204, 255], [0, 230, 199],
    [0, 255, 110], [0, 212, 255], [0, 255, 140], [0, 255, 85], [0, 199, 255],
    [0, 255, 179], [0, 255, 64], [0, 179, 255], [0, 255, 225], [0, 140, 255],
    [0, 255, 47], [0, 162, 255], [0, 255, 199], [0, 255, 102], [0, 225, 255],
    [0, 119, 255], [0, 255, 28], [0, 243, 255], [0, 255, 162], [0, 199, 255],
    [0, 85, 255], [0, 255, 140], [0, 255, 225], [0, 255, 64], [0, 162, 255],
    [0, 230, 255], [0, 255, 110], [0, 255, 47], [0, 140, 255], [0, 255, 179],
    [0, 255, 85], [0, 199, 255], [0, 255, 28], [0, 119, 255], [0, 243, 255],
    [0, 255, 162], [0, 225, 255], [0, 64, 255], [0, 255, 140], [0, 255, 199],
    [0, 255, 102], [0, 179, 255], [0, 255, 47], [0, 162, 255], [0, 255, 225],
    [0, 255, 110], [0, 199, 255], [0, 85, 255], [0, 255, 140], [0, 255, 64],
    [0, 230, 255], [0, 255, 179], [0, 255, 28], [0, 119, 255], [0, 243, 255],
    [0, 255, 162], [0, 225, 255], [0, 47, 255], [0, 255, 140], [0, 255, 199],
    [0, 255, 102], [0, 179, 255], [0, 255, 64], [0, 162, 255], [0, 255, 225],
    [0, 255, 110], [0, 199, 255], [0, 85, 255], [0, 255, 140], [0, 255, 47],
    [0, 230, 255], [0, 255, 179], [0, 255, 28], [0, 119, 255], [0, 243, 255]
]


if __name__ == '__main__':
    cfg = Line_solov2_r50_light(mode='detect')
    cfg.print_cfg

    model = SOLOv2(cfg).cuda()
    state_dict = torch.load(cfg.val_weight)
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    print(f'Detecting with "{cfg.val_weight}.')

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    save_path = 'results/detect/line'
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    data_loader = make_data_loader(cfg)
    dataset = data_loader.dataset
    val_num = len(data_loader)
    print(f'Length of dataloader: {val_num}.')
    timer.reset(reset_at=0)

    segm_json_results = []
    for i, (img, resize_shape, img_name, img_reszied) in enumerate(data_loader):
        timer.start()

        with torch.no_grad(), timer.counter('forward'):
            seg_result = model(img.cuda().detach, resize_shape=resize_shape, post_mode='detect')[0]

        with timer.counter('draw'):
            if seg_result is not None:
                seg_pred = seg_result[0].cpu().numpy()
                cate_label = seg_result[1].cpu().numpy()
                cate_score = seg_result[2].cpu().numpy()

                seg_show = img_reszied.copy()
                for j in range(seg_pred.shape[0]):
                    cur_mask = seg_pred[j, :, :]
                    assert cur_mask.sum() != 0, 'current mask sum == 0.'

                    cur_cate = cate_label[j]
                    cur_score = cate_score[j]

                    color = PALLETE[j]

                    if cfg.detect_mode == 'overlap':
                        mask_bool = cur_mask.astype('bool')
                        seg_show[mask_bool] = img_reszied[mask_bool] * 0.5 + np.array(color, dtype='uint8') * 0.5
                    elif cfg.detect_mode == 'contour':
                        _, img_thre = cv2.threshold(cur_mask, 0, 255, cv2.THRESH_BINARY)
                        contours, _ = cv2.findContours(img_thre, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(seg_show, contours, coutourIdx=-1, color=color, thickness=1)

                    
                    label_text = f'{cfg.class_names[cur_cate]} {cur_score:.02f}'
                    center_y, center_x = ndimage.center_of_mass(cur_mask)
                    vis_pos = max((int(center_x) - 10, 0), int(center_y))
                    cv2.putText(seg_show, label_text, vis_pos, cv2.FONT_HERSHEY_COMPLEX, 0.4, tuple(color))


                # cv2.imshow('aa', seg_show)
                # cv2.waitKey()
                cv2.imwrite(f'{save_path}/{img_name}', seg_show)
            else:
                cv2.imwrite(f'{save_path}/{img_name}', img_reszied)

        timer.add_batch_time()

        if timer.started:
            t_t, t_d, t_f, t_draw = timer.get_times(['batch', 'data', 'forward', 'draw'])
            seconds = (val_num - i) * t_t
            eta = str(datetime.timedelta(seconds=seconds)).split('.')[0]

            print(f'\rDetecting: {i + 1}/{val_num} | t_t: {t_t:.3f} | t_d: {t_d:.3f} | t_f: {t_f:.3f}'
                  f' | t_draw: {t_draw:.3f} | ETA: {eta}', end='')
            
    print()
    print(f'Done, results saved in "{save_path}".')
