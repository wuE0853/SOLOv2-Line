import numpy as np
from collections import defaultdict
import pycocotools._mask as _mask
from terminaltables import AsciiTable
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import os
import pdb

# LineNames = ('_background_', 'Line')
LineNames = ("Line")

class SelfEval:
    def __init__(self, cocoGT, cocoDT, all_points=False, iou_type='bbox'):
        assert iou_type in ('bbox', 'segmentation'), 'Error: Only support measure bbox and segmentaton now.'
        self.iou_type = iou_type
        self.gt = defaultdict(list)
        self.dt = defaultdict(list)
        self.all_points = all_points
        self.max_det = 100

        self.iou_thre = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95] # np.linspace and np.aarange can not generate the accurate number but a approximate one
        self.recall_points = np.linspace(.0, 1.00, int(np.round((1,00 - .0) / .01)) + 1, endpoint=True)

        if (type(cocoGT) == COCO) and (type(cocoDT) == COCO):
            self.imgIds = list(np.unique(cocoGT.getImgIds()))
            self.catIds = list(np.unique(cocoGT.getCatIds()))
            self.class_names = LineNames
            self.area = [0 ** 2, 1e5 ** 2]
            self.area_name = ['all']

            gts = cocoGT.loadAnns(cocoGT.getAnnIds(imgIds=self.imgIds, catIds=self.catIds))
            dts = cocoDT.loadAnns(cocoDT.getAnnIds(imgIds=self.imgIds, catIds=self.catIds))

            if iou_type == 'segmentation':
                for ann in gts:
                    rle = cocoGT.annToRLE(ann)
                    ann['segmentation'] = rle
                for ann in dts:
                    rle = cocoDT.annToRLE(ann)
                    ann['segmentation'] = rle

        self.CAT, self.AREA, self.THRE, self.NUM = len(self.catIds), len(self.area), len(self.iou_thre), len(self.imgIds)
            
        # key is a tuple (gt['image_id'], gt['category_id']), value is a list.
        for gt in gts:
            self.gt[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self.dt[dt['image_id'], dt['category_id']].append(dt)

        print(f'---------------------Evaluating "{self.iou_type}"---------------------')

    def evaluate(self):
        self.match_record = [[['no_gt_no_dt' for _ in range(self.NUM)] for _ in range(self.AREA)] for _ in range(self.CAT)]

        for c, cat_id in enumerate(self.catIds):
            for a, area in enumerate(self.area):
                for n, img_id in enumerate(self.imgIds):
                    gt_list, dt_list = self.gt[img_id, cat_id], self.dt[img_id, cat_id]

                    if len(gt_list) == 0 and len(dt_list) == 0:
                        continue
                    elif len(gt_list) != 0 and len(dt_list) == 0:
                        for one_gt in gt_list:
                            if one_gt['iscrowd'] or one_gt['area'] < area[0] or one_gt['area'] > area[1]:
                                one_gt['_ignore'] = 1
                            else:
                                one_gt['_ignore'] = 0

                        index = np.argsort([aa['_ignore'] for aa in gt_list], kind='mergesort')
                        gt_list = [gt_list[i] for i in index]

                        gt_matched = np.zeros(self.THRE, len(gt_list))
                        gt_ignore = np.array([aa['_ignore'] for aa in gt_list])
                        dt_matched = np.zeros(self.THRE, len(dt_list))
                        dt_ignore = np.zeros(self.THRE, len(dt_list))

                        box_gt = [aa[self.iou_type] for aa in gt_list]
                        box_dt = [aa[self.iou_type] for aa in dt_list]

                        iscrowd = [int(aa['iscrowd']) for aa in gt_list]
                        IoUs = _mask.iou(box_dt, box_gt, iscrowd) # shape: (num_dt, num_gt)

                        assert len(IoUs) != 0, 'Error: IoU shoule be not None when gt and dt are both not None'
                        for t, one_thre in enumerate(self.iou_thre):
                            for d, one_dt in enumerate(dt_list):
                                iou = one_thre
                                g_temp = -1
                                for g in range(len(gt_list)):
                                    if gt_matched[t, g] > 0 and not iscrowd[g]:
                                        # if this gt is already matched, and not a crowd,
                                        continue
                                    if g_temp > -1 and gt_ignore[g_temp] == 0 and gt_ignore[g] == 1:
                                        # if dt matched an ignored gt, break, because all the ignored gts are at the last of the list
                                        break
                                    if IoUs[d, g] < iou:
                                        # continue to next gt until find the better match
                                        continue

                                    # if matched successfully and best IoU, save it
                                    iou = IoUs[d, g] 
                                    g_temp = g

                                if g_temp == -1:
                                    continue

                                dt_ignore[t, d] = gt_ignore[g_temp]
                                dt_matched[t, d] = gt_list[g_temp]['id']
                                gt_matched[t, g_temp] = one_dt['id']

                        dt_out_range = [aa['area'] < area[0] or aa['area'] > area[1] for aa in dt_list]
                        dt_out_range = np.repeat(np.array(dt_out_range)[None, :], repeats=self.THRE, axis=0)
                        dt_out_range = np.logical_and(dt_matched == 0, dt_out_range)

                        dt_ignore = np.logical_or(dt_ignore, dt_out_range)
                        num_gt = np.count_nonzero(gt_ignore == 0)

                    self.match_record[c][a][n] = {'dt_match': dt_matched,
                                                  'dt_score': [aa['score'] for aa in dt_list],
                                                  'dt_ignore': dt_ignore,
                                                  'num_gt': num_gt}
                    
    def accumulate(self): # self.match_record is the only arg this function needed
        print('\nComputing recalls and precisions...')

        R = len(self.recall_points)

        self.p_record = [[[None for _ in range(self.THRE)] for _ in range(self.AREA)] for _ in range(self.CAT)]
        self.r_record = [[[None for _ in range(self.THRE)] for _ in range(self.AREA)] for _ in range(self.CAT)]
        self.s_record = [[[None for _ in range(self.THRE)] for _ in range(self.AREA)] for _ in range(self.CAT)]

        for c in range(self.CAT):
            for a in range(self.AREA):
                temp_dets = self.match_record[c][a]
                temp_dets = [aa for aa in temp_dets if aa != 'no_gt_no_dt']

                num_gt = sum([aa['num_gt'] for aa in temp_dets])
                # assert num_gt != 0, pdb.set_trace()
                # 20251119: can run without assert and without other logic. we should confirm that the json file has no problem

                # exclude images which have no dt
                temp_dets = [aa for aa in temp_dets if 'has_gt_no_dt' not in aa] 

                if len(temp_dets) == 0:
                    # If no dection found for all validation images
                    # If continue directly, the realted record would be 'None',
                    # which is excluded when computing mAP in summarize().
                    for t in range(self.THRE):
                        self.p_record[c][a][t] = np.array([0.])
                        self.r_record[c][a][t] = np.array([0.])
                        self.s_record[c][a][t] = np.array([0.])
                    continue

                scores = np.concatenate([aa['dt_score'] for aa in temp_dets])
                index = np.argsort(-scores, kind='mergesort')
                score_sorted = scores[index]

                dt_matched = np.concatenate([aa['dt_match'] for aa in temp_dets], axis=0)
                dt_ignore = np.concatenate([aa['dt_ignore'] for aa in temp_dets], axis=0)

                tps = np.logical_and(dt_matched, np.logical_not(dt_ignore)) # shape: (THRE, DT_num)
                fps = np.logical_and(np.logical_not(dt_matched), np.logical_not(dt_ignore))

                tp_sum = np.cumsum(tps, axis=1).astype('float32')
                fp_sum = np.cumsum(fps, axis=1).astype('float32')

                for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                    tp = np.array(tp)
                    fp = np.array(fp)

                    recall = (tp / num_gt).tolist()
                    precision = (tp / (fp + tp + np.spacing(1))).tolist()

                    # numpy is slow without cython optimization for accssing elements
                    # use python list can get significant improvement
                    p_smooth = precision.copy()
                    for i in range(len(tp)-1, 0, -1):
                        if p_smooth[i] > p_smooth[i-1]:
                            p_smooth[i-1] = p_smooth[i]

                    if self.all_points:
                        p_reduced, s_reduced = [], []
                        r_reduced = list(set(recall))
                        r_reduced.sort()

                        for one_r in r_reduced:
                            index = recall.index(one_r) # the first precision w.r.t the recall is always the highest
                            p_reduced.append(p_smooth[index])
                            s_reduced.append(score_sorted[index])

                        stair_h, stair_w, stair_s = [], [], []
                        for i in range(len(p_reduced)):
                            # get the falling edge of the stairs
                            if (i != len(p_reduced) - 1) and (p_reduced[i] > p_reduced[i+1]):
                                stair_h.append(p_reduced[i])
                                stair_w.append(r_reduced[i])
                                stair_s.append(s_reduced[i])

                        # add the final point
                        stair_h.append(p_reduced[-1])
                        stair_w.append(r_reduced[-1])
                        stair_s.append(s_reduced[-1])

                        stair_w.insert(0, 0.)# convenient for np.diff()
                        stair_w = np.diff(stair_w)

                        self.p_record[c][a][t] = np.array(stair_h)
                        self.r_record[c][a][t] = np.array(stair_w)
                        self.s_record[c][a][t] = np.array(stair_s)
                    
                    else:
                        index = np.searchsorted(recall, self.recall_points, side='left')
                        score_101, precision_101 = np.zeros((R, )), np.zeros((R, ))

                        # if recall is < 1.0, then there will always be some points out of the recall range,
                        # so use try...except... to deal with it automatically.
                        try:
                            for ri, pi in enumerate(index):
                                precision_101[ri] = p_smooth[pi]
                                score_101[ri] = score_sorted[pi]
                        except:
                            pass

                        self.p_record[c][a][t] = precision_101
                        num_points = len(precision_101)
                        # COCO's ap = mean of the 101 precision points, I use this way to keep the code compatibility,
                        # so the width of the stair is 1 / num_points. This can get the same AP. But recall is
                        # different. COCO's recall is the last value of all recall values, and mine is the last value
                        # of 101 recall values.
                        self.r_record[c][a][t] = np.array([1 / num_points] * num_points)
                        self.s_record[c][a][t] = score_101


    @staticmethod
    def mr4(array):
        return round(float(np.mean(array)), 4)
    
    def summarize(self):
        print('Sumarizing...')
        self.AP_matrix = np.zeros((self.CAT, self.AREA, self.THRE)) - 1
        self.AR_matrix = np.zeros((self.CAT, self.AREA, self.THRE)) - 1
        if self.all_points:
            self.MPP_matrix = np.zeros((self.CAT, self.AREA, self.THRE)) - 1

        for c in range(self.CAT):
            for a in range(self.AREA):
                for t in range(self.THRE):
                    if self.p_record[c][a][t] is not None:
                        # exclude absent categories. the related AP is -1
                        self.AP_matrix[c, a, t] = (self.p_record[c][a][t] * self.r_record[c][a][t]).sum()

                        # In all points mode, recall is always the sum of 'stair_w', but in 101 points mode,
                        # we need to find where precision reduce to 0., and thus calculate the recall.
                        if self.all_points:
                            self.AR_matrix[c, a, t] = self.r_record[c][a][t].sum()
                            r_cumsum = np.cumsum(self.r_record[c][a][t])
                            ap_array = self.p_record[c][a][t] * r_cumsum
                            index = np.argmax(ap_array)
                            p_max = self.p_record[c][a][t][index]
                            r_max = r_cumsum[index]
                            s_max = self.s_record[c][a][t][index]
                            mpp = ap_array[index] # max practical precision

                            # ff: a rate of current ap and  max value
                            #  If ap == 0 for a certain threshold, ff should be taken into calculation because
                            # it's not an absent category, so ff should be 0 instead of nan.
                            ff = 0. if self.AP_matrix[c, a, t] == 0 else mpp/ self.AP_matrix[c, a, t] 
                            self.MPP_matrix[c, a, t] = self.array([p_max, r_max, s_max, mpp, ff])

                        else:
                            r_mask = self.p_recordp[c][a][t] != 0
                            self.AR_matrix[c, a, t] = (self.r_record[c][a][t])[r_mask].sum()

        table_c_list = [['Category', 'AP', 'Recall'] * 3]
        c_line = ['all', self.mr4(self.AP_matrix[:, 0, :]), self.mr4(self.AR_matrix[:, 0, :])]

        if self.all_points:
            table_mpp_list = [['Category', 'P_max', 'R_max', 'Score', 'MPP', 'FF'] * 3]
            mpp_line = ['all', self.mr4(self.MPP_matrix[:, 0, :, 0]),
                        self.mr4(self.MPP_matrix[:, 0, :, 1]),
                        self.mr4(self.MPP_matrix[:, 0, :, 2]),
                        self.mr4(self.MPP_matrix[:, 0, :, 3]),
                        self.mr4(self.MPP_matrix[:, 0, :, 4])]
            
        for i in range(self.CAT):
            if -1 in self.AP_matrix[i, 0, :]:
                # the category is absent
                assert self.AP_matrix[i, 0, :].sum() == -len(self.iou_thre), 'Error: Not all ap is -1 in absent category.'
                c_line += [self.class_names[i], 'absent', 'absent']
                if self.all_points:
                    mpp_line += [self.class_names[i], 'absent', 'absent', 'absent', 'absent', 'absent']
            else:
                c_line += [self.class_names[i], self.mr4(self.AP_matrix[i, 0, :]), self.mr4(self.AR_matrix[i, 0, :])]
                if self.all_points:
                    mpp_line += [self.class_names[i],
                                 self.mr4(self.MPP_matrix[i, 0, :, 0]),
                                 self.mr4(self.MPP_matrix[i, 0, :, 1]),
                                 self.mr4(self.MPP_matrix[i, 0, :, 2]),
                                 self.mr4(self.MPP_matrix[i, 0, :, 3]),
                                 self.mr4(self.MPP_matrix[i, 0, :, 4])]
            if (i + 2) % 3 == 0:
                table_c_list.append(c_line)
                c_line = []

                if self.all_points:
                    table_mpp_list.append(mpp_line)
                    mpp_line = []
        
        if c_line:
            table_c_list.append(c_line)
        if self.all_points and mpp_line:
            table_mpp_list.append(mpp_line)

        table_iou_list = [['IoU'] + self.iou_thre, 'AP', 'Recall']
        for i in range(self.THRE):
            ap_m = self.AP_matrix[:, 0, i] # absent category is not included
            ar_m = self.AR_matrix[:, 0, i]
            table_iou_list[1].append(self.mr4(ap_m[ap_m > -1]))
            table_iou_list[2].append(self.mr4(ar_m[ar_m > -1]))

        table_area_list = [['Area'] + self.area_name, ['AP'], ['Recall']]
        for i in range(self.AREA):
            ap_m = self.AP_matrix[:, i, :]
            ar_m = self.AR_matrix[:, i, :]
            table_area_list[1].append(self.mr4(ap_m[ap_m > -1]))
            table_area_list[2].append(self.mr4(ar_m[ar_m > -1]))

        table_c = AsciiTable(table_c_list)
        table_iou = AsciiTable(table_iou_list)
        table_area = AsciiTable(table_area_list)

        if self.all_points:
            print()
            table_mpp = AsciiTable(table_mpp_list)
            print(table_mpp.table)

        print()
        print(table_c.table)
        print()
        print(table_iou.table)
        print()
        print(table_area.table)       
