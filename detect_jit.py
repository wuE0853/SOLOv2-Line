import torch
import cv2
import numpy as np
from utils.onnx_trans import base_input_trans
from model.head import SOLOv2Head
from configs import *
from scipy import ndimage

PALLETE = [
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

model_path = "export_files/Line_solov2_r50_light.pt"
model = torch.jit.load(model_path)
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

cfg = Line_solov2_r50_light(mode='onnx')
cfg.print_cfg()
head = SOLOv2Head(num_classes=cfg.num_classes, stacked_convs=cfg.head_stacked_convs,
                                        scale_ranges=cfg.head_scale_ranges, seg_feat_channels=cfg.head_seg_feat_c,
                                        ins_out_channels=cfg.head_ins_out_c)
head = head.to(device)


img_path = 'dataset/line_sep_d/train/244.jpg'
image = cv2.imread(img_path, cv2.IMREAD_COLOR)
resized_img = cv2.resize(image, cfg.onnx_shape)
input_img = torch.from_numpy(resized_img).cuda()

input_img = base_input_trans(input_img)

# inference
with torch.no_grad():
    mask_feat_pred, cate_preds, kernel_preds = model(input_img)

seg_result = head.get_seg(cate_preds, kernel_preds, mask_feat_pred,
                      None, cfg.onnx_shape, cfg.postprocess_para)[0]
print(type(seg_result))

# seg_result visualization
if seg_result is not None:
    seg_pred = seg_result[0].cpu().numpy()
    cate_label = seg_result[1].cpu().numpy()
    cate_score = seg_result[2].cpu().numpy()

    seg_show = resized_img.copy()
    for j in range(seg_pred.shape[0]):
        cur_mask = seg_pred[j, :, :]
        assert cur_mask.sum() != 0, 'cur_mask.sum() == 0.'

        cur_cate = cate_label[j] + 1  # this is the correct one
        cur_score = cate_score[j]

        color = PALLETE[j]
        if cfg.detect_mode == 'overlap':
            mask_bool = cur_mask.astype('bool')
            seg_show[mask_bool] = resized_img[mask_bool] * 0.5 + np.array(color, dtype='uint8') * 0.5
        elif cfg.detect_mode == 'contour':
            _, img_thre = cv2.threshold(cur_mask, 0, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(img_thre, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(seg_show, contours, contourIdx=-1, color=color, thickness=1)

        label_text = f'{cfg.class_names[cur_cate]} {cur_score:.02f}'
        center_y, center_x = ndimage.center_of_mass(cur_mask)
        vis_pos = (max(int(center_x) - 10, 0), int(center_y))
        cv2.putText(seg_show, label_text, vis_pos, cv2.FONT_HERSHEY_COMPLEX, 0.4, tuple(color))

    cv2.imwrite('results/detect/onnx/244.jpg', seg_show)
else:
    print('No mask detected.')
    cv2.imwrite('results/detect/onnx/244.jpg', input_img)