import os
import cv2
import numpy as np
import json
from tqdm import tqdm
from typing import Dict, List, Tuple, Any


def patches_without_anns(
        input_dir: str,
        output_dir: str,
        patch_size: int = 512,
        stride: int = 512,
        fill_value: int = 0,
        **kwargs  ) -> None:
    """

    Args:
        input_dir: the input images direction
        output_dir: Patch output direction
        patch_size: Patch size(512, 1024 ...)
        stride: the step length
        fill_value: The value fill with padding, 0 is black, 25 is white
    """
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir)
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    for filename in tqdm(image_files, desc="Generating Patch"):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue

        H, W = img.shape[:2]
        base_name = os.path.splitext(filename)[0]

        # Calculating the padding
        pad_h = (patch_size - (H % patch_size)) % patch_size
        pad_w = (patch_size - (W % patch_size)) % patch_size

        # padding imags to ensure the image size can be divided by patch_size
        padded_img = cv2.copyMakeBorder(
            img,
            top=0, bottom=pad_h,
            left=0, right=pad_w,
            borderType=cv2.BORDER_CONSTANT,
            value=(fill_value, fill_value, fill_value)
        )

        # slide the window to crop
        new_H, new_W = padded_img.shape[:2]
        for y in range(0, new_H - patch_size + 1, stride):
            for x in range(0, new_W - patch_size + 1, stride):
                patch = padded_img[y:y + patch_size, x:x + patch_size]
                patch_name = f"{base_name}_{y}_{x}.jpg"
                cv2.imwrite(os.path.join(output_dir, patch_name), patch)


def patches_dataset(
        input_dir: str,
        anno_path: str,
        output_dir: str,
        patch_size: int = 1024,
        stride: int = 512,
        min_overlap_ratio: float = 0.95):
    """

    参数:
        input_dir: 原始图像目录
        anno_path: 原始COCO标注文件
        output_dir: 输出目录
        patch_size: Patch大小
        stride: 滑动步长
        min_overlap_ratio: target min overlap ratio(0 - 1)
    """
    os.makedirs(output_dir, exist_ok=True)
    patches_images_dir = os.path.join(output_dir, "images")
    os.makedirs(patches_images_dir, exist_ok=True)

    # load the coco annos
    with open(anno_path, 'r') as f:
        original_anno: Dict[str, Any] = json.load(f)

    # 初始化新标注
    new_anno = {
        "info": original_anno.get("info", {}) | {
            "description": f"Patches with complete targets (min_overlap={min_overlap_ratio})"
        },
        "licenses": original_anno.get("licenses", []),
        "categories": original_anno["categories"],
        "images": [],
        "annotations": []
    }

    # 构建辅助数据结构
    original_images: Dict[int, Dict] = {img["id"]: img for img in original_anno["images"]}
    img_to_anns: Dict[int, List[Dict]] = {}
    for ann in original_anno["annotations"]:
        img_to_anns.setdefault(ann["image_id"], []).append(ann)

    # ID计数器
    new_image_id = 0
    new_anno_id = 0
    skipped_patches = 0  # 记录跳过的空patch
    incomplete_targets = 0  # 记录跳过的不完整目标

    # 进度条
    progress_bar = tqdm(total=len(original_images), desc="Processing images")

    # 处理每张原始图像
    for img_id, img_info in original_images.items():
        img_path = os.path.join(input_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}, skipping")
            continue

        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}, skipping")
            continue

        img_height, img_width = img.shape[:2]
        if img_height < patch_size or img_width < patch_size:
            print(f"Image {img_info['file_name']} smaller than patch size, skipping")
            continue

        # 获取当前图像的标注
        annotations = img_to_anns.get(img_id, [])

        # 遍历切割位置
        for y in range(0, img_height - patch_size + 1, stride):
            for x in range(0, img_width - patch_size + 1, stride):
                # 截取图像Patch
                patch = img[y:y + patch_size, x:x + patch_size]
                if patch.size == 0:
                    continue

                # 检查当前patch是否包含完整目标
                has_complete_target = False
                patch_annotations = []  # 存储当前patch的有效标注

                for ann in annotations:
                    # 跳过crowd标注
                    if ann.get("iscrowd", 0) == 1:
                        continue

                    # 计算边界框与Patch的交集
                    bbox = ann["bbox"]  # [x, y, w, h]
                    bbox_x1, bbox_y1 = bbox[0], bbox[1]
                    bbox_x2, bbox_y2 = bbox[0] + bbox[2], bbox[1] + bbox[3]

                    # 计算交集区域
                    inter_x1 = max(bbox_x1, x)
                    inter_y1 = max(bbox_y1, y)
                    inter_x2 = min(bbox_x2, x + patch_size)
                    inter_y2 = min(bbox_y2, y + patch_size)

                    inter_w = inter_x2 - inter_x1
                    inter_h = inter_y2 - inter_y1

                    # 跳过无交集的标注
                    if inter_w <= 0 or inter_h <= 0:
                        continue

                    # 计算目标在Patch中的覆盖率
                    inter_area = inter_w * inter_h
                    original_area = bbox[2] * bbox[3]
                    coverage_ratio = inter_area / original_area

                    # 检查目标是否完整（覆盖率 > 阈值）
                    if coverage_ratio < min_overlap_ratio:
                        incomplete_targets += 1
                        continue

                    # 标记发现有效目标
                    has_complete_target = True

                    # 计算新边界框（Patch坐标系）
                    new_bbox = [inter_x1 - x, inter_y1 - y, inter_w, inter_h]

                    # 调整分割标注坐标（确保完整目标）
                    new_segmentation = []
                    if "segmentation" in ann and ann["segmentation"]:
                        for segment in ann["segmentation"]:
                            # 将每个点转换到Patch坐标系
                            adjusted_segment = []
                            for i in range(0, len(segment), 2):
                                px = segment[i] - x
                                py = segment[i + 1] - y
                                # 检查点是否在patch内
                                if (0 <= px <= patch_size and 0 <= py <= patch_size):
                                    adjusted_segment.extend([px, py])
                            new_segmentation.append(adjusted_segment)

                    # 创建新标注
                    new_ann = {
                        "id": new_anno_id,
                        "image_id": new_image_id,
                        "category_id": ann["category_id"],
                        "bbox": new_bbox,
                        "area": original_area,  # 使用原始面积（完整目标）
                        "iscrowd": 0,
                        "segmentation": new_segmentation,
                        "coverage_ratio": coverage_ratio  # 记录覆盖率
                    }

                    # 复制额外字段
                    for key in ann.keys():
                        if key not in new_ann:
                            new_ann[key] = ann[key]

                    patch_annotations.append(new_ann)
                    new_anno_id += 1

                # 无完整目标则跳过当前patch
                if not has_complete_target:
                    skipped_patches += 1
                    continue

                # 保存Patch图像
                original_name = os.path.splitext(img_info["file_name"])[0]
                patch_filename = f"{new_image_id}.jpg"
                patch_path = os.path.join(patches_images_dir, patch_filename)
                cv2.imwrite(patch_path, patch)

                # 添加新图像信息
                new_img_info = {
                    "id": new_image_id,
                    "file_name": patch_filename,
                    "width": patch_size,
                    "height": patch_size,
                    "original_image_id": img_id,
                    "original_x": x,
                    "original_y": y
                }
                new_anno["images"].append(new_img_info)

                # 添加当前patch的所有标注
                new_anno["annotations"].extend(patch_annotations)
                new_image_id += 1

        progress_bar.update(1)

    progress_bar.close()

    # 保存新标注
    new_anno_path = os.path.join(output_dir, "annotations.json")
    with open(new_anno_path, "w") as f:
        json.dump(new_anno, f, indent=2)

    print(f"Complete! Generate {len(new_anno['images'])} Patches")
    print(f"Generate {len(new_anno['annotations'])} annotations")
    print(f"Skip {skipped_patches} Patches without complete target")
    print(f"Skip {incomplete_targets} incomplete targets")
    print(f"imags saved in: {patches_images_dir}")
    print(f"annotations saved in: {new_anno_path}")


if __name__ == "__main__":
    config = {
        "input_dir": "/home/amax/Public/SOLOv2_minimal/dataset/1215_3k",
        "anno_path": "" ,
        "output_dir": "/home/amax/Public/SOLOv2_minimal/dataset/1215_1k",
        "dataset_dir": "",

        "patch_size": 1024,
        "stride": 512,
        "min_overlap_ratio": 0.95,
        "mode": "detect"  
    }
    mode = config.get("mode", "detect")
    if config["mode"] == "train":
        patches_dataset(**config)
    elif config["mode"] == "detect":
        patches_without_anns(**config)
    else:
        raise ValueError('mode is invalid, which must be "train" or "detect')