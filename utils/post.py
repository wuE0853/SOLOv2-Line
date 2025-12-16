import os
import re
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict


def reconstruct_from_patches(
        patch_dir: str,
        output_dir: str,
        original_size: tuple = (3224, 3224),
        patch_size: int = 512,
        filename_pattern: str = r"(.+?)_(\d+)_(\d+)\.(jpg|png|jpeg)",  # 匹配原始名_y_x.后缀
        blend_method: str = "average"  # 新增融合方法选项
) -> None:
    """

    参数:
        patch_dir: Patch图像目录
        output_dir: 复原图像输出目录
        original_size: 原始图像尺寸 (宽, 高)
        patch_size: Patch尺寸
        filename_pattern: 文件名解析正则表达式
        blend_method: 重叠区域融合方法 ("average"平均值/"max"最大值/"weighted"距离加权)
    """
    os.makedirs(output_dir, exist_ok=True)
    patch_files = [f for f in os.listdir(patch_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    # 按原始图像名分组Patch
    image_groups = defaultdict(list)
    for filename in patch_files:
        match = re.match(filename_pattern, filename, re.IGNORECASE)
        if match:
            base_name, y, x, ext = match.groups()
            image_groups[base_name].append((filename, int(y), int(x)))

    if not image_groups:
        print("Cannot find Patch files，please check the file names")
        return

    print(f"Recognize {len(image_groups)}  original Patches")
    print(f"The blend method is {blend_method}")

    # 处理每组Patch
    for base_name, patches in tqdm(image_groups.items(), desc="Reconstruct images"):
        # 创建三个画布：累积值/权重/计数器
        accumulator = np.zeros((original_size[1], original_size[0], 3), dtype=np.float32)
        weight_map = np.zeros((original_size[1], original_size[0], 3), dtype=np.float32)
        counter = np.zeros((original_size[1], original_size[0]), dtype=np.uint16)
        patch_count = len(patches)

        # 按坐标排序
        patches.sort(key=lambda x: (x[1], x[2]))

        # 处理每个Patch
        for filename, y, x in patches:
            patch_path = os.path.join(patch_dir, filename)
            patch_img = cv2.imread(patch_path)
            if patch_img is None:
                continue

            h, w = patch_img.shape[:2]
            end_y = min(y + h, original_size[1])
            end_x = min(x + w, original_size[0])

            # 计算有效区域
            patch_area = patch_img[:end_y - y, :end_x - x]
            canvas_area = accumulator[y:end_y, x:end_x]
            weight_area = weight_map[y:end_y, x:end_x]
            counter_area = counter[y:end_y, x:end_x]

            # 更新计数器
            counter[y:end_y, x:end_x] += 1

            # 根据融合方法计算权重
            if blend_method == "average":
                # 平均融合：直接累加像素值
                accumulator[y:end_y, x:end_x] += patch_area
            elif blend_method == "max":
                # 最大值融合：取重叠区域的最大值
                mask = patch_area > canvas_area
                accumulator[y:end_y, x:end_x][mask] = patch_area[mask]
            elif blend_method == "weighted":
                # 距离加权融合：中心区域权重更高
                weight_matrix = np.zeros((h, w, 1), dtype=np.float32)

                # 生成中心加权的权重图（中心为1，边缘为0.3）
                for i in range(h):
                    for j in range(w):
                        # 计算到中心的距离
                        dist_x = abs(j - w / 2) / (w / 2)
                        dist_y = abs(i - h / 2) / (h / 2)
                        dist = np.sqrt(dist_x ** 2 + dist_y ** 2) / np.sqrt(2)
                        weight = 1.0 - 0.7 * dist  # 中心权重1.0，边缘0.3
                        weight_matrix[i, j] = max(weight, 0.3)

                # 应用权重
                weighted_patch = patch_area.astype(np.float32) * weight_matrix
                accumulator[y:end_y, x:end_x] += weighted_patch[:end_y - y, :end_x - x]
                weight_map[y:end_y, x:end_x] += weight_matrix[:end_y - y, :end_x - x]

        # 最终融合处理
        if blend_method == "average":
            # 计算平均值（避免除零）
            canvas = np.divide(accumulator, counter[..., np.newaxis],
                               where=counter[..., np.newaxis] > 0)
        elif blend_method == "max":
            canvas = accumulator
        elif blend_method == "weighted":
            # 加权平均
            canvas = np.divide(accumulator, weight_map,
                               where=weight_map > 0)

        # 转换为uint8并处理边界
        canvas = np.rint(canvas).astype(np.uint8)

        # 保存复原图像
        output_path = os.path.join(output_dir, f"{base_name}_reconstructed.jpg")
        cv2.imwrite(output_path, canvas)
        print(f"Saved: {os.path.basename(output_path)} (with {patch_count} Patches)")
        print(f"max overlaping: {counter.max()} layers")


if __name__ == "__main__":
    # 配置参数
    config = {
        "patch_dir": "E:\GRADUATE2025\\result1120",  # Patch目录
        "output_dir": "E:\GRADUATE2025\\reconstruct1120",  # 输出目录
        "original_size": (3224, 3224),  # 原始图像尺寸
        "patch_size": 512,  # Patch尺寸
        "filename_pattern": r"(.+?)_(\d+)_(\d+)\.(jpg|png)",  # 根据实际文件名调整
        "blend_method": "average"  # 推荐使用加权融合,加权有bug，后面修
    }

    # 执行复原
    reconstruct_from_patches(**config)