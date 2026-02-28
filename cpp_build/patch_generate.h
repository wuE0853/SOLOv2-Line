#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>

// 定义一个结构体来存储patch及其位置信息和对应的掩膜
struct PatchInfo {
    cv::Mat patch;      // patch图像
    cv::Mat mask;       // patch对应的二值分割掩膜（可选）
    int x;              // 在原图中的x坐标（左上角）
    int y;              // 在原图中的y坐标（左上角）
    int patch_size;     // patch尺寸
};

class PatchGenerator {
private:
    cv::Mat image_;
    cv::Mat padded_img_;
    int patch_size_;
    int stride_;
    int fill_value_;
    int pad_h_;
    int pad_w_;

public:
    // 构造函数
    PatchGenerator(const cv::Mat& image, int patch_size = 512, int stride = 512, int fill_value = 0)
        : image_(image.clone()), patch_size_(patch_size), stride_(stride), fill_value_(fill_value) {

        if (image_.empty()) {
            return;
        }

        // 计算padding
        int H = image_.rows;
        int W = image_.cols;
        pad_h_ = (patch_size_ - (H % patch_size_)) % patch_size_;
        pad_w_ = (patch_size_ - (W % patch_size_)) % patch_size_;

        // 对图像进行padding
        cv::copyMakeBorder(image_, padded_img_,
                          0, pad_h_, 0, pad_w_,
                          cv::BORDER_CONSTANT,
                          cv::Scalar(fill_value_, fill_value_, fill_value_));
    }

    // 获取所有patch及其位置信息
    std::vector<PatchInfo> getAllPatches() {
        std::vector<PatchInfo> patchInfos;

        if (padded_img_.empty()) {
            return patchInfos;
        }

        int new_H = padded_img_.rows;
        int new_W = padded_img_.cols;

        // 滑动窗口裁剪patch
        for (int y = 0; y <= new_H - patch_size_; y += stride_) {
            for (int x = 0; x <= new_W - patch_size_; x += stride_) {
                cv::Rect roi(x, y, patch_size_, patch_size_);
                cv::Mat patch = padded_img_(roi);

                PatchInfo info;
                info.patch = patch.clone();
                // mask初始化为空，由后续处理填充
                info.mask = cv::Mat();
                info.x = x;
                info.y = y;
                info.patch_size = patch_size_;
                patchInfos.push_back(info);
            }
        }

        return patchInfos;
    }

    // 获取padding信息（用于后续重建）
    std::pair<int, int> getPaddingInfo() const {
        return std::make_pair(pad_h_, pad_w_);
    }

    // 获取原始图像尺寸（不包括padding）
    std::pair<int, int> getOriginalSize() const {
        return std::make_pair(image_.rows, image_.cols);
    }

    // 获取padded图像尺寸（包括padding）
    std::pair<int, int> getPaddedSize() const {
        return std::make_pair(padded_img_.rows, padded_img_.cols);
    }

    // 获取stride
    int getStride() const {
        return stride_;
    }

    // 获取patch_size
    int getPatchSize() const {
        return patch_size_;
    }
};

// 辅助函数：处理单张图像并返回所有patch的信息
std::vector<PatchInfo> getAllPatches(
    const cv::Mat& image,
    int patch_size = 512,
    int stride = 512,
    int fill_value = 0
) {
    PatchGenerator generator(image, patch_size, stride, fill_value);
    return generator.getAllPatches();
}

// 重建函数：将处理后的patch重新组合成原图，考虑重叠区域的叠加平均值
cv::Mat reconstructImage(
    const std::vector<PatchInfo>& processedPatches,
    const std::pair<int, int>& originalSize,
    const std::pair<int, int>& paddingInfo,
    int channels = 3
) {
    int original_h = originalSize.first;
    int original_w = originalSize.second;
    int pad_h = paddingInfo.first;
    int pad_w = paddingInfo.second;

    // 创建带padding的累加图像和计数图像
    cv::Mat accumulated = cv::Mat::zeros(original_h + pad_h, original_w + pad_w, CV_32FC(channels));
    cv::Mat count = cv::Mat::zeros(original_h + pad_h, original_w + pad_w, CV_32FC1);

    // 检查patch是否为空
    if (processedPatches.empty()) {
        std::cerr << "Warning: No patches provided for reconstruction!" << std::endl;
        // 将累加图像转换为8UC3并裁剪padding
        cv::Mat result;
        accumulated.convertTo(result, CV_8UC3);
        if (pad_h > 0 || pad_w > 0) {
            result = result(cv::Rect(0, 0, original_w, original_h));
        }
        return result;
    }

    // 获取第一个patch的信息以确定patch_size
    int patch_size = processedPatches[0].patch_size;

    // 将每个patch放回原位，并累加到accumulated图像
    for (const auto& patchInfo : processedPatches) {
        // 验证patch尺寸是否匹配
        if (patchInfo.patch.rows != patch_size ||
            patchInfo.patch.cols != patch_size) {
            std::cerr << "Warning: Patch size mismatch! Expected "
                      << patch_size << "x" << patch_size
                      << ", got " << patchInfo.patch.rows << "x"
                      << patchInfo.patch.cols << std::endl;
            continue;
        }

        // 计算patch在重建图像中的位置
        int x = patchInfo.x;
        int y = patchInfo.y;

        // 验证坐标是否在图像范围内
        if (x >= 0 && y >= 0 &&
            x + patch_size <= accumulated.cols &&
            y + patch_size <= accumulated.rows) {

            // 将patch转换为浮点型
            cv::Mat patch_float;
            patchInfo.patch.convertTo(patch_float, CV_32FC(channels));

            // 累加patch到accumulated图像
            cv::Mat roi_accumulated = accumulated(cv::Rect(x, y, patch_size, patch_size));
            roi_accumulated += patch_float;

            // 更新计数图像
            cv::Mat roi_count = count(cv::Rect(x, y, patch_size, patch_size));
            roi_count += 1.0;
        } else {
            std::cerr << "Warning: Patch at (" << x << ", " << y
                      << ") is out of bounds!" << std::endl;
        }
    }

    // 计算平均值（避免除以0）
    cv::Mat averaged = cv::Mat::zeros(accumulated.size(), accumulated.type());
    for (int i = 0; i < accumulated.rows; i++) {
        for (int j = 0; j < accumulated.cols; j++) {
            float cnt = count.at<float>(i, j);
            if (cnt > 0.0) {
                if (channels == 3) {
                    averaged.at<cv::Vec3f>(i, j) = accumulated.at<cv::Vec3f>(i, j) / cnt;
                } else if (channels == 1) {
                    averaged.at<float>(i, j) = accumulated.at<float>(i, j) / cnt;
                }
            }
        }
    }

    // 将浮点图像转换回8UC3
    cv::Mat result;
    averaged.convertTo(result, CV_8UC3);

    // 去掉padding，恢复原始尺寸
    if (pad_h > 0 || pad_w > 0) {
        result = result(cv::Rect(0, 0, original_w, original_h));
    }

    return result;
}

// 重建分割掩膜函数：将patch的分割掩膜重新组合成原图
cv::Mat reconstructMask(
    const std::vector<PatchInfo>& processedPatches,
    const std::pair<int, int>& originalSize,
    const std::pair<int, int>& paddingInfo,
    int mask_channels = 1
) {
    int original_h = originalSize.first;
    int original_w = originalSize.second;
    int pad_h = paddingInfo.first;
    int pad_w = paddingInfo.second;

    // 检查是否有掩膜
    bool has_mask = false;
    for (const auto& patchInfo : processedPatches) {
        if (!patchInfo.mask.empty()) {
            has_mask = true;
            break;
        }
    }

    if (!has_mask) {
        std::cerr << "Warning: No masks found in patches!" << std::endl;
        return cv::Mat();  // 返回空掩膜
    }

    // 创建结果掩膜，初始值为0
    cv::Mat result_mask = cv::Mat::zeros(original_h + pad_h, original_w + pad_w, CV_8UC1);

    // 获取第一个patch的信息以确定patch_size
    int patch_size = processedPatches[0].patch_size;

    // 将每个掩膜放回原位，使用逻辑或合并
    for (const auto& patchInfo : processedPatches) {
        if (patchInfo.mask.empty()) {
            continue;  // 跳过没有掩膜的patch
        }

        // 验证掩膜尺寸是否匹配
        if (patchInfo.mask.rows != patch_size ||
            patchInfo.mask.cols != patch_size) {
            std::cerr << "Warning: Mask size mismatch! Expected "
                      << patch_size << "x" << patch_size
                      << ", got " << patchInfo.mask.rows << "x"
                      << patchInfo.mask.cols << std::endl;
            continue;
        }

        // 计算掩膜在重建图像中的位置
        int x = patchInfo.x;
        int y = patchInfo.y;

        // 验证坐标是否在图像范围内
        if (x >= 0 && y >= 0 &&
            x + patch_size <= result_mask.cols &&
            y + patch_size <= result_mask.rows) {

            // 将掩膜转换为浮点型
            cv::Mat roi_result = result_mask(cv::Rect(x, y, patch_size, patch_size));

            // 对掩膜进行二值化处理（确保是0或255）
            cv::Mat binary_mask;
            if (patchInfo.mask.channels() > 1) {
                // 如果是多通道掩膜，转换为单通道灰度
                cv::cvtColor(patchInfo.mask, binary_mask, cv::COLOR_BGR2GRAY);
            }
            else {
                binary_mask = patchInfo.mask;
            }

            // 将掩膜二值化为0和255
            cv::Mat binary_mask_255;
            cv::threshold(binary_mask, binary_mask_255, 0, 255, cv::THRESH_BINARY);

            // 使用逻辑或运算合并掩膜
            cv::bitwise_or(roi_result, binary_mask_255, roi_result);
        } else {
            std::cerr << "Warning: Mask at (" << x << ", " << y
                      << ") is out of bounds!" << std::endl;
        }
    }

    // 去掉padding，恢复原始尺寸
    if (pad_h > 0 || pad_w > 0) {
        result_mask = result_mask(cv::Rect(0, 0, original_w, original_h));
    }

    return result_mask;
}

// 同时重建图像和掩膜
void reconstructImageAndMask(
    const std::vector<PatchInfo>& processedPatches,
    const PatchGenerator& generator,
    cv::Mat& reconstructed_img,
    cv::Mat& reconstructed_mask
) {
    auto originalSize = generator.getOriginalSize();
    auto paddingInfo = generator.getPaddingInfo();

    // 判断图像和掩膜的通道数
    int img_channels = 3; // 默认为3通道
    int mask_channels = 1; // 掩膜默认为单通道

    if (!processedPatches.empty()) {
        if (processedPatches[0].patch.channels() == 1) {
            img_channels = 1;
        }

        // 检查掩膜通道数
        bool has_mask = false;
        for (const auto& patchInfo : processedPatches) {
            if (!patchInfo.mask.empty()) {
                mask_channels = patchInfo.mask.channels();
                has_mask = true;
                break;
            }
        }

        if (!has_mask) {
            mask_channels = 1; // 如果没有掩膜，使用默认值
        }
    }

    reconstructed_img = reconstructImage(processedPatches, originalSize, paddingInfo, img_channels);
    reconstructed_mask = reconstructMask(processedPatches, originalSize, paddingInfo, mask_channels);
}

// 简化版重建函数（需要原始PatchGenerator对象）
cv::Mat reconstructImageFromGenerator(
    const std::vector<PatchInfo>& processedPatches,
    const PatchGenerator& generator
) {
    auto originalSize = generator.getOriginalSize();
    auto paddingInfo = generator.getPaddingInfo();

    // 判断图像通道数
    int channels = 3; // 默认为3通道
    if (!processedPatches.empty() && processedPatches[0].patch.channels() == 1) {
        channels = 1;
    }

    return reconstructImage(processedPatches, originalSize, paddingInfo, channels);
}