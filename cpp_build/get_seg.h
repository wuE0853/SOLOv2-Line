#ifndef SOLO_V2_SEGMENTATION_H
#define SOLO_V2_SEGMENTATION_H

#include <torch/script.h>
#include <vector>
#include <tuple>
#include <string>
#include <iostream>
#include <unordered_map>

namespace solo_v2 {

    struct SOLOv2Config {
        // 网络配置参数
        std::vector<int> num_grids = { 40, 36, 24, 16, 12 };
        std::vector<int> strides = { 8, 8, 16, 32, 32 };
        int max_num = 100;
        int kernel_out_ch = 128; // 默认是SOLOv2_r50_light的输出参数
        int cate_out_ch = 2;

        // 后处理参数
        float score_thr = 0.1f;
        float mask_thr = 0.5f;
        float update_thr = 0.05f;
        int nms_pre = 500;
    };

    struct SegmentationResult {
        torch::Tensor masks;      // 分割掩码 [N, H, W]
        torch::Tensor labels;     // 类别标签 [N]
        torch::Tensor scores;     // 置信度分数 [N]

        bool has_detection() const {
            return masks.defined() && masks.size(0) > 0;
        }
    };

    class MatrixNMS {
    public:
        static torch::Tensor compute(const torch::Tensor& seg_masks,
            const torch::Tensor& cate_labels,
            const torch::Tensor& cate_scores,
            const torch::Tensor& sum_masks,
            float kernel_sigma = 2.0f) {
            int n_samples = seg_masks.size(0);

            // 重塑seg_masks为(n, h*w)
            auto seg_masks_flat = seg_masks.reshape({ n_samples, -1 }).to(torch::kFloat32);

            // 计算交互矩阵 inter_matrix
            auto inter_matrix = torch::mm(seg_masks_flat, seg_masks_flat.transpose(1, 0));

            // 计算并集和IoU矩阵
            auto sum_masks_x = sum_masks.expand({ n_samples, n_samples });
            auto union_matrix = sum_masks_x + sum_masks_x.transpose(1, 0) - inter_matrix;
            auto iou_matrix = inter_matrix / union_matrix;

            // 只保留上三角部分（不包括对角线）
            iou_matrix = iou_matrix.triu(1);

            // 创建标签特定的矩阵 label_matrix
            auto cate_labels_x = cate_labels.expand({ n_samples, n_samples });
            auto label_matrix = (cate_labels_x == cate_labels_x.transpose(1, 0)).to(torch::kFloat32);
            label_matrix = label_matrix.triu(1);

            // IoU补偿：计算每列的最大IoU
            auto compensate_iou_max = (iou_matrix * label_matrix).max(0);
            auto compensate_iou = std::get<0>(compensate_iou_max);
            compensate_iou = compensate_iou.expand({ n_samples, n_samples }).transpose(1, 0);

            // IoU衰减
            auto decay_iou = iou_matrix * label_matrix;

            // 矩阵NMS，高斯核
            auto decay_matrix = torch::exp(-1.0f * kernel_sigma * (decay_iou * decay_iou));
            auto compensate_matrix = torch::exp(-1.0f * kernel_sigma * (compensate_iou * compensate_iou));

            // 计算衰减系数
            auto decay_coefficient_min = (decay_matrix / compensate_matrix).min(0);
            auto decay_coefficient = std::get<0>(decay_coefficient_min);

            // 更新分数
            auto cate_scores_update = cate_scores * decay_coefficient;

            return cate_scores_update;
        }
    };

    class SOLOv2Segmentation {
    private:
        std::vector<int> num_grids_;
        std::vector<int> strides_;
        int max_num_;
        int cate_out_ch_;
        int kernel_out_ch_;

    public:
        SOLOv2Segmentation(
            const std::vector<int>& num_grids,
            const std::vector<int>& strides,
            int max_num,
            int cate_out_ch,
            int kernel_out_ch
        ) :
             num_grids_(num_grids), strides_(strides),
             max_num_(max_num), cate_out_ch_(cate_out_ch),
             kernel_out_ch_(kernel_out_ch) {}

        std::vector<SegmentationResult> get_seg(
            const std::vector<torch::Tensor>& cate_preds,
            const std::vector<torch::Tensor>& kernel_preds,
            const torch::Tensor& seg_pred,
            const std::vector<int64_t>& ori_shape = {},
            const std::vector<int64_t>& resize_shape = {},
            const std::unordered_map<std::string, float>& cfg = {},
            const std::string& post_mode = "detect") {

            int batch_size = seg_pred.size(0);
            int num_levels = cate_preds.size();
            auto featmap_size = seg_pred.sizes().slice(2); //获取H, W

            std::vector<SegmentationResult> results;

            for (int j = 0; j < batch_size; ++j) {
                try {
                    // 提取当前batch的数据
                    std::vector<torch::Tensor> cate_pred_list;
                    std::vector<torch::Tensor> kernel_pred_list;

                    for (int i = 0; i < num_levels; ++i) {
                        auto cate_pred = cate_preds[i][j].reshape({ -1, cate_out_ch_ }).detach();
                        auto kernel_pred = kernel_preds[i][j].permute({ 1, 2, 0 })
                            .reshape({ -1, kernel_out_ch_ })
                            .detach();
                        cate_pred_list.push_back(cate_pred);
                        kernel_pred_list.push_back(kernel_pred);
                    }

                    auto seg_pred_single = seg_pred[j].unsqueeze(0);
                    auto cate_pred_merged = torch::cat(cate_pred_list, 0);
                    auto kernel_pred_merged = torch::cat(kernel_pred_list, 0);

                    // 调用get_seg_single
                    auto result = get_seg_single(cate_pred_merged, seg_pred_single, kernel_pred_merged,
                        featmap_size, resize_shape, ori_shape, cfg, post_mode);
                    results.push_back(result);

                }
                catch (const std::exception& e) {
                    std::cerr << "Error processing batch " << j << ": " << e.what() << std::endl;
                    results.push_back({ torch::Tensor(), torch::Tensor(), torch::Tensor() });
                }
            }

            return results;
        }

    private:
        SegmentationResult get_seg_single(
            const torch::Tensor& cate_preds,
            const torch::Tensor& seg_preds,
            const torch::Tensor& kernel_preds,
            const c10::IntArrayRef& featmap_size,
            const std::vector<int64_t>& resize_shape,
            const std::vector<int64_t>& ori_shape,
            const std::unordered_map<std::string, float>& cfg,
            const std::string& post_mode = "detect") {

            try {
                // 设置默认配置参数
                float score_thr = cfg.count("score_thr") ? cfg.at("score_thr") : 0.1f;
                float mask_thr = cfg.count("mask_thr") ? cfg.at("mask_thr") : 0.5f;
                float update_thr = cfg.count("update_thr") ? cfg.at("update_thr") : 0.05f;
                int nms_pre = cfg.count("nms_pre") ? static_cast<int>(cfg.at("nms_pre")) : 500;

                // process.
                auto inds = (cate_preds > score_thr);
                auto cate_scores = cate_preds.index({ inds });

                if (cate_scores.size(0) == 0) {
                    std::cout << "no cate score" << std::endl;
                    return { torch::Tensor(), torch::Tensor(), torch::Tensor() };
                }

                // cate_labels & kernel_preds
                std::cout << "Kernel_pred's size:" << kernel_preds.sizes() << std::endl;
                auto nonzero_inds = inds.nonzero();
                auto cate_labels = nonzero_inds.index({ torch::indexing::Slice(), 1 });
                auto kernel_preds_selected = kernel_preds.index({ nonzero_inds.index({torch::indexing::Slice(), 0}) });
                std::cout << "Kernl_pred_select's SHAPE:" << kernel_preds_selected.sizes() << std::endl;

                // trans vector.
                auto num_grids_tensor = torch::tensor(num_grids_, torch::kInt64).to(cate_labels.device());
                auto size_trans = torch::pow(num_grids_tensor, 2).cumsum(0);
                auto strides = torch::ones({ size_trans[-1].item<int64_t>() },
                    torch::dtype(torch::kFloat32).device(kernel_preds_selected.device()));

                int n_stage = num_grids_.size();
                strides.index({ torch::indexing::Slice(0, size_trans[0].item<int64_t>()) }) *= strides_[0];
                for (int i = 1; i < n_stage; ++i) {
                    int64_t start = size_trans[i - 1].item<int64_t>();
                    int64_t end = size_trans[i].item<int64_t>();
                    strides.index({ torch::indexing::Slice(start, end) }) *= strides_[i];
                }
                strides = strides.index({ nonzero_inds.index({torch::indexing::Slice(), 0}) });

                // mask encoding.
                int I = kernel_preds_selected.size(0);
                int N = kernel_preds_selected.size(1);
                auto kernel_preds_reshaped = kernel_preds_selected.view({ I, N, 1, 1 });

                auto seg_preds_conv = torch::conv2d(seg_preds, kernel_preds_reshaped, {}, 1);
                auto seg_preds_sigmoid = seg_preds_conv.squeeze(0).sigmoid();

                // mask.
                auto seg_masks = seg_preds_sigmoid > mask_thr;
                auto sum_masks = seg_masks.sum({ 1, 2 }).to(torch::kFloat32);

                // filter.
                auto keep = sum_masks > strides;

                seg_masks = seg_masks.index({ keep });
                auto seg_preds_filtered = seg_preds_sigmoid.index({ keep });
                sum_masks = sum_masks.index({ keep });
                cate_scores = cate_scores.index({ keep });
                cate_labels = cate_labels.index({ keep });

                // maskness.
                auto seg_scores = (seg_preds_filtered * seg_masks.to(torch::kFloat32)).sum({ 1, 2 }) / sum_masks;
                cate_scores = cate_scores * seg_scores;

                // sort and keep top nms_pre
                auto sort_inds = torch::argsort(cate_scores, -1, true);
                if (sort_inds.size(0) > nms_pre) {
                    sort_inds = sort_inds.index({ torch::indexing::Slice(0, nms_pre) });
                }

                seg_masks = seg_masks.index({ sort_inds });
                seg_preds_filtered = seg_preds_filtered.index({ sort_inds });
                sum_masks = sum_masks.index({ sort_inds });
                cate_scores = cate_scores.index({ sort_inds });
                cate_labels = cate_labels.index({ sort_inds });
                std::cout << "cate_scores shape:" << cate_scores.sizes() << std::endl;

                // Matrix NMS
                auto cate_scores_nms = MatrixNMS::compute(seg_masks, cate_labels, cate_scores, sum_masks);

                // filter.
                keep = cate_scores_nms >= update_thr;
                std::cout << "Keep count: " << keep.sum().item<int64_t>() << std::endl;

                if (keep.sum().item<int64_t>() == 0) {
                    std::cout << "no cate score larger than threshold" << std::endl;
                    return { torch::Tensor(), torch::Tensor(), torch::Tensor() };
                }

                seg_preds_filtered = seg_preds_filtered.index({ keep });
                cate_scores_nms = cate_scores_nms.index({ keep });
                cate_labels = cate_labels.index({ keep });

                // sort and keep top_k
                sort_inds = torch::argsort(cate_scores_nms, -1, true);

                if (post_mode == "val") {
                    if (sort_inds.size(0) > max_num_) {
                        sort_inds = sort_inds.index({ torch::indexing::Slice(0, max_num_) });
                    }
                }

                seg_preds_filtered = seg_preds_filtered.index({ sort_inds });
                cate_scores_nms = cate_scores_nms.index({ sort_inds });
                cate_labels = cate_labels.index({ sort_inds });

                // 上采样
                int64_t h = resize_shape[0], w = resize_shape[1];
                std::vector<int64_t> upsampled_size_out = {
                    static_cast<int64_t>(featmap_size[0] * 4),
                    static_cast<int64_t>(featmap_size[1] * 4)
                };

                auto seg_masks_upsampled = torch::nn::functional::interpolate(
                    seg_preds_filtered.unsqueeze(0),
                    torch::nn::functional::InterpolateFuncOptions()
                    .size(upsampled_size_out)
                    .mode(torch::kBilinear)
                ).index({ torch::indexing::Slice(),
                        torch::indexing::Slice(),
                        torch::indexing::Slice(0, h),
                        torch::indexing::Slice(0, w) });

                if (post_mode == "val") {
                    seg_masks_upsampled = torch::nn::functional::interpolate(
                        seg_masks_upsampled,
                        torch::nn::functional::InterpolateFuncOptions()
                        .size(ori_shape)
                        .mode(torch::kBilinear)
                    );
                }

                auto seg_masks_final = seg_masks_upsampled.squeeze(0);
                auto seg_masks_binary = (seg_masks_final > mask_thr).to(torch::kUInt8);

                // 后处理排序
                if (post_mode == "detect" || post_mode == "onnx") {
                    auto mask_density = seg_masks_binary.sum({ 1, 2 });
                    auto orders = torch::argsort(mask_density, -1, true);
                    seg_masks_binary = seg_masks_binary.index({ orders });
                    cate_labels = cate_labels.index({ orders });
                    cate_scores_nms = cate_scores_nms.index({ orders });
                }

                return { seg_masks_binary, cate_labels, cate_scores_nms };

            }
            catch (const std::exception& e) {
                std::cerr << "Error in get_seg_single: " << e.what() << std::endl;
                return { torch::Tensor(), torch::Tensor(), torch::Tensor() };
            }
        }

    };
}// namespace solo_v2

#endif // SOLO_V2_SEGMENTATION_H