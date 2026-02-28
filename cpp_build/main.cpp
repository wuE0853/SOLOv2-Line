#include <torch/script.h> // 一站式头文件
#include <torch/torch.h>
#include <torch/version.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

#include <Windows.h>
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include "get_seg.h"
#include "patch_generate.h"

using namespace cv;
using namespace std;

int main() {
    // 设置模块参数
    std::vector<string> class_names = { "_background_", "Line" };

    int patch_size = 512;
    int stride = 256;

    string detect_mode = "overlap"; // 检测模式，可选"overlap", "contour"
    std::vector<int> num_grids = { 40, 36, 24, 16, 12 };
    std::vector<int> strides = { 8, 8, 16, 32, 32 };
    int max_num = 100;
    int cate_out_ch = 2; //类别数
    int kernel_out_ch = 128;

    solo_v2::SOLOv2Segmentation segmenter(num_grids, strides, max_num, cate_out_ch, kernel_out_ch);

    std::unordered_map<std::string, float> cfg = {
        {"score_thr", 0.1f},
        {"mask_thr", 0.5f},
        {"update_thr", 0.3f},
        {"nms_pre", 500}
    };

    std::vector<int64_t> ori_shape = {};
    std::vector<int64_t> resize_shape = {512, 512};


    // 加载特定版本库
    const char* dll_path = R"(C:\libtorch-win-shared-with-deps-1.13.1+cu117\libtorch\lib)";
    if (SetDllDirectoryA(dll_path)) {
        std::cout << "DLL搜索路径已设置成功。" << std::endl;
    }
    else {
        DWORD error = GetLastError();
        std::cerr << "设置DLL搜索路径失败，错误代码: " << error << std::endl;
        return 1;
    }


    // 初始化设备类型，默认为CPU
    torch::DeviceType device_type = torch::kCPU;

    // 检查CUDA是否可用，如果可用优先使用GPU
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available and is using GPU for inference." << std::endl;
        device_type = torch::kCUDA;
    }
    else {
        std::cout << "Using CPU only." << std::endl;
    }
    torch::Device device(device_type);

    std::cout << "LibTorch版本: " << TORCH_VERSION << std::endl;

    // 加载TorchScript模型
    torch::jit::script::Module module;
    std::string model_path = "C:\\solov2.pt";

    try {
        module = torch::jit::load(model_path, device);
        std::cout << "模型加载成功！" << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "模型加载失败: " << e.what() << std::endl;
        return -1;
    }

    module.to(device_type);
    // 设置模型为评估模式（禁用dropout等训练层）
    module.eval();

    // 为提升性能，禁用梯度计算
    torch::NoGradGuard no_grad;

    // 准备输入数据：创建一个示例张量
    // auto input = torch::ones({ 1, 3, 512, 512 }).to(device);
    // std::vector<torch::jit::IValue> inputs;
    // inputs.push_back(input);

    //读取并预处理图片
    auto start = std::chrono::high_resolution_clock::now(); // 开始计时
    std::string image_path = R"(C:\demo_test0227_ori\34.jpg)";
    cv::Mat ori_image = cv::imread(image_path);
    if (ori_image.empty()) {
        std::cerr << "Error: Cannot read the target image. Please check the image_path" << std::endl;
        return -1;
    }

    PatchGenerator generator(ori_image, patch_size, stride, 0);
    std::vector<PatchInfo> patches = generator.getAllPatches();

    std::cout << "Generated " << patches.size() << " patches." << std::endl;

    for (auto& patchInfo : patches) {
        cv::Mat image = patchInfo.patch;
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        //cv::resize(image, image, cv::Size(512, 512));
        // image.convertTo(image, CV_32FC3, 1.0 / 255.0);



        // auto tensor = torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kFloat32);
        auto tensor = torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kByte); //先tensor后归一化
        tensor = tensor.to(torch::kFloat).div(255.0);
        tensor = tensor.permute({ 2, 0, 1 }).unsqueeze(0);
        tensor = tensor.to(device);

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(tensor);

        // 执行推理
        // 模型输出为元组，长度为3
        torch::jit::IValue out = module.forward(inputs);
        auto output_tuple = out.toTuple();
        auto outputs = output_tuple->elements();

        at::Tensor mask_feat_pred = outputs[0].toTensor();

        c10::List<c10::IValue> cate_preds = outputs[1].toList();
        // 遍历列表，逐个元素处理
        std::vector<at::Tensor> cate_list; // 用于存储转换后的张量
        for (size_t i = 0; i < cate_preds.size(); ++i) {
            c10::IValue element_ivalue = cate_preds.get(i);
            at::Tensor tensor_in_list = element_ivalue.toTensor();
            cate_list.push_back(tensor_in_list);
        }

        c10::List<c10::IValue> kernel_preds = outputs[2].toList();
        // 遍历列表，逐个元素处理
        std::vector<at::Tensor> kernel_list; // 用于存储转换后的张量
        for (size_t i = 0; i < kernel_preds.size(); ++i) {
            c10::IValue element_ivalue = kernel_preds.get(i);
            at::Tensor tensor_in_list = element_ivalue.toTensor();
            kernel_list.push_back(tensor_in_list);
            // 现在 tensor_list 中包含了从generic_list成功转换的所有张量
        }


        try {
            auto seg_result = segmenter.get_seg(
                cate_list, kernel_list, mask_feat_pred,
                ori_shape, resize_shape, cfg, "detect"
            )[0];
            if (seg_result.masks.defined()) {
                std::cout << "Image " << ": " << seg_result.masks.size(0) << " instances detected" << std::endl;
                auto seg_pred = seg_result.masks.cpu().to(torch::kFloat32);
                auto cate_label = seg_result.labels.cpu().to(torch::kByte);
                auto cate_score = seg_result.scores.cpu().to(torch::kFloat32);

                cv::Mat seg_show = image.clone(); // 创建原图拷贝
                cv::cvtColor(seg_show, seg_show, cv::COLOR_BGR2RGB);

                int num_masks = seg_pred.size(0);
                for (int j = 0; j < num_masks; ++j) {
                    auto cur_mask_tensor = seg_pred.index({ j, "..." });

                    auto mask_sum = cur_mask_tensor.sum().item<float>();
                    if (mask_sum == 0) {
                        std::cout << "mask_sum == 0" << std::endl; // 空掩码
                        continue;
                    }

                    auto cur_mask = cur_mask_tensor.squeeze().contiguous();
                    cv::Mat mask_mat(cur_mask.size(0), cur_mask.size(1), CV_32FC1, cur_mask.data_ptr<float>());

                    cv::Mat mask_uint8;
                    mask_mat.convertTo(mask_uint8, CV_8UC1, 255.0);

                    // 获取类别和分数
                    int cur_cate = cate_label.index({ j }).item<int>() + 1;
                    float cur_score = cate_score.index({ j }).item<float>();

                    // 调色板
                    cv::Scalar color(0, 255, 0);

                    // 根据检测模式进行处理
                    if (detect_mode == "overlap") {
                        // 创建bool掩码
                        cv::Mat mask_bool;
                        cv::threshold(mask_uint8, mask_bool, 128, 255, cv::THRESH_BINARY);
                        if (j == 0) {
                            patchInfo.mask = mask_bool;
                        }
                        else {
                            cv::bitwise_or(patchInfo.mask, mask_bool, patchInfo.mask);
                        }

                        // 应用重叠效果
                        cv::Mat overlay;
                        seg_show.copyTo(overlay);
                        overlay.setTo(color, mask_bool);
                        cv::addWeighted(seg_show, 0.5, overlay, 0.5, 0, seg_show);
                    }
                    else if (detect_mode == "contour") {
                        // 轮廓模式
                        cv::Mat img_thre;
                        cv::threshold(mask_uint8, img_thre, 128, 255, cv::THRESH_BINARY);

                        std::vector<std::vector<cv::Point>> contours;
                        cv::findContours(img_thre, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

                        // 绘制轮廓
                        cv::drawContours(seg_show, contours, -1, color, 1);
                    }

                    // 计算质心
                    cv::Moments m = cv::moments(mask_uint8, true);
                    if (m.m00 == 0) continue;

                    int center_x = static_cast<int>(m.m10 / m.m00);
                    int center_y = static_cast<int>(m.m01 / m.m00);
                    cv::Point vis_pos(std::max(center_x - 10, 0), center_y); // 确保标签文本位置有效

                    std::string label_text = class_names[cur_cate] + " " + std::to_string(cur_score).substr(0, 4);

                    // 添加文本
                    cv::putText(seg_show, label_text, vis_pos, cv::FONT_HERSHEY_COMPLEX, 0.4, color, 1);
                }
                // cv::imwrite("result.jpg", seg_show);
                patchInfo.patch = seg_show;
            }
            else {
                cv::cvtColor(patchInfo.patch, patchInfo.patch, cv::COLOR_BGR2RGB);
                std::cout << "Image " << ": No instances detected" << std::endl;
            }
        }
        catch (const c10::Error& e) {
            std::cerr << "后处理失败: " << e.what() << std::endl;
            return -1;
        }
    }

    // cv::Mat reconstructed = reconstructImageFromGenerator(patches, generator);
    cv::Mat reconstructed_img, reconstructed_mask;
    reconstructImageAndMask(patches, generator, reconstructed_img, reconstructed_mask);

    // 结果展示
    cv::imshow("result", reconstructed_mask);
    cv::waitKey(0);
    cv::destroyAllWindows();

    // 保存结果
    cv::imwrite("reconstructed.jpg", reconstructed_img);
    cv::imwrite("mask_re.jpg", reconstructed_mask);

    auto end = std::chrono::high_resolution_clock::now(); // 结束计时
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "耗时: " << duration.count() << " 毫秒" << std::endl;

    return 0;
}