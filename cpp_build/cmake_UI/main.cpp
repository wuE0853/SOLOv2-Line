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

#include <commdlg.h>

using namespace cv;
using namespace std;

// 定义全局变量用于界面交互
cv::Mat g_original_image;
std::vector<PatchInfo> g_patches;
PatchGenerator* g_generator = nullptr;
cv::Mat g_reconstructed_img;
cv::Mat g_reconstructed_mask;
bool g_image_loaded = false;
bool g_processing_done = false;
bool g_result_saved = false;
int g_patch_size = 512;
int g_stride = 256;
int g_fill_value = 0;
std::string g_save_path = "C:/Inference_UI_results/";

// 界面相关全局变量
cv::Mat g_main_window;          // 主窗口图像
cv::Point g_mouse_pos(0, 0);    // 鼠标位置
bool g_update_ui = true;        // 是否需要更新UI

// 窗口尺寸
const int WINDOW_WIDTH = 1200;
const int WINDOW_HEIGHT = 800;
const int CONTROL_WIDTH = 400;   // 控制面板宽度
const int IMAGE_WIDTH = 800;     // 图像显示区域宽度

// 按钮区域定义
struct Button {
    cv::Rect rect;           // 按钮区域
    std::string text;        // 按钮文本
    bool pressed;           // 按钮是否被按下
    int id;                 // 按钮ID
    bool enabled;           // 按钮是否可用

    Button(int x, int y, int w, int h, const std::string& text, int id, bool enabled = true)
        : rect(x, y, w, h), text(text), pressed(false), id(id), enabled(enabled) {}

    bool contains(const cv::Point& pt) const {
        return rect.contains(pt);
    }
};

// 参数调节区域定义
struct ParameterControl {
    cv::Rect rect;           // 参数显示区域
    std::string label;       // 参数标签
    int* value;             // 参数值指针
    int min_value;          // 最小值
    int max_value;          // 最大值
    Button btn_minus;       // 减少按钮
    Button btn_plus;        // 增加按钮

    ParameterControl(int x, int y, int w, int h, const std::string& label,
        int* value, int min_val, int max_val, int base_id)
        : rect(x, y, w, h), label(label), value(value),
        min_value(min_val), max_value(max_val),
        btn_minus(x + 10, y + 30, 40, 30, "-", base_id),
        btn_plus(x + w - 50, y + 30, 40, 30, "+", base_id + 1) {}
};


// 按钮和滑块集合
std::vector<Button> g_buttons;
std::vector<ParameterControl> g_params;

// 按钮ID定义
enum ButtonID {
    BTN_IMPORT = 1,
    BTN_EXECUTE,
    BTN_SAVE,
    BTN_EXIT,
    BTN_PATCH_MINUS = 10,
    BTN_PATCH_PLUS,
    BTN_STRIDE_MINUS,
    BTN_STRIDE_PLUS
};

// 绘制圆形按钮
void drawButton(cv::Mat& img, Button& button, bool highlight = false) {
    cv::Scalar bg_color;
    cv::Scalar text_color;

    if (!button.enabled) {
        // 禁用状态
        bg_color = cv::Scalar(40, 40, 40);
        text_color = cv::Scalar(100, 100, 100);
    }
    else if (button.pressed) {
        // 按下状态
        bg_color = cv::Scalar(100, 100, 200);
        text_color = cv::Scalar(255, 255, 255);
    }
    else if (highlight) {
        // 悬停状态
        bg_color = cv::Scalar(80, 180, 80);
        text_color = cv::Scalar(255, 255, 255);
    }
    else {
        // 正常状态
        bg_color = cv::Scalar(60, 60, 60);
        text_color = cv::Scalar(255, 255, 255);
    }

    // 绘制圆角矩形背景
    cv::rectangle(img, button.rect, bg_color, -1, cv::LINE_AA);

    // 添加边框
    cv::rectangle(img, button.rect, cv::Scalar(200, 200, 200), 2, cv::LINE_AA);

    // 计算文本位置
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(button.text, cv::FONT_HERSHEY_SIMPLEX,
        0.8, 2, &baseline);
    cv::Point text_pos(
        button.rect.x + (button.rect.width - text_size.width) / 2,
        button.rect.y + (button.rect.height + text_size.height) / 2
    );

    // 绘制文本
    cv::putText(img, button.text, text_pos, cv::FONT_HERSHEY_SIMPLEX,
        0.8, text_color, 2, cv::LINE_AA);
}

// 绘制参数调节区域
void drawParameterControl(cv::Mat& img, ParameterControl& param) {
    // 绘制背景
    cv::rectangle(img, param.rect, cv::Scalar(40, 40, 40), -1, cv::LINE_AA);
    cv::rectangle(img, param.rect, cv::Scalar(100, 100, 100), 1, cv::LINE_AA);

    // 绘制标签
    cv::putText(img, param.label, cv::Point(param.rect.x + 10, param.rect.y + 20),
        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);

    // 绘制当前值
    std::string value_str = std::to_string(*(param.value));
    int baseline = 0;
    cv::Size value_size = cv::getTextSize(value_str, cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, &baseline);
    cv::Point value_pos(
        param.rect.x + (param.rect.width - value_size.width) / 2,
        param.rect.y + 55
    );
    cv::putText(img, value_str, value_pos, cv::FONT_HERSHEY_SIMPLEX,
        0.8, cv::Scalar(255, 255, 100), 2, cv::LINE_AA);

    // 绘制按钮
    drawButton(img, param.btn_minus);
    drawButton(img, param.btn_plus);
}

// 创建控制面板区域（左侧）
void createControlPanel(cv::Mat& panel_img) {
    panel_img = cv::Mat::zeros(WINDOW_HEIGHT, CONTROL_WIDTH, CV_8UC3);

    // 绘制标题
    cv::putText(panel_img, "Line Segmentation", cv::Point(30, 40),
        cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 255, 255), 3, cv::LINE_AA);

    // 绘制分隔线
    cv::line(panel_img, cv::Point(20, 60), cv::Point(CONTROL_WIDTH - 20, 60),
        cv::Scalar(100, 100, 100), 2, cv::LINE_AA);

    // 创建按钮
    g_buttons.clear();
    g_buttons.push_back(Button(50, 80, 300, 50, "Import Image", BTN_IMPORT));
    g_buttons.push_back(Button(50, 150, 300, 50, "Execute Processing", BTN_EXECUTE));
    g_buttons.push_back(Button(50, 220, 300, 50, "Save Results", BTN_SAVE));

    // 创建参数调节区域
    g_params.clear();
    g_params.push_back(ParameterControl(50, 300, 300, 80, "Patch Size:",
        &g_patch_size, 64, 1024, BTN_PATCH_MINUS));
    g_params.push_back(ParameterControl(50, 400, 300, 80, "Stride:",
        &g_stride, 64, 512, BTN_STRIDE_MINUS));

    // 绘制状态信息区域
    cv::rectangle(panel_img, cv::Rect(30, 420, 340, 180),
        cv::Scalar(40, 40, 40), -1, cv::LINE_AA);
    cv::rectangle(panel_img, cv::Rect(30, 420, 340, 180),
        cv::Scalar(100, 100, 100), 2, cv::LINE_AA);

    cv::putText(panel_img, "Status Information:", cv::Point(40, 530),
        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(200, 200, 255), 1, cv::LINE_AA);

    cv::putText(panel_img, "Instructions:", cv::Point(40, 680),
        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(200, 255, 200), 1, cv::LINE_AA);

    cv::putText(panel_img, "1. Import image", cv::Point(50, 700),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
    cv::putText(panel_img, "2. Adjust parameters", cv::Point(50, 720),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
    cv::putText(panel_img, "3. Execute processing", cv::Point(50, 740),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
    cv::putText(panel_img, "4. View & save results", cv::Point(50, 760),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
    cv::putText(panel_img, "CAUTIONS: Patch size must be 512 now!!", cv::Point(50, 780),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 200), 1, cv::LINE_AA);
}

// 更新控制面板显示
void updateControlPanel(cv::Mat& panel_img) {
    // 重新绘制控制面板
    createControlPanel(panel_img);

    // 绘制所有功能按钮
    for (auto& btn : g_buttons) {
        bool highlight = false;
        // 根据鼠标位置决定是否高亮
        if (btn.enabled && btn.contains(g_mouse_pos)) {
            highlight = true;
        }
        drawButton(panel_img, btn, highlight);
    }

    // 绘制所有参数控件
    for (auto& param : g_params) {
        drawParameterControl(panel_img, param);
    }

    // 更新状态信息
    std::string status_text = g_image_loaded ? "Image Loaded" : "No Image Loaded";
    cv::Scalar status_color = g_image_loaded ? cv::Scalar(0, 255, 0) : cv::Scalar(100, 100, 255);

    if (g_processing_done) {
        status_text += " | Completed";
        status_color = cv::Scalar(0, 255, 0);
    }

    cv::putText(panel_img, status_text, cv::Point(40, 560),
        cv::FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1, cv::LINE_AA);

    if (g_result_saved) {
        std::string save_text = "Save dir: " + g_save_path;
        cv::putText(panel_img, save_text, cv::Point(40, 580),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1, cv::LINE_AA);
    }

    // 显示图像信息
    if (g_image_loaded && !g_original_image.empty()) {
        std::string img_info = "Image Size: " +
            std::to_string(g_original_image.cols) + "x" +
            std::to_string(g_original_image.rows);
        cv::putText(panel_img, img_info, cv::Point(40, 620),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);

        std::string param_info = "Patch: " + std::to_string(g_patch_size) +
            " | Stride: " + std::to_string(g_stride);
        cv::putText(panel_img, param_info, cv::Point(40, 640),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
    }
}

// 创建图像显示区域（右侧）
void createImageDisplay(cv::Mat& image_display) {
    image_display = cv::Mat::zeros(WINDOW_HEIGHT, IMAGE_WIDTH, CV_8UC3);

    // 绘制背景
    cv::rectangle(image_display, cv::Rect(0, 0, IMAGE_WIDTH, WINDOW_HEIGHT),
        cv::Scalar(30, 30, 30), -1, cv::LINE_AA);

    // 绘制分隔线
    cv::line(image_display, cv::Point(0, 0), cv::Point(0, WINDOW_HEIGHT),
        cv::Scalar(100, 100, 100), 3, cv::LINE_AA);

    if (g_image_loaded && !g_original_image.empty()) {
        // 显示标题
        cv::putText(image_display, "Original Image", cv::Point(280, 30),
            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

        cv::putText(image_display, "Reconstructed Mask", cv::Point(260, WINDOW_HEIGHT / 2 + 30),
            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

        // 计算显示区域大小
        int display_height = WINDOW_HEIGHT / 2 - 100;
        int display_width = display_height * g_original_image.cols / g_original_image.rows;

        // 确保宽度不超过显示区域
        if (display_width > IMAGE_WIDTH - 100) {
            display_width = IMAGE_WIDTH - 100;
            display_height = display_width * g_original_image.rows / g_original_image.cols;
        }

        // 显示原始图像（上半部分）
        cv::Mat original_display;
        cv::resize(g_original_image, original_display, cv::Size(display_width, display_height));

        // 居中显示原始图像
        int x_offset = (IMAGE_WIDTH - display_width) / 2;
        int y_offset = 80;

        cv::Mat roi_original = image_display(cv::Rect(x_offset, y_offset, display_width, display_height));
        original_display.copyTo(roi_original);

        // 添加边框
        cv::rectangle(image_display, cv::Rect(x_offset, y_offset, display_width, display_height),
            cv::Scalar(200, 200, 200), 2, cv::LINE_AA);

        // 添加标签
        cv::putText(image_display, "Original", cv::Point(x_offset + 10, y_offset + 25),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

        // 显示重建的掩膜（下半部分）
        if (g_processing_done && !g_reconstructed_mask.empty()) {
            cv::Mat mask_display;

            // 如果是单通道掩膜，转换为彩色显示
            if (g_reconstructed_mask.channels() == 1) {
                cv::cvtColor(g_reconstructed_mask, mask_display, cv::COLOR_GRAY2BGR);
            }
            else {
                g_reconstructed_mask.copyTo(mask_display);
            }

            // 调整掩膜大小
            cv::resize(mask_display, mask_display, cv::Size(display_width, display_height));

            // 居中显示掩膜
            y_offset = WINDOW_HEIGHT / 2 + 50;
            cv::Mat roi_mask = image_display(cv::Rect(x_offset, y_offset, display_width, display_height));
            mask_display.copyTo(roi_mask);

            // 添加边框
            cv::rectangle(image_display, cv::Rect(x_offset, y_offset, display_width, display_height),
                cv::Scalar(200, 200, 200), 2, cv::LINE_AA);

            // 添加标签
            cv::putText(image_display, "Mask", cv::Point(x_offset + 10, y_offset + 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

            // 显示掩膜统计信息
            int white_pixels = cv::countNonZero(g_reconstructed_mask);
            float white_ratio = (float)white_pixels / (g_reconstructed_mask.rows * g_reconstructed_mask.cols);

            std::string mask_info = "White pixels: " + std::to_string(white_pixels) +
                " (" + std::to_string(white_ratio * 100).substr(0, 4) + "%)";

            cv::putText(image_display, mask_info, cv::Point(x_offset, y_offset + display_height + 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 100), 1, cv::LINE_AA);
        }
        else {
            // 显示等待处理提示
            cv::putText(image_display, "Click 'Execute Processing' to generate mask",
                cv::Point(180, WINDOW_HEIGHT / 2 + 200),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(150, 150, 150), 1, cv::LINE_AA);
        }
    }
    else {
        // 显示欢迎界面
        cv::putText(image_display, "Image Display", cv::Point(300, 30),
            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

        cv::putText(image_display, "No Image Loaded", cv::Point(250, 300),
            cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(200, 200, 200), 3, cv::LINE_AA);

        cv::putText(image_display, "Click 'Import Image' to load an image",
            cv::Point(150, 350), cv::FONT_HERSHEY_SIMPLEX, 0.8,
            cv::Scalar(150, 150, 150), 1, cv::LINE_AA);

        // 绘制示例图像区域
        cv::rectangle(image_display, cv::Rect(100, 400, IMAGE_WIDTH - 200, 150),
            cv::Scalar(80, 80, 80), 2, cv::LINE_AA);
    }
}


// 合并控制面板和图像显示
void createMainWindow() {
    cv::Mat control_panel, image_display;

    // 创建控制面板
    updateControlPanel(control_panel);

    // 创建图像显示
    createImageDisplay(image_display);

    // 合并到主窗口
    g_main_window = cv::Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3);

    // 将控制面板复制到左侧
    control_panel.copyTo(g_main_window(cv::Rect(0, 0, CONTROL_WIDTH, WINDOW_HEIGHT)));

    // 将图像显示复制到右侧
    image_display.copyTo(g_main_window(cv::Rect(CONTROL_WIDTH, 0, IMAGE_WIDTH, WINDOW_HEIGHT)));
}

void importImage() {
    OPENFILENAMEA ofn;
    char szFile[260] = { 0 };

    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFile = szFile;
    ofn.nMaxFile = sizeof(szFile);
    ofn.lpstrFilter = "Image Files\0*.jpg;*.jpeg\0All Files\0*.*\0";
    ofn.nFilterIndex = 1;
    ofn.lpstrFileTitle = NULL;
    ofn.nMaxFileTitle = 0;
    ofn.lpstrInitialDir = NULL;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

    if (GetOpenFileNameA(&ofn) == TRUE) {
        std::string filename = ofn.lpstrFile;

        g_original_image = cv::imread(filename);
        if (g_original_image.empty()) {
            std::cerr << "图像加载失败: " << filename << std::endl;
            return;
        }

        g_image_loaded = true;
        g_processing_done = false;
        g_result_saved = false;

        // 清除之前的生成器和patch
        if (g_generator != nullptr) {
            delete g_generator;
            g_generator = nullptr;
        }
        g_patches.clear();

        std::cout << "加载图像 " << filename << std::endl;
        std::cout << "图像大小: " << g_original_image.cols << "x" << g_original_image.rows << std::endl;
    }
    else {
        std::cout << "文件选择取消" << std::endl;
    }
}

// 保存结果函数
void saveResults() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&time_t_now);

    // 格式化为字符串
    std::stringstream ss;
    ss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    std::string timestamp = ss.str();

    std::string ori_path = g_save_path + "oriimg_" + timestamp + ".jpg";
    cv::imwrite(ori_path, g_original_image);
    std::cout << "Original Image saved as " + ori_path << std::endl;

    if (!g_reconstructed_mask.empty()) {
        std::string mask_path = g_save_path + "mask_" + timestamp + ".jpg";
        cv::imwrite(mask_path, g_reconstructed_mask);
        std::cout << "Mask saved as " + mask_path << std::endl;
    }
    if (!g_reconstructed_img.empty()) {
        std::string img_path = g_save_path + "reconstruct_image_" + timestamp + ".jpg";
        cv::imwrite(img_path, g_reconstructed_img);
        std::cout << "Image saved as " + img_path << std::endl;
    }

    if (g_processing_done) {
        // 显示保存成功提示
        std::cout << "Results saved successfully!" << std::endl;
    }
    g_result_saved = true;
}

void executeProcessing() {

    if (!g_image_loaded) {
        std::cerr << "请先导入图像！" << std::endl;
        return;
    }

    // 设置模块参数
    std::vector<string> class_names = { "_background_", "Line" };

    int patch_size = 512;
    int stride = 256;

    string detect_mode = "overlap"; // 检测模式，可选"overlap", "contour"。“contour”未实现
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
        return;
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
        return;
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
    std::cout << "开始处理图像..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now(); // 开始计时
    // 创建PatchGenerator
    if (g_generator != nullptr) {
        delete g_generator;
    }
    g_generator = new PatchGenerator(g_original_image, g_patch_size, g_stride, g_fill_value);

    // 获取所有patch
    g_patches = g_generator->getAllPatches();


    std::cout << "Generated " << g_patches.size() << " patches." << std::endl;

    for (auto& patchInfo : g_patches) {
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
            return;
        }
    }

    // cv::Mat reconstructed = reconstructImageFromGenerator(patches, generator);
    reconstructImageAndMask(g_patches, *g_generator, g_reconstructed_img, g_reconstructed_mask);
    g_processing_done = true;

    // 显示重建的掩膜
    if (!g_reconstructed_mask.empty()) {
        cv::Mat display_mask = g_reconstructed_mask.clone();
        cv::resize(display_mask, display_mask, cv::Size(600, 400));
        cv::imshow("重建的掩膜", display_mask);

        // 统计掩膜中白色像素的比例
    //     int white_pixels = cv::countNonZero(g_reconstructed_mask);
    //     float white_ratio = (float)white_pixels / (g_reconstructed_mask.rows * g_reconstructed_mask.cols);
    //     std::cout << "掩膜重建完成！" << std::endl;
    //     std::cout << "白色像素数量: " << white_pixels << " (" << (white_ratio * 100) << "%)" << std::endl;
    //
    }

    // 保存结果
   // if (!g_reconstructed_mask.empty()) {
       // cv::imwrite("reconstructed_mask.png", g_reconstructed_mask);
       // std::cout << "掩膜已保存为 reconstructed_mask.png" << std::endl;
    // }
    saveResults();

    std::cout << "处理完成！" << std::endl;
    auto end = std::chrono::high_resolution_clock::now(); // 结束计时
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "耗时: " << duration.count() << " 毫秒" << std::endl;
}



// 鼠标回调函数
void onMouse(int event, int x, int y, int flags, void* userdata) {
    (void)flags;
    (void)userdata;

    // 保存全局鼠标位置
    g_mouse_pos = cv::Point(x, y);

    // 检查是否点击在控制面板区域（左侧）
    if (x < CONTROL_WIDTH) {
        // 调整鼠标位置为控制面板局部坐标
        cv::Point local_pos = cv::Point(x, y);

        if (event == cv::EVENT_LBUTTONDOWN) {
            // 检查功能按钮点击
            for (auto& btn : g_buttons) {
                if (btn.enabled && btn.contains(local_pos)) {
                    btn.pressed = true;
                    g_update_ui = true;

                    // 根据按钮ID执行相应操作
                    switch (btn.id) {
                    case BTN_IMPORT:
                        std::cout << "Import button clicked" << std::endl;
                        importImage();
                        break;
                    case BTN_EXECUTE:
                        std::cout << "Execute button clicked" << std::endl;
                        executeProcessing();
                        break;
                    case BTN_SAVE:
                        std::cout << "Save button clicked" << std::endl;
                        saveResults();
                        break;
                    }
                    return;
                }
            }

            // 检查参数调节按钮点击
            for (auto& param : g_params) {
                // 检查减少按钮
                if (param.btn_minus.contains(local_pos)) {
                    param.btn_minus.pressed = true;
                    *(param.value) = std::max(param.min_value, *(param.value) / 2);
                    std::cout << param.label << " decreased to: " << *(param.value) << std::endl;
                    g_update_ui = true;
                    return;
                }

                // 检查增加按钮
                if (param.btn_plus.contains(local_pos)) {
                    param.btn_plus.pressed = true;
                    *(param.value) = std::min(param.max_value, *(param.value) * 2);
                    std::cout << param.label << " increased to: " << *(param.value) << std::endl;
                    g_update_ui = true;
                    return;
                }
            }

        }
        else if (event == cv::EVENT_LBUTTONUP) {
            // 释放所有按钮
            for (auto& btn : g_buttons) {
                if (btn.pressed) {
                    btn.pressed = false;
                    g_update_ui = true;
                }
            }

            // 释放所有参数按钮
            for (auto& param : g_params) {
                if (param.btn_minus.pressed) {
                    param.btn_minus.pressed = false;
                    g_update_ui = true;
                }
                if (param.btn_plus.pressed) {
                    param.btn_plus.pressed = false;
                    g_update_ui = true;
                }
            }

        }
        else if (event == cv::EVENT_MOUSEMOVE) {
            // 更新UI显示
            g_update_ui = true;
        }
    }
}

int main() {
    std::cout << "==============================" << std::endl;
    std::cout << "  Segmentation Inference App v5.0  " << std::endl;
    std::cout << "  (Improved Interface)        " << std::endl;
    std::cout << "==============================" << std::endl;

    // 创建主窗口
    cv::namedWindow("Segmentation Inference App", cv::WINDOW_NORMAL);
    cv::resizeWindow("Segmentation Inference App", WINDOW_WIDTH, WINDOW_HEIGHT);

    // 设置鼠标回调
    cv::setMouseCallback("Segmentation Inference App", onMouse, NULL);

    // 创建初始界面
    createMainWindow();
    cv::imshow("Segmentation Inference App", g_main_window);

    std::cout << "\nInstructions:" << std::endl;
    std::cout << "1. Click 'Import Image' to select an image" << std::endl;
    std::cout << "2. Adjust parameters using +/- buttons" << std::endl;
    std::cout << "3. Click 'Execute Processing' to process the image" << std::endl;
    std::cout << "4. View results (original image and mask)" << std::endl;
    std::cout << "5. Click 'Save Results' to save the mask" << std::endl;
    std::cout << "6. Press ESC to exit" << std::endl;

    // 主循环
    while (true) {
        // 更新UI
        if (g_update_ui) {
            createMainWindow();
            cv::imshow("Segmentation Inference App", g_main_window);
            g_update_ui = false;
        }

        // 等待按键
        int key = cv::waitKey(30);

        // ESC键退出
        if (key == 27) {
            break;
        }

        // 键盘快捷键
        switch (key) {
        case 'i':
        case 'I':
            importImage();
            g_update_ui = true;
            break;
        case 'e':
        case 'E':
            executeProcessing();
            g_update_ui = true;
            break;
        case 's':
        case 'S':
            saveResults();
            g_update_ui = true;
            break;
        case 'q':
        case 'Q':
            goto exit_program;
        case '+':
            g_patch_size = std::min(1024, g_patch_size * 2);
            std::cout << "Patch size increased to: " << g_patch_size << std::endl;
            g_update_ui = true;
            break;
        case '-':
            g_patch_size = std::max(64, g_patch_size / 2);
            std::cout << "Patch size decreased to: " << g_patch_size << std::endl;
            g_update_ui = true;
            break;
        case '>':
        case '.':
            g_stride = std::min(512, g_stride * 2);
            std::cout << "Stride increased to: " << g_stride << std::endl;
            g_update_ui = true;
            break;
        case '<':
        case ',':
            g_stride = std::max(64, g_stride / 2);
            std::cout << "Stride decreased to: " << g_stride << std::endl;
            g_update_ui = true;
            break;
        }
    }

exit_program:
    // 清理资源
    if (g_generator != nullptr) {
        delete g_generator;
    }

    cv::destroyAllWindows();
    std::cout << "Program exited" << std::endl;

    return 0;
}