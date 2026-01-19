# SOLOv2-Line #

This is a SOLOv2 project based on only pytorch and opencv-python
created on Nov, 26

## Update ##
Jan, 19 V1.1: ** DCNv2 is available now! **
For using it, choose the config: Line_solov2_r50_light_dcn in train.py and detect.py

## Dependency ##
Pytorch 1.13.1
CUDA 11.7 + cudnn 8.5.0
opencv-python, pycocotool
(maybe tqdm needed)
> Notice that cv2.imwrite may save no image without error reporting. Refer to the new detect.py for details.

Now testing on ubuntu 20.04 and Windows 10

## Data ##
[X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) by cvhub is recommened for data annotation.
And THIS project use the format of COCO_instance. You can export or transform to this format by X-anylabeling easily.
Do not forget to edit the class name and number in the file "configs.py"

## Training, evaluating and detecting ##
run 'train.py', 'val.py', 'detect.py' to try the model.

## Reference & weights ##
The original site of [SOLOv2](https://arxiv.org/abs/2003.10152)
And the project [SOLOv2_minimal](https://github.com/feiyuhuahuo/SOLOv2_minimal) is a good work. Also thanks for the [weights](https://github.com/feiyuhuahuo/SOLOv2_minimal/releases/tag/v1.0 "Download weights here")
