# TensorRT-Yolo-Deepsort

## Table of contents
* [PedTrack](PedTrack/): YoloV4-tiny with DeepSort for pedestrian tracking.
* [Pedestrian-photo](Pedestrian-photo): People and hand detection using YoloV4-tiny for user expirience.

## Install

### Environment

- python 3.xx
- Jetson nano with TernsorRT x.x.x.x

### Requirements

- jetpack >= 4.5, following the [instructions](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#intro) <br />
- onnx == 1.4.0 or 1.4.1 <br />

### Step by step

1. Clone this repo
  ```
  git clone xxxx
  ```

2. Download weights


### Initial configuration

Edit configs/DG_labels.txt file with the labels in yolo model. <br />
Include yolo .cfg file in foler configs <br />
Include yolo .weights file in foler weights <br />

### Convert yolo model to Onnx

Use the following code: <br />
python yolo_to_onnx.py --model yolov4-tiny-512 --weights weights/xxxx.weights --config confis/xxxx.cfg --output_file model_onnx/xxxx.onnx --category_num 3 <br />

### Convert Onnx model to TensorRT engine
Use the following code: <br />
python onnx_to_tensorrt.py --onnx_model model_onnx/xxxx.onnx --output_engine model_tensorRT/xxxx.engine

### Run code

python run_hand_photo.py --engine_path model_tensorRT/xxxx.engine --usb 


### Contact us
Daniel Garcia Murillo (danielggarciam@gmail.com)
Julian Caicedo Acosta (juccaicedoac@unal.edu.co )




