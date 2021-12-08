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

### Training yolo model
[Training](Training/Training_yolov4_tiny.ipynb) YoloV4-tiny model

### Step by step

1. Clone this repo
  ```
  git clone https://github.com/OpenMLCo/TensorRT-Yolo-Deepsort.git
  ```

2. Download pre-training weights and include this file in weights folder.
  ```shell
  gdown --id 10WXslmbCEqVR34jpMKfRYDVr2KhmzC9p
  ```
3. Include yolo .cfg file in configs folder.
4. Edit configs/DG_labels.txt file with the labels in yolo model.
5. Convert yolo model to Onnx.
  ```shell
  python3 yolo_to_onnx.py --model yolov4-tiny-512 --weights weights/xxxx.weights --config confis/xxxx.cfg --output_file model_onnx/xxxx.onnx --category_num x
  ```
6. Convert Onnx model to TensorRT engine. 
  ```shell
  python3 onnx_to_tensorrt.py --onnx_model model_onnx/xxxx.onnx --output_engine model_tensorRT/xxxx.engine
  ```
7. Run demos.
  * PedTrack
    ```shell
    #Using usb camera:
    python3 run_tracker_tini.py --engine_path model_tensorRT/xxx.engine --usb
    ```    
  * Pedestrian-photo
    ```shell
    #Using usb camera:
    python3 run_hand_photo.py --engine_path model_tensorRT/xxxx.engine --usb
    ```

## Credit
This repo was created under xxx UAM xxx Smart cities in Caldas, Colombia. !!!

## Contact us
- Daniel Garcia Murillo (danielggarciam@gmail.com)
- Julián Caicedo Acosta (julianc.caicedoa@autonoma.edu.co, juccaicedoac@gmail.com)




