import sys
import argparse

from utils.camera_setting import *
from utils.parser import get_config
import cv2
import time

from tracker.tracker_tiny import Tracker_tiny
from threading import Thread
from queue import Queue

WINDOW_NAME = 'TrtYolov3_tiny_deepsort'

def parse_args():
    """Parse camera and input setting arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time MOT with TensorRT optimized '
            'YOLOv3 model on Jetson Nano')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    #TODO change default tiny engine
    parser.add_argument('--engine_path', type=str, default='./weights/yolov4-tiny-800.engine', help='set your engine file path to load')
    parser.add_argument('--config_deepsort', type=str, default="./configs/deep_sort.yaml")
    parser.add_argument('--output_file', type=str, default='./test.mp4', help='path to save your video like  ./test.mp4')

    args = parser.parse_args()
    return args

def open_window(window_name, width, height, title):
    """Open the display window."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    cv2.setWindowTitle(window_name, title)

def read_img(cam,yolo_image_queue):
    while cam.cap.isOpened():
        img = cam.read()
        yolo_image_queue.put(img)
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
                break
    cam.stop()
    cam.release()
    cv2.destroyAllWindows()

def inference(cam,img_out,yolo_image_queue,tracker):
    while cam.cap.isOpened():
        img = yolo_image_queue.get()
        img_final = tracker.run(img)
        img_out.put(img_final)
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
                break
    cam.stop()
    cam.release()
    cv2.destroyAllWindows()

def drawing(cam,img_out):
    while cam.cap.isOpened():
        img_final = img_out.get()
        cv2.imshow(WINDOW_NAME, img_final)
        cam.write(img_final)
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
                break
    cam.stop()
    cam.release()
    cv2.destroyAllWindows()

def main():
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_deepsort)

    cam = Camera(args)
    cam.open()
    if not cam.is_opened:
        sys.exit('Failed to open camera!')

    tracker = Tracker_tiny(cfg, args.engine_path) #TODO

    cam.start()
    open_window(WINDOW_NAME, args.image_width, args.image_height,
                'TrtYolov3_deepsort')

    ###### threading
    yolo_image_queue = Queue(maxsize=5)
    img_out = Queue(maxsize=5)
    
    Thread(target=read_img, args=(cam,yolo_image_queue)).start()
    Thread(target=inference, args=(cam,img_out,yolo_image_queue,tracker)).start()
    Thread(target=drawing, args=(cam,img_out)).start()
    # ############


if __name__ == '__main__':
    main()
