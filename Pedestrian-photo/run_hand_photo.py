import sys
import argparse

from utils.camera_setting import *
from utils.object_detection import people_hand_detector

import cv2
import time

WINDOW_NAME = 'TrtYolov4-tiny'

def parse_args():
    """Parse camera and input setting arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time MOT and hand detection with TensorRT optimized ')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    #TODO change default tiny engine
    parser.add_argument('--engine_path', type=str, default='./weights/yolov4-tiny-800.engine',
                            help='set your engine file path to load')
    parser.add_argument('--output_file', type=str, default='./test.mp4', 
                            help='path to save your video like  ./test.mp4')
    parser.add_argument('--output_photo', type=str, default='img.jpg', 
                            help='path to save photo')
    parser.add_argument('--server_url',type=str,default='http://192.168.1.13:3333/device/photo',
                            help='server url')
    parser.add_argument('--folder_save',type=str,default='image/jpg',
                            help='folder path to save imagen on server')                            
    args = parser.parse_args()
    return args

def open_window(window_name, width, height, title):
    """Open the display window."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    cv2.setWindowTitle(window_name, title)

def loop_and_detect(cam, model, arg):
    """Continuously capture images from camera and do object detection.
    # Arguments
      cam: the camera instance (video source).
      model: the TRT object detector model.
    """
    if arg.filename:
        while True:
            if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
                break
            img = cam.read()
            if img is not None: #this line is a must in case not reading img correctly
                start = time.time()
                img_final = model.run(img)
                cv2.imshow(WINDOW_NAME, img_final)
                cam.write(img_final)
                end = time.time()
                print("time: {:.03f}s, fps: {:.03f}".format(end - start, 1 / (end - start)))
            key = cv2.waitKey(1)
            if key == 27:  # ESC key: quit program
                break
    else:
        while True:
            # if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            #     break
            img = cam.read()
            if img is not None: #this line is a must in case not reading img correctly
                start = time.time()
                img_final = model.run(img)
                cv2.imshow(WINDOW_NAME, img_final)
                end = time.time()
                print("time: {:.03f}s, fps: {:.03f}".format(end - start, 1 / (end - start)))
            key = cv2.waitKey(1)
            if key == 27:  # ESC key: quit program
                break


def main():
    args = parse_args()
    cam = Camera(args)
    cam.open()
    if not cam.is_opened:
        sys.exit('Failed to open camera!')
    model = people_hand_detector(args.engine_path,args.output_photo,args.server_url,args.folder_save)

    cam.start()
    open_window(WINDOW_NAME, args.image_width, args.image_height,
                'TrtYolov4-tiny')
    loop_and_detect(cam, model, args)

    cam.stop()
    cam.release()
    cv2.destroyAllWindows()

    if args.filename:
        print('result video saved at (%s)' %(args.output_file))
    else:
        print('close')

if __name__ == '__main__':
    main()
