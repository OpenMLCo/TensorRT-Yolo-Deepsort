import sys
import argparse

from utils.camera_setting import *
from utils.parser import get_config
import cv2
import time
import requests

from tracker.tracker_tiny import Tracker_tiny

WINDOW_NAME = 'TrtYolov3_tiny_deepsort'

def send_image_uid(data,url):
    payload={'data': data}
    try:
        response = requests.request("POST", url, data=payload)
        return response
    except:
        e = sys.exc_info()[1]
        return -1

def parse_args():
    """Parse camera and input setting arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time MOT with TensorRT optimized '
            'YOLOv3 model on Jetson Nano')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    #TODO change default tiny engine
    parser.add_argument('--engine_path', type=str, default='./weights/yolov4-tiny-512.engine', help='set your engine file path to load')
    parser.add_argument('--config_deepsort', type=str, default="./configs/deep_sort.yaml")
    parser.add_argument('--output_file', type=str, default='./test.mp4', help='path to save your video like  ./test.mp4')
    parser.add_argument('--server_url',type=str,default='http://192.168.1.13:3333/device/track',
                            help='server url')
    parser.add_argument('--frame_send',type=int,default=30,
                            help='send data frequency in frames')
    args = parser.parse_args()
    return args

def open_window(window_name, width, height, title):
    """Open the display window."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    cv2.setWindowTitle(window_name, title)


def loop_and_track(cam, tracker, arg):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      tracker: the TRT YOLOv3 object detector instance.
    """

    if arg.filename:
        while True:
            if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
                break

            img = cam.read()
            if img is not None: #this line is a must in case not reading img correctly
                start = time.time()
                img_final, counts = tracker.run(img)
                cv2.imshow(WINDOW_NAME, img_final)
                cam.write(img_final)
                end = time.time()
                print("time: {:.03f}s, fps: {:.03f}".format(end - start, 1 / (end - start)))
                if frame_id%args.frame_send==0:
                    send_image_uid(counts,args.server_url)
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
                img_final, counts = tracker.run(img)
                cv2.imshow(WINDOW_NAME, img_final)
                end = time.time()
                print("time: {:.03f}s, fps: {:.03f}".format(end - start, 1 / (end - start)))
                if frame_id%args.frame_send==0:
                    send_image_uid(counts,args.server_url)                
            key = cv2.waitKey(1)
            if key == 27:  # ESC key: quit program
                break




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
    loop_and_track(cam, tracker, args)

    cam.stop()

    cam.release()
    cv2.destroyAllWindows()

    if args.filename:
        print('result video saved at (%s)' %(args.output_file))
    else:
        print('close')


if __name__ == '__main__':
    main()
