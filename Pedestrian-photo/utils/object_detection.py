import tensorrt as trt
from utils import common
from utils.data_processing import *
from utils.draw import draw_boxes, put_text_frame, save_img, put_QR
import time
import qrcode as qr
import uuid
from PIL import Image
import sys
import requests

TRT_LOGGER = trt.Logger()

def send_image_uid(uid,url,image_path,folder_save):
    name= uid+'_img.'+image_path.split('.')[-1]
    payload={'code': uid}
    files=[
    ('image',(name,open(image_path,'rb'),folder_save))
    ]
    headers = {}
    try:
        response = requests.request("POST", url, headers=headers, data=payload, files=files)
        return response
    except:
        e = sys.exc_info()[1]
        return -1
    
def get_engine(engine_file_path):
    # If a serialized engine exists, use it instead of building an engine.
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())
def generate_QR():
    uid = str(uuid.uuid1())
    QR = qr.QRCode(
        version=1,
        error_correction=qr.constants.ERROR_CORRECT_H,
        box_size=3,
        border=4,
        )
    QR.add_data('https://medium.com/id?'+uid)
    QR.make(uid)
    QR_im = QR.make_image(fill_color="black", back_color="white").convert('RGB')    
    return uid, np.array(QR_im)
class people_hand_detector():
    def __init__(self, engine_file_path,img_path,url="http://192.168.1.13:3333/device/photo",folder_save='image/jpg',background_img=None):
        #---tensorrt----#
        self.engine = get_engine(engine_file_path)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)
        # ---tensorrt----#
        # initializate current photo
        self.count_hand_frames = 0
        self.background_img=background_img
        self.img_path=img_path
        self.count_frames = 0
        self.ori_im_qr=None
        self.time_show_photo=0
        self.flag_show_photo = False
        self.prev_time = time.time()
        self.save_foto_flag=False
        self.time_before_photo=2
        self.url=url
        self.folder_save=folder_save
        #---input info for yolov3-416------#
        self.input_resolution_yolov3_HW = (512, 512)#(416, 416)

        self.preprocessor = PreprocessYOLO(self.input_resolution_yolov3_HW)

        # self.image_raw, self.image = self.preprocessor.process(ori_im)

        # self.shape_orig_WH = image_raw.size
        #TODO tiny
        self.output_shapes = [(1, 24, 16, 16), (1, 24, 32, 32)]#[(1, 255, 13, 13), (1, 255, 26, 26)]
        self.postprocessor_args = {"yolo_masks": [ (3, 4, 5), (0, 1, 2)],
                              # A list of 3 three-dimensional tuples for the YOLO masks
                              "yolo_anchors": [(12,58), (21,85), (29,130), (41,164), (51,247), (80,337)],#[(10, 14), (23, 27), (37, 58),(81, 82), (135, 169), (344, 319)],
                              "obj_threshold": 0.5,  # Threshold for object coverage, float value between 0 and 1
                              "nms_threshold": 0.3,
                              # Threshold for non-max suppression algorithm, float value between 0 and 1
                              "yolo_input_resolution": self.input_resolution_yolov3_HW}

        self.postprocessor = PostprocessYOLO(**self.postprocessor_args)

    def _xywh_to_xyxy(self, bbox_xywh):
        x,y,w,h = bbox_xywh
        x1 = max(int(x-w/2),0) # 0
        x2 = min(int(x+w/2),self.width-1) #150
        y1 = max(int(y-h/2),0) # 0
        y2 = min(int(y+h/2),self.height-1) #150
        return x1,y1,x2,y2

    def take_photo(self,ori_im):
        if time.time()-self.prev_time > self.time_before_photo:
            if self.count_hand_frames/self.count_frames > 0.1 or self.save_foto_flag:
                #color=(0,0,255)
                #text = 'Tiempo con mano {:1f}'.format(time.time()-self.prev_time)
                self.save_foto_flag=True
            else:    
                #text = 'Sin mano'
                #color=(255,0,0)
                self.count_hand_frames=0
                self.count_frames=0
                self.prev_time = time.time()
            if self.save_foto_flag:
                color=(0,255,255)
                if self.time_before_photo+5-(int(time.time()-self.prev_time)) < 0:
                    save_img(self.img_path,ori_im,self.background_img)
                    uid, QR_im = generate_QR()
                    outserver = send_image_uid(uid,self.url,self.img_path,self.folder_save)
                    self.ori_im_qr = put_QR(ori_im,QR_im,outserver)
                    self.flag_show_photo = True
                    self.save_foto_flag=False
                    self.count_hand_frames=0
                    self.count_frames=0
                    self.prev_time = time.time()
                    self.time_show_photo = time.time()
                    ori_im = np.ones_like(ori_im,dtype=np.uint8)*255
                else:
                    text = '{:d}'.format(self.time_before_photo+5-(int(time.time()-self.prev_time)))
                    ori_im = put_text_frame(ori_im,text,color)
        #else:
            #text = 'Sin mano'
            #color=(255,0,0)
        return ori_im

    def run(self, ori_im):
        self.count_frames+=1
        image_raw, image = self.preprocessor.process(ori_im)
        shape_orig_WH = image_raw.size
        self.width = shape_orig_WH[0]
        self.height = shape_orig_WH[1]
        # print('type of image:',  type(image))

        self.inputs[0].host = image
        trt_outputs = common.do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)

        trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, self.output_shapes)]
        bbox_xywh, cls_ids, cls_conf = self.postprocessor.process(trt_outputs, (shape_orig_WH))
        #print(cls_ids)
        ori_im = self.take_photo(ori_im)
        if self.flag_show_photo:
            ori_im = self.ori_im_qr
            bbox_xywh=None
            if time.time()- self.time_show_photo > 30:
                self.flag_show_photo = False
        mask = cls_ids == 2
        bbox_xywh = bbox_xywh[mask]
        if bbox_xywh is not None:
            # select person class
            #bbox_xywh[:, 3:] *= 1.2
            #cls_conf = cls_conf[mask]
            # print('hahahat', bbox_xywh.dtype)
            # do tracking
            #outputs = self.deepsort.update(bbox_xywh, cls_conf, ori_im)

            # draw boxes for visualization
            #if len(outputs) > 0:
            #bbox_xyxy = outputs[:, :4]
            bbox_xyxy = []
            for box in bbox_xywh:
                x1,y1,x2,y2 = self._xywh_to_xyxy(box)
                bbox_xyxy.append(np.array([x1,y1,x2,y2],dtype=np.int))
            bbox_xyxy = np.stack(bbox_xyxy,axis=0)
            #bbox_xyxy=bbox_xywh
            #identities = outputs[:, -1]
            if sum(cls_ids == 2)>0:
                self.count_hand_frames += 1
            if not self.save_foto_flag:
                ori_im = draw_boxes(ori_im, bbox_xyxy)
        return ori_im

