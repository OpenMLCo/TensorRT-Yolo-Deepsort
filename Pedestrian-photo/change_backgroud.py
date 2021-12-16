import math
import random
import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
#from tqdm import tqdm
from config import device
#from data_gen import data_transforms
import time
from detectron2.utils.logger import setup_logger
setup_logger()
import time
# import some common libraries
import numpy as np
import cv2
#from google.colab.patches import cv2_imshow
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
prev=time.time()

data_transforms = {
    'train': transforms.Compose([
        transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def gen_test_names():
    num_fgs = 50
    num_bgs = 1000
    num_bgs_per_fg = 20
    names = []
    bcount = 0
    for fcount in range(num_fgs):
        for i in range(num_bgs_per_fg):
            names.append(str(fcount) + '_' + str(bcount) + '.png')
            bcount += 1
    return names

def process_test(im_name, bg_name, trimap):
    # print(bg_path_test + bg_name)
    im = cv.imread(im_name)
    a = cv.imread(im_name, 0)
    h, w = im.shape[:2]
    bg = cv.imread(bg_name)
    bh, bw = bg.shape[:2]
    wratio = w / bw
    hratio = h / bh
    ratio = wratio if wratio > hratio else hratio
    if ratio > 1:
        bg = cv.resize(src=bg, dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)), interpolation=cv.INTER_CUBIC)
    return composite4_test(im, bg, a, w, h, trimap)

def composite4(fg, bg, a, w, h):
    print(fg.shape, bg.shape, a.shape, w, h)
    fg = np.array(fg, np.float32)
    bg_h, bg_w = bg.shape[:2]
    x = 0
    if bg_w > w:
        x = np.random.randint(0, bg_w - w)
    y = 0
    if bg_h > h:
        y = np.random.randint(0, bg_h - h)
    bg = np.array(bg[y:y + h, x:x + w], np.float32)
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a
    im = alpha * fg + (1 - alpha) * bg
    im = im.astype(np.uint8)
    return im, bg

def gen_trimap(alpha):
    k_size = random.choice(range(1, 5))
    iterations = np.random.randint(1, 20)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_size, k_size))
    dilated = cv.dilate(alpha, kernel, iterations)
    eroded = cv.erode(alpha, kernel, iterations)
    trimap = np.zeros(alpha.shape)
    trimap.fill(128)
    trimap[eroded >= 255] = 255
    trimap[dilated <= 0] = 0
    return trimap

def composite4_test(fg, bg, a, w, h, trimap):
    fg = np.array(fg, np.float32)
    bg_h, bg_w = bg.shape[:2]
    x = max(0, int((bg_w - w) / 2))
    y = max(0, int((bg_h - h) / 2))
    crop = np.array(bg[y:y + h, x:x + w], np.float32)
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a / 255.
    # trimaps = np.zeros((h, w, 1), np.float32)
    # trimaps[:,:,0]=trimap/255.
    im = alpha * fg + (1 - alpha) * crop
    im = im.astype(np.uint8)
    new_a = np.zeros((bg_h, bg_w), np.uint8)
    new_a[y:y + h, x:x + w] = a
    new_trimap = np.zeros((bg_h, bg_w), np.uint8)
    new_trimap[y:y + h, x:x + w] = trimap
    #cv.imwrite('images/test/new/tripmap_new.png', new_trimap)
    new_im = bg.copy()
    new_im[y:y + h, x:x + w] = im
    # cv.imwrite('images/test/new_im/'+trimap_name,new_im)
    return new_im, new_a, fg, bg, new_trimap, y, y + h, x, x + w


def semantic_segmentation(im_name):
    im = cv2.imread(im_name)
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    prev_2=time.time()
    outputs = predictor(im)
    #v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    #out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #imagen = out.get_image()[:, :, ::-1]
    #cv2.imwrite('Imagen_result_person.jpg',imagen)
    rr=outputs["instances"].to("cpu")
    for i,mask in enumerate(rr.pred_masks):
        return (255*mask.numpy()).astype(np.uint8)

def change_backgroud(im_name,bg_name):
    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model'].module
    model = model.to(device)
    model.eval()
    transformer = data_transforms['valid']


    mask = semantic_segmentation(im_name)
    #im_name = '/home/daniel/Documents/cambio_fondo/person.jpeg'
    #bg_name = '/home/daniel/Documents/cambio_fondo/backgroun.jpeg'

    kernel = np.ones((10, 10), np.uint8)
    foreground = cv.erode(mask, kernel,3)
    kernel = np.ones((10, 10), np.uint8)
    nose = cv.dilate(mask, kernel)
    nose = nose-foreground
    trimap = np.zeros_like(mask)+foreground+nose*0.5
    #cv.imwrite('mask.jpg',trimap)


    print('set_net',time.time()-prev)
    prev_2 = time.time()
    img, alpha, fg, bg, new_trimap,y_2,yh, x_2,xw = process_test(im_name, bg_name, trimap)

    h, w = img.shape[:2]
    trimap = gen_trimap(alpha)
    # mytrimap = gen_trimap(alpha)
    # cv.imwrite('images/test/new_im/'+trimap_name,mytrimap)

    x = torch.zeros((1, 4, h, w), dtype=torch.float)
    img = img[..., ::-1]  # RGB
    img = transforms.ToPILImage()(img)  # [3, 320, 320]
    img = transformer(img)  # [3, 320, 320]
    x[0:, 0:3, :, :] = img
    x[0:, 3, :, :] = torch.from_numpy(new_trimap.copy() / 255.)

    # Move to GPU, if available
    x = x.type(torch.FloatTensor).to(device)  # [1, 4, 320, 320]
    alpha = alpha / 255.

    with torch.no_grad():
        pred = model(x)  # [1, 4, 320, 320]

    pred = pred.cpu().numpy()
    pred = pred.reshape((h, w))  # [320, 320]

    pred[new_trimap == 0] = 0.0
    pred[new_trimap == 255] = 1.0
    #cv.imwrite('/content/drive/MyDrive/La_victoria_Caldas_jetson_nano/Codes_Victoria_vision/Matting image/tales.jpg', pred * 255)

    im_i= cv.imread(im_name)
    bg_i=   cv.imread(bg_name)
    width,heigth,_= im_i.shape
    pred_3d = cv.resize(pred[:,:,np.newaxis], (heigth,width), interpolation = cv.INTER_AREA)
    pred_3d = pred_3d[:,:,np.newaxis]
    final = cv.resize(bg_i, (heigth,width), interpolation = cv.INTER_AREA)
    final[y_2:yh, x_2:xw] = final[y_2:yh, x_2:xw]*(1-pred_3d[y_2:yh, x_2:xw])+im_i[y_2:yh, x_2:xw]*(pred_3d[y_2:yh, x_2:xw])
    final=final.astype(np.uint8)
    #cv.imwrite('tales.jpg',final)
    print('final total',time.time()-prev,'predict',time.time()-prev_2)
    return final

#if __name__ == "__main__":
#    # unconmment next line for an example of batch processing
#    # batch_detection_example()
#    change_backgroud('/home/daniel/Documents/cambio_fondo/person.jpeg','/home/daniel/Documents/cambio_fondo/backgroun.jpeg')