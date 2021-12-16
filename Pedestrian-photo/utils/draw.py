import numpy as np
import cv2
from change_backgroud import change_backgroud

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def put_QR(ori_im,QR_im,outserver):
    w,h,_ = QR_im.shape
    if outserver == -1:
        QR_im = cv2.line(QR_im, (w, 0), (0, h), (0,0,255), 3)
        QR_im = cv2.line(QR_im, (0, 0), (w, h), (0,0,255), 3)
    ori_im[-w::,-h::] = QR_im
    return ori_im


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def put_text_frame(img,text,color):
    font=cv2.FONT_HERSHEY_DUPLEX
    TEXT_SCALE = 5
    TEXT_THICKNESS = 10
    textsize = cv2.getTextSize(text, font, TEXT_SCALE, TEXT_THICKNESS)[0]
    textX = int((img.shape[1] - textsize[1])/2)
    textY = int((img.shape[0] + textsize[0])/2)
    img=cv2.putText(img, text, (textX, textY ), font, TEXT_SCALE, color, TEXT_THICKNESS)
    img=cv2.circle(img, (int(img.shape[1]/2), int(img.shape[0]/2) ), 100, color, 10)
    return img

def save_img(img_path,img,background_img):
    cv2.imwrite(img_path,img)
    if background_img is not None:
        img_final = change_backgroud(img_path,background_img)
        cv2.imwrite(img_path,img_final)
def draw_boxes(img, bbox, identities=None, offset=(0,0)):
    for i,box in enumerate(bbox):
        x1,y1,x2,y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0    
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]

        cv2.rectangle(img,(x1, y1),(x2,y2),color,3)
        cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)

        cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
    return img



if __name__ == '__main__':
    for i in range(82):
        print(compute_color_for_labels(i))
