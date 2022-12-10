import numpy as np
import time
import cv2
import os
from utils.utils import *
import argparse

from dbnet.infer_x3 import dbnet_model
from crnn.infer_x3 import crnn_model


def init_args():
    parser = argparse.ArgumentParser(description='DBNet.pytorch')
    parser.add_argument('--dbnet_path', default='../../data/model_zoo/dbnet_simp.bin', type=str)
    parser.add_argument('--crnn_path', default='../../data/model_zoo/crnn_simp.bin', type=str)
    parser.add_argument('--image_path', default='test_images/test2.png', type=str, help='img path for predict')
    parser.add_argument('--show', default=True, type=bool)
    parser.add_argument('--output_folder', default='./output', type=str, help='img path for output')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = init_args()
    dbnet = dbnet_model(args.dbnet_path) 
    
    alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
    converter = strLabelConverter(alphabet)   
    crnn = crnn_model(args.crnn_path,converter)
    img0 = cv2.imread(args.image_path) 
    img_rec = img0.copy()
    img0_h,img0_w = img0.shape[:2]
    
    contours, boxes_list, t = dbnet.predict(img0)
    
    for i,cnt in enumerate(contours):
        mask_t = np.zeros((img0_h, img0_w), dtype=np.uint8)
        cv2.fillPoly(mask_t, [cnt], (255), 8, 0)
        pick_img = cv2.bitwise_and(img0, img0, mask=mask_t)
        x, y, w, h = cv2.boundingRect(cnt)
        crnn_infer_img =  pick_img[y:y+h,x:x+w,:]
        crnn_infer_img = cv2.cvtColor(crnn_infer_img,cv2.COLOR_BGR2GRAY)
        raw_pred,sim_pred = crnn.predict(crnn_infer_img)
        print('%-20s => %-20s' % (raw_pred, sim_pred))
        if args.output_folder:
            cv2.putText(img_rec, sim_pred, (x,y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
    
    if args.output_folder:
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
        cv2.drawContours(img_rec, contours, -1, (22,222,22), 1, cv2.LINE_AA)
        cv2.imwrite(args.output_folder + '/predict.png', img_rec)


    