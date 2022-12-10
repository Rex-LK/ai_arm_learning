
import numpy as np
import time
import cv2
import torch
import onnxruntime

import pyclipper
import onnxruntime



class dbnet_model:
    def __init__(self, model_path):
        
        self.model = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.thr = 0.5
        self.ratio_prime = 1
        

    def predict(self, img, d_size = (640,640), min_area: int = 100):
        img0_h,img0_w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        img = cv2.resize(img,d_size)
        img = img / 255
        img = img.astype(np.float32)
        img = img.transpose(2,0,1)
        img = np.ascontiguousarray(img)
        img = img[None, ...]
        preds = self.model.run(["output"], {"image": img})[0]
        preds = torch.from_numpy(preds[0])
        scale = (preds.shape[2] / img0_w, preds.shape[1] / img0_h)
        '''inference'''
        start = time.time()
        prob_map, thres_map = preds[0], preds[1]
        out = (prob_map > self.thr).float() * 255
        out = out.data.cpu().numpy().astype(np.uint8)
        contours, hierarchy = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [(i / scale).astype(np.int) for i in contours if len(i)>=4] 
        dilated_polys = []
        for poly in contours:
            poly = poly[:,0,:]
            D_prime = cv2.contourArea(poly) * self.ratio_prime / cv2.arcLength(poly, True) # formula(10) in the thesis
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            dilated_poly = np.array(pco.Execute(D_prime))
            if dilated_poly.size == 0 or dilated_poly.dtype != np.int or len(dilated_poly) != 1:
                continue
            dilated_polys.append(dilated_poly)
            
        boxes_list = []
        for cnt in dilated_polys:
            if cv2.contourArea(cnt) < min_area:
                continue
            rect = cv2.minAreaRect(cnt)
            box = (cv2.boxPoints(rect)).astype(np.int)
            boxes_list.append(box)
        t = time.time() - start
        boxes_list = np.array(boxes_list)
        return dilated_polys, boxes_list, t




if __name__ == '__main__':
    from utils.util import show_img, draw_bbox
    model_path = './dbnet_simp.onnx'    
    
    img_path = "test_images/dbtest.png"
    img0 = cv2.imread(img_path) 
    
    # 初始化网络
    model = dbnet_model(model_path) 
    contours, boxes_list, t = model.predict(img0)
    print('Time: %.4f' %t)
 
    img = cv2.imread(img_path)[:, :, ::-1]
    imgc = img.copy()
    cv2.drawContours(imgc, contours, -1, (22,222,22), 2, cv2.LINE_AA)
    cv2.imwrite('contour.png', imgc)
    img = draw_bbox(img, boxes_list)
    cv2.imwrite('predict.jpg', img)