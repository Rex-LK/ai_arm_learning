import cv2

import numpy as np
import cv2

import pyclipper
from hobot_dnn import pyeasy_dnn


def get_hw(pro):
    if pro.layout == "NCHW":
        return pro.shape[2], pro.shape[3]
    else:
        return pro.shape[1], pro.shape[2]
    
def bgr2nv12_opencv(image):
    height, width = image.shape[0], image.shape[1]
    area = height * width
    yuv420p = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
    y = yuv420p[:area]
    uv_planar = yuv420p[area:].reshape((2, area // 4))
    uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))

    nv12 = np.zeros_like(yuv420p)
    nv12[:height * width] = y
    nv12[height * width:] = uv_packed
    return nv12

def draw_bbox(img_path, result, color=(128, 240, 128), thickness=3):
    if isinstance(img_path, str):
        img_path = cv2.imread(img_path)
        # img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    img_path = img_path.copy()
    for point in result:
        point = point.astype(int)
        cv2.line(img_path, tuple(point[0]), tuple(point[1]), color, thickness)
        cv2.line(img_path, tuple(point[1]), tuple(point[2]), color, thickness)
        cv2.line(img_path, tuple(point[2]), tuple(point[3]), color, thickness)
        cv2.line(img_path, tuple(point[3]), tuple(point[0]), color, thickness)
    return img_path

class det_model:
    def __init__(self, model_path):
        self.model = pyeasy_dnn.load(model_path)
        self.thr = 0.5
        self.ratio_prime = 3.5
        h, w = get_hw(self.model[0].inputs[0].properties)
        self.des_dim = (w, h)

    def predict(self, img, d_size = (640,640), min_area: int = 100):
        
        img0_h,img0_w = img.shape[:2]
        image_input = cv2.resize(img, self.des_dim, interpolation=cv2.INTER_AREA)
        image_input = bgr2nv12_opencv(image_input)
        preds = self.model[0].forward(image_input)
        preds = preds[0].buffer

        preds = np.where(preds[0][0]>0.5,255,0)
        preds = preds.astype(np.uint8)
        preds = cv2.resize(preds,(img0_w,img0_h))
        
        contours, hierarchy = cv2.findContours(preds, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        
        dilated_polys = []
        for poly in contours:
            poly = poly[:,0,:]
            D_prime = cv2.contourArea(poly) * self.ratio_prime / cv2.arcLength(poly, True) # formula(10) in the thesis
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            dilated_poly = np.array(pco.Execute(D_prime))
            if dilated_poly.size == 0 or dilated_poly.dtype != np.int_ or len(dilated_poly) != 1:
                continue
            dilated_polys.append(dilated_poly)
            
        boxes_list = []
        for cnt in dilated_polys:
            if cv2.contourArea(cnt) < min_area:
                continue
            rect = cv2.minAreaRect(cnt)
            box = (cv2.boxPoints(rect)).astype(np.int_)
            boxes_list.append(box)
        
            
        boxes_list = np.array(boxes_list)
        return dilated_polys, boxes_list




if __name__ == '__main__':

    model_path = 'det_static.bin'    
    
    img_path = "word.jpg"
    img0 = cv2.imread(img_path) 
    
    # 初始化网络
    model = det_model(model_path) 
    contours, boxes_list= model.predict(img0)
 
    img = cv2.imread(img_path)[:, :, ::-1]
    imgc = img.copy()
    cv2.drawContours(imgc, contours, -1, (22,222,22), 2, cv2.LINE_AA)
    cv2.imwrite('contour.png', imgc)
    img = draw_bbox(img, boxes_list)
    cv2.imwrite('predict.jpg', img)




