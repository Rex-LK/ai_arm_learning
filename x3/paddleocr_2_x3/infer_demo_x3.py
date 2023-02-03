import numpy as np
import cv2
import argparse
import pyclipper
import torch
from torch.autograd import Variable
import os

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
    def __init__(self, model_path,infer_type):
        self.infer_type = infer_type
        if self.infer_type == "onnx":
            import onnxruntime
            self.model = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        elif self.infer_type == "x3":
            from hobot_dnn import pyeasy_dnn
            self.model = pyeasy_dnn.load(model_path)
            h, w = get_hw(self.model[0].inputs[0].properties)
            self.des_dim = (w, h)
        self.thr = 0.5
        self.ratio_prime = 2
        

    def predict(self, img, d_size = (640,640), min_area: int = 100):

        img0_h,img0_w = img.shape[:2]
        if self.infer_type == "onnx":
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            image_input = cv2.resize(img, d_size) 
            image_input = image_input[..., ::-1]  # bgr -> rgb
            image_input = (image_input / 255.0 - mean) / std
            image_input = image_input.astype(np.float32)
            image_input = image_input.transpose(2, 0, 1)
            image_input = np.ascontiguousarray(image_input)
            image_input = image_input[None, ...]
            preds = self.model.run(["sigmoid_0.tmp_0"], {"x": image_input})[0]
        elif self.infer_type == "x3":
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


class rec_model:
    def __init__(self, model_path,converter,infer_type):
        self.infer_type = infer_type
        if self.infer_type == "onnx":
            import onnxruntime
            self.model = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        elif self.infer_type == "x3":
            from hobot_dnn import pyeasy_dnn
            self.model = pyeasy_dnn.load(model_path)
            h, w = get_hw(self.model[0].inputs[0].properties)
            self.des_dim = (w, h)
        self.converter = converter
        
    def preprocess(self,image_o, image_d_size, imagenet_mean = [0.5], imagenet_std = [0.5]):
        if self.infer_type == "onnx":
            image_input = cv2.resize(image_o, image_d_size)  # resize
            image_input = image_input[..., ::-1]  # bgr -> rgb
            image_input = (image_input / 255.0 - imagenet_mean) / imagenet_std  # normalize
            image_input = image_input.astype(np.float32)  # float64 -> float32
            image_input = np.ascontiguousarray(image_input)  # contiguous array memory
            image_input = image_input.transpose(2, 0, 1)
            image_input = np.ascontiguousarray(image_input)
            image_input = image_input[None, ...]
            return image_input
        elif self.infer_type == "x3":
            image_input = cv2.resize(image_o, self.des_dim, interpolation=cv2.INTER_AREA)
            image_input = bgr2nv12_opencv(image_input)
            return image_input

    def predict(self,img):
        image_input = self.preprocess(img,(320,48))
        if self.infer_type == "onnx":
            preds = self.model.run(["softmax_2.tmp_0"], {"x": image_input})[0]
        elif self.infer_type == "x3":
            preds = self.model[0].forward(image_input)
            preds = preds[0].buffer
        preds = torch.from_numpy(preds)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        raw_pred = self.converter.decode(preds.data, preds_size.data, raw=True)
        sim_pred = self.converter.decode(preds.data, preds_size.data, raw=False)
        return raw_pred,sim_pred

class Run:
    def __init__(self, args):
        self.args = args
        self.det_net = det_model(self.args.det_model_path,self.args.infer_type) 
        alphabet = """0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~!"#$%&'()*+,-./  """
        converter = utils.strLabelConverter(alphabet)    
        self.rec_net = rec_model(self.args.rec_model_path,converter,args.infer_type)
    
    def run(self,img0):

        img_rec = img0.copy()
        img0_h,img0_w = img0.shape[:2]
        
        contours, boxes_list = self.det_net.predict(img0)
        
        for i,box in enumerate(boxes_list):
            mask_t = np.zeros((img0_h, img0_w), dtype=np.uint8)
            cv2.fillPoly(mask_t, [box], (255), 8, 0)
            pick_img = cv2.bitwise_and(img0, img0, mask=mask_t)
            x, y, w, h = cv2.boundingRect(box)
            rec_infer_img =  pick_img[y:y+h,x:x+w,:]
            raw_pred,sim_pred = self.rec_net.predict(rec_infer_img)
            print('%-20s => %-20s' % (raw_pred, sim_pred))
            if self.args.output_folder:
                cv2.putText(img_rec, sim_pred, (x,y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
        
        if self.args.output_folder:
            if not os.path.exists(self.args.output_folder):
                os.mkdir(self.args.output_folder)
            imgc = img0.copy()
            cv2.drawContours(imgc, contours, -1, (22,222,22), 1, cv2.LINE_AA)
            cv2.imwrite(self.args.output_folder + '/contour.png', imgc)
            img_draw = draw_bbox(img_rec, boxes_list)
            cv2.imwrite(self.args.output_folder + '/predict.jpg', img_draw)

def init_args():
    parser = argparse.ArgumentParser(description='paddleocr')
    parser.add_argument('--infer_type', choices=["onnx","x3"],default='onnx', type=str)
    parser.add_argument('--det_model_path', default='/data/cv_demo/x3j/work/ai_arm_learning/data/paddleocr_model/model/det_static.onnx', type=str)
    parser.add_argument('--rec_model_path', default='/data/cv_demo/x3j/work/ai_arm_learning/data/paddleocr_model/model/rec_static.onnx', type=str)
    parser.add_argument('--image_path', default='word.jpg', type=str, help='img path for predict')
    parser.add_argument('--show', default=True, type=bool)
    parser.add_argument('--output_folder', default='./output', type=str, help='img path for output')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import utils
    args = init_args()
    # 最后面必须加空格
    img0 = cv2.imread(args.image_path) 
    work = Run(args)
    work.run(img0)


    