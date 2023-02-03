
import cv2
import numpy as np
import torch
from torch.autograd import Variable
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

class rec_model:
    def __init__(self, model_path,converter):
        
        self.model = pyeasy_dnn.load(model_path)

        self.converter = converter

        h, w = get_hw(self.model[0].inputs[0].properties)
        self.des_dim = (w, h)
        

    def predict(self,img):
        image_input = cv2.resize(img, self.des_dim, interpolation=cv2.INTER_AREA)
        image_input = bgr2nv12_opencv(image_input)

        preds = self.model[0].forward(image_input)
        preds = preds[0].buffer[...,0]
        print(preds.shape)
        preds = torch.from_numpy(preds)
        preds = preds.permute(1,0,2)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        raw_pred = self.converter.decode(preds.data, preds_size.data, raw=True)
        sim_pred = self.converter.decode(preds.data, preds_size.data, raw=False)
        return raw_pred,sim_pred
        

if __name__ == '__main__':
    import utils

    img_path = '0.png'
    model_path = 'rec_static.onnx.bin'
    # 最后面必须加一个空格，因为可以检测空格
    alphabet = """0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~!"#$%&'()*+,-./  """
    converter = utils.strLabelConverter(alphabet)   
    
    img0 = cv2.imread(img_path)
    print(img0.shape)
    rec = rec_model(model_path,converter)
    raw_pred,sim_pred = rec.predict(img0)
    print('%-20s => %-20s' % (raw_pred, sim_pred))

