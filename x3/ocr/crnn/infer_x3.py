
import cv2
import utils
import numpy as np
import torch
from torch.autograd import Variable
from hobot_dnn import pyeasy_dnn

class crnn_model:
    def __init__(self,model_path,converter):
        self.model = pyeasy_dnn.load(model_path)
        self.converter = converter
    def predict(self,img):
        img = cv2.resize(img,(100,32))
        preds = self.model[0].forward(img)[0].buffer
        preds = preds[...,0]
        preds = torch.from_numpy(preds)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        raw_pred = self.converter.decode(preds.data, preds_size.data, raw=True)
        sim_pred = self.converter.decode(preds.data, preds_size.data, raw=False)
        return raw_pred,sim_pred
if __name__ == '__main__':
    import utils
    img_path = './data/demo.png'
    img0 = cv2.imread(img_path,0)
    model_path = 'crnn_simp.bin'
    alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
    converter = utils.strLabelConverter(alphabet) 
    crnn = crnn_model(model_path,converter)
    raw_pred,sim_pred = crnn.predict(img0)
    print('%-20s => %-20s' % (raw_pred, sim_pred))


