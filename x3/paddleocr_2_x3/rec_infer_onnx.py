
import onnxruntime
import cv2
import numpy as np
import torch
from torch.autograd import Variable



class rec_model:
    def __init__(self, model_path,converter):
        
        self.model = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.converter = converter
        
    def preprocess_gray(self,image_o, image_d_size, imagenet_mean = [0.5], imagenet_std = [0.5]):
        image_input = cv2.resize(image_o, image_d_size)  # resize
        image_input = image_input[..., ::-1]  # bgr -> rgb
        image_input = (image_input / 255.0 - imagenet_mean) / imagenet_std  # normalize
        image_input = image_input.astype(np.float32)  # float64 -> float32
        image_input = np.ascontiguousarray(image_input)  # contiguous array memory
        image_input = image_input.transpose(2, 0, 1)
        image_input = np.ascontiguousarray(image_input)
        image_input = image_input[None, ...]
        return image_input

    def predict(self,img):
        image_input = self.preprocess_gray(img,(320,48))
        preds = self.model.run(["softmax_2.tmp_0"], {"x": image_input})[0]
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
    model_path = 'model/rec_static.onnx'
    # 最后面必须加一个空格，因为可以检测空格
    alphabet = """0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~!"#$%&'()*+,-./  """
    converter = utils.strLabelConverter(alphabet)   
    
    img0 = cv2.imread(img_path)
    print(img0.shape)
    rec = rec_model(model_path,converter)
    raw_pred,sim_pred = rec.predict(img0)
    print('%-20s => %-20s' % (raw_pred, sim_pred))

