import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging
from .model import Net
from hobot_dnn import pyeasy_dnn as dnn

def bgr2nv12_opencv(image):
    height, width = image.shape[0], image.shape[1]
    area = height * width
    yuv420p = cv2.cvtColor(image, cv2.COLOR_RGB2YUV_I420).reshape((area * 3 // 2,))
    y = yuv420p[:area]
    uv_planar = yuv420p[area:].reshape((2, area // 4))
    uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))

    nv12 = np.zeros_like(yuv420p)
    nv12[:height * width] = y
    nv12[height * width:] = uv_packed
    return nv12
class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.net =dnn.load('deepsort_feature.bin')
        self.size = (64, 128)

    def _preprocess(self, im_crops):
        def _resize(im, size):
            return cv2.resize(im, size)
        ims = [_resize(im, self.size) for im in im_crops]
        return ims


    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        sample = np.empty([len(im_batch),512])
        for i,im in enumerate(im_batch):
            nv12_data = bgr2nv12_opencv(im)
            outputs = self.net[0].forward(nv12_data)
            pred = outputs[0].buffer
            pred = pred[:,:,0,0]
            sample[i,:] = pred
        return sample
        # return features.cpu().numpy()

if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:,:,(2,1,0)]
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(img)
    print(feature.shape)

