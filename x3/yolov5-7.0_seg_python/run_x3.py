import numpy as np
import cv2
import time
import argparse
from asyncio.log import logger
import logging
from hobot_dnn import pyeasy_dnn as dnn
from scipy import ndimage
 
LOGGER = logging.getLogger("v5-seg")

 
class Yolov5_7_Seg:
 
    def __init__(self,
                 model_path,
                 conf_thres=0.25,
                 iou_thres=0.45,
                 max_det=1000):
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
 
        self.models = dnn.load(self.model_path)
        
        self.net_input_height, self.net_input_width = self.get_hw(self.models[0].inputs[0].properties)
    

    def get_hw(self,pro):
        if pro.layout == "NCHW":
            return pro.shape[2], pro.shape[3]
        else:
            return pro.shape[1], pro.shape[2]

    def letterbox(self,
                  color=(114, 114, 114),
                  auto=False,
                  scaleFill=False,
                  scaleup=True,
                  stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = self.image0.shape[:2]  # current shape [height, width]
 
        new_shape = (self.net_input_height, self.net_input_width)
 
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)
 
        ratio = r, r 
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[
            1]  
        if auto: 
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  
        elif scaleFill:  
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[
                0]  
 
        dw /= 2  
        dh /= 2
 
        if shape[::-1] != new_unpad:  
            im = cv2.resize(self.image0,
                            new_unpad,
                            interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im,
                                top,
                                bottom,
                                left,
                                right,
                                cv2.BORDER_CONSTANT,
                                value=color) 
        return im, ratio, (dw, dh)
 
    def box_area(self, box):
        return (box[2] - box[0]) * (box[3] - box[1])
 
    def box_iou(self, box1, box2, eps=1e-7):
        (a1, a2), (b1, b2) = np.split(box1[:, None], 2,
                                      axis=2), np.split(box2, 2, axis=1)
        array = np.minimum(a2, b2) - np.maximum(a1, b1)
        inter = array.clip(0)
        inter = inter.prod(2)
 
        return inter / (self.box_area(box1.T)[:, None] +
                        self.box_area(box2.T) - inter + eps)
 
    def xywh2xyxy(self, x):
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2 
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y
 
    def sigmoid(self, x):
        s = 1 / (1 + np.exp(-x))
        return s
 
    def non_max_suppression(self,
                            prediction,
                            conf_thres=0.25,
                            iou_thres=0.45,
                            max_det=300,
                            nm=0):
 
        bs = prediction.shape[0] 
        nc = prediction.shape[2] - nm - 5
        xc = prediction[..., 4] > conf_thres 

        max_wh = 7680  
        max_nms = 30000 
        time_limit = 0.5 + 0.05 * bs 
        redundant = True  
        merge = False  
 
        t = time.time()
        mi = 5 + nc 
 
        output = []
        output_final = []
        for xi, x in enumerate(prediction):  
            x = x[xc[xi]]
 
            if not x.shape[0]:
                continue
 
            x[:, 5:] *= x[:, 4:5]  
            box = self.xywh2xyxy(x[:, :4])  
            mask = x[:, mi:]  
            conf, j = x[:, 5:mi].max(axis=1), x[:, 5:mi].argmax(axis=1)
            conf, j = conf.reshape(-1, 1), j.reshape(-1, 1)
            x = np.concatenate((box, conf, j.astype(float), mask),
                               axis=1)[conf.reshape(-1) > conf_thres]
            n = x.shape[0] 
            if not n:  
                continue
            elif n > max_nms: 
                x = x[(-x[:, 4]).argsort()[:max_nms]] 
            else:
                x = x[(-x[:, 4]).argsort()] 

            c = x[:, 5:6] * max_wh 
          
            boxes, scores = x[:, :4] + c, x[:, 4]
            i = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(),
                                 self.conf_thres, self.iou_thres)  # NMS
 
            if i.shape[0] > max_det: 
                i = i[:max_det]
            if merge and (1 < n <
                          3E3): 
               
                iou = self.box_iou(boxes[i], boxes) > iou_thres  
                weights = iou * scores[None] 
                x[i, :4] = np.dot(weights, x[:, :4]).astype(
                    np.float32) / weights.sum(1, keepdims=True)  
                if redundant:
                    i = i[iou.sum(1) > 1] 
 
            output.append(x[i])
            if (time.time() - t) > time_limit:
                LOGGER.warning(
                    f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
                break 
 
        output = np.array(output).reshape(-1, 6 + nm)
        output_final.append(output)
 
        return output_final
 
    def clip_boxes(self, boxes, shape):
        if isinstance(boxes, np.ndarray): 
            # np.array (faster grouped)
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])
        else:
            logger.info("type wont be supported")
 
    def scale_boxes(self, img1_shape, boxes, img0_shape, ratio_pad=None):
        if ratio_pad is None:  
            gain = min(img1_shape[0] / img0_shape[0],
                       img1_shape[1] / img0_shape[1]) 
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
                img1_shape[0] - img0_shape[0] * gain) / 2 
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]
 
        boxes[:, [0, 2]] -= pad[0]
        boxes[:, [1, 3]] -= pad[1]
        boxes[:, :4] /= gain
        self.clip_boxes(boxes, img0_shape)
        return boxes
 
    def crop_mask(self, masks, boxes):
        n, h, w = masks.shape
        x1, y1, x2, y2 = np.split(boxes[:, :, None], 4,
                                  axis=1)
        r = np.arange(w, dtype=x1.dtype)[None, None, :]
        c = np.arange(h, dtype=x1.dtype)[None, :, None]
 
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))
 
    def process_mask(self, protos, masks_in, bboxes, shape, upsample=False):
        c, mh, mw = protos.shape 
        ih, iw = shape
        ttt = protos.astype(np.float32).reshape(c, -1)
 
        ppp = masks_in @ ttt
        masks = self.sigmoid(
            (masks_in @ protos.astype(np.float32).reshape(c, -1))).reshape(
                -1, mh, mw)  # CHW
 
        downsampled_bboxes = bboxes.copy()
        downsampled_bboxes[:, 0] *= mw / iw
        downsampled_bboxes[:, 2] *= mw / iw
        downsampled_bboxes[:, 3] *= mh / ih
        downsampled_bboxes[:, 1] *= mh / ih
 
        masks = self.crop_mask(masks, downsampled_bboxes) 
        if upsample:
            masks = ndimage.zoom(masks[None], (1, 1, 4, 4),
                                 order=1,
                                 mode="nearest")[0]

        t = masks.__gt__(0.5).astype(np.float32)
        return t 
 
    def preprocess(self, img_src_bgr):
        self.input_data_shape = [640,640]
        self.image0 = img_src_bgr
        letterbox_img = self.letterbox()[0]
        return letterbox_img

    def detect(self, input_src):
        net_input_data = self.preprocess(input_src)
        nv12_data = self.bgr2nv12_opencv(net_input_data)
        outputs = self.models[0].forward(nv12_data)
        det = outputs[0].buffer[...,0]
        seg = outputs[1].buffer
        
        return [det,seg]

    def bgr2nv12_opencv(self,image):
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
    def postprocess(self, pred_out):
        pred_det, proto = pred_out[0], pred_out[1]
 
        #nms
        pred = self.non_max_suppression(pred_det,
                                        self.conf_thres,
                                        self.iou_thres,
                                        max_det=self.max_det,
                                        nm=32)
 
        for i, det in enumerate(pred):
            if len(det):
                masks = self.process_mask(
                    proto[i],
                    det[:, 6:],
                    det[:, :4],
                    self.input_data_shape,
                    upsample=True) 
                det[:, :4] = self.scale_boxes(self.input_data_shape,
                                              det[:, :4],
                                              self.image0.shape).round()
        np.save("numpy_array_masks.npy", masks)
        return masks, det[:, :6]
 
 
class DRAW_RESULT(Yolov5_7_Seg):
 
    def __init__(self, modelpath, conf_thres, iou_thres):
        super(DRAW_RESULT, self).__init__(modelpath, conf_thres, iou_thres)
 
    def draw_detections(self, image, boxes, scores, class_ids):
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box.astype(int)
 
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
            label = class_id
            label = f'{label} {int(score * 100)}%'
 
            cv2.putText(image,
                        label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0),
                        thickness=2)
 
    def draw_mask(self, masks, colors_, im_src, alpha=0.5):
        if len(masks) == 0:
            return
        if isinstance(masks, np.ndarray):
            masks = np.asarray(masks, dtype=np.uint8)
            masks = np.ascontiguousarray(masks.transpose(1, 2, 0))
        masks = self.scale_image(masks.shape[:2], masks, im_src.shape)
        masks = np.asarray(masks, dtype=np.float32)
        colors_ = np.asarray(colors_, dtype=np.float32) 
        s = masks.sum(2, keepdims=True).clip(0, 1) 
        masks = (masks @ colors_).clip(0, 255)
        im_src[:] = masks * alpha + im_src * (1 - s * alpha)
 
    def scale_image(self, im1_shape, masks, im0_shape, ratio_pad=None):
        if ratio_pad is None: 
            gain = min(im1_shape[0] / im0_shape[0],
                       im1_shape[1] / im0_shape[1]) 
            pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (
                im1_shape[0] - im0_shape[0] * gain) / 2
        else:
            pad = ratio_pad[1]
        top, left = int(pad[1]), int(pad[0])  # y, x
        bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])
 
        if len(masks.shape) < 2:
            raise ValueError(
                f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}'
            )
        masks = masks[top:bottom, left:right]
        masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))
 
        if len(masks.shape) == 2:
            masks = masks[:, :, None]
        return masks
 
 
class Colors:
 
    def __init__(self):
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A',
                '92CC17', '3DDB86', '1A9334', '00D4BB', '2C99A8', '00C2FF',
                '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF',
                'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)
 
    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c
 
    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
 
 
if __name__ == "__main__":
 
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath',
                        type=str,
                        default="bus.jpg",
                        help="image path")
    parser.add_argument('--modelpath',
                        type=str,
                        default="yolov5n-seg.bin",
                        help="model path")
    parser.add_argument('--confThreshold',
                        default=0.25,
                        type=float,
                        help='class confidence')
    parser.add_argument('--nmsThreshold',
                        default=0.45,
                        type=float,
                        help='nms iou thresh')
 
    args = parser.parse_args()
  
    image0 = cv2.imread(args.imgpath)
    v5_7_seg_detector = Yolov5_7_Seg(args.modelpath,
                                 conf_thres=args.confThreshold,
                                 iou_thres=args.nmsThreshold)
    prediction_out = v5_7_seg_detector.detect(image0)
    final_mask, final_det = v5_7_seg_detector.postprocess(prediction_out)
 
    print("start plotting")
 
    d_r = DRAW_RESULT(args.modelpath,
                      conf_thres=args.confThreshold,
                      iou_thres=args.nmsThreshold)
 
    colors_obj = Colors(
    )  # create instance for 'from utils.plots import colors'
    d_r.draw_mask(final_mask,
                  colors_=[colors_obj(x, True) for x in final_det[:, 5]],
                  im_src=image0)
 
    d_r.draw_detections(image0, final_det[:, :4], final_det[:, 4],
                        final_det[:, 5])
 
    cv2.imwrite("masks_det.jpg", image0)