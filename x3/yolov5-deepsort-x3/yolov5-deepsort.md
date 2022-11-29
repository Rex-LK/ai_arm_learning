

## [旭日x3] 动手实践之yolov5-deepsort  python测试

### 1、前言

最近在x3上尝试了不同的算法，但是多目标跟踪的算法一直没有进行尝试，这几天最终选择了yolov5-deepsort来进行算法移植测试，在移植的同时，同时复习一下deepsort的实现流程。

yolov5-deepsort地址: [yolov5-deepsort](https://github.com/Sharpiless/yolov5-deepsort/)

本文测试代码地址: https://github.com/Rex-LK/ai_arm_learning/tree/master/x3/yolov5-deepsort-x3

完整测试代码、视频以及bin模型百度云链接: https://pan.baidu.com/s/19rabju72cYRZNwNCBpjIXQ?pwd=iyg8 提取码: iyg8

### 2、原代码测试

#### 2.1 torch环境下测试

在x3上运行前，首先要保证在pytorch的环境下运行没问题，在clone代码下来后，修改demo.py 里面的视频路径，然后运行

```
python demo.py 
```

下载代码库里面自带yolov5模型以及deepsort模型，直接运行的话，除了库的问题，另外会报下面错两个错误，修改对应的地方的代码就好了。

- 问题1

![Screenshot from 2022-11-28 20-56-23](Screenshot%20from%202022-11-28%2020-56-23.png)

修改为:

![Screenshot from 2022-11-28 20-56-56](Screenshot%20from%202022-11-28%2020-56-56.png)

- 问题2

![Screenshot from 2022-11-28 20-57-15](Screenshot%20from%202022-11-28%2020-57-15.png)

修改 /home/rex/miniconda3/lib/python3.9/site-packages/torch/nn/modules/upsampling.py 

![Screenshot from 2022-11-28 20-57-42](Screenshot%20from%202022-11-28%2020-57-42.png)

#### 2.2、导出onnx

需要导出两个onnx模型，yolov5检测模型以及deepsort用到的特征提取模型，分别在下面这两个地方添加上导出onnx的代码。

- yolov5s导出的onnx 模型，修改AIDetector_pytorch.py 里的 init_model 函数:

```python
def init_model(self):
    self.weights = 'weights/yolov5s.pt'
    self.device = 'cpu'
    self.device = select_device(self.device)
    model = attempt_load(self.weights, map_location=self.device)
    model.to(self.device).eval()
    model.float()
    # torch.save(model, 'test.pt')\\

    dummy = torch.zeros(1, 3, 384, 640).float()
    torch.onnx.export(
        model, (dummy,),
        "deepsort_yolov5.onnx",
        input_names=["image"],
        output_names=["output"],
        opset_version=11
    )

    self.m = model
    self.names = model.module.names if hasattr(
        model, 'module') else model.names

def preprocess(self, img):
    # 因为实在cpu上运行的，不支持half()
    img = img.float()  # 半精度
```

导出deepsort的onnx模型，需要修改deep_sort/deep_sort/deep/feature_extractor.py 的Extractor 的 init以及call函数

```python
def __init__:
	self.device = "cpu"

def __call__(self, im_crops):
    im_batch = self._preprocess(im_crops)
    with torch.no_grad():
    im_batch = im_batch.to(self.device)
    features = self.net(im_batch)
    torch.onnx.export(
        self.net, (im_batch,),
        "deepsort_feature.onnx",
        input_names=["image"],
        output_names=["output"],
        opset_version=11
    )
    return features.cpu().numpy()
```

修改完毕后再运行python demo.py ,在根目录就生成了 两个onnx模型，导出完毕后推荐用onnxsim 调整一下，可以避免后续的一些错误。

### 3、 模型量化

#### 3.1 yolov5s量化

yolov5s量化过程可以参考之前的文章:https://developer.horizon.ai/forumDetail/118363914936418940

#### 3.2 deepsort模型量化

由于deepsort的特征提取模型的输入大小为 h=128,w=64，就需要在先修改preprocess.py里面的   target_size=(128,64)，然后运行sh 02_preprocess.sh，然后利用03_03_build.sh来进行模型的量化。deepsort特征提取模型的配置文件如下:

```yaml
model_parameters:
  onnx_model: 'deepsort_feature.onnx'
  output_model_file_prefix: 'deepsort_feature'
  march: 'bernoulli2'
input_parameters:
  input_type_train: 'rgb'
  input_layout_train: 'NCHW'
  input_type_rt: 'nv12'
  norm_type: 'data_mean_and_scale'
  mean_value: '123.675 116.28 103.53'
  scale_value: '58.395 57.12 57.375'
  input_layout_rt: 'NHWC'
calibration_parameters:
  cal_data_dir: './calibration_data_rgb_f32'
  calibration_type: 'max'
  max_percentile: 0.9999
compiler_parameters:
  compile_mode: 'latency'  
  optimize_level: 'O3'
  debug: False
  core_num: 2 
```

### 4、x3上运行yolov5-deepsort

准备好上面的两个bin文件后，就可以在x3上进行视频测试了，下面简要展示一下yolo目标检测以及deepsort特征提取的部分代码，目标检测的模型在AIDetector_x3.py

```python
# yolo5目标检测
class Detector(baseDet):

    def __init__(self):
        super(Detector, self).__init__()
        self.init_model()
        self.build_config()
	
    def init_model(self):
        # 加载模型
        self.m =  dnn.load('deepsort_yolov5.bin')

    def preprocess(self, img):
        img0 = img.copy()
        img = cv2.resize(img,(640,384))
        nv12_data = bgr2nv12_opencv(img)
        return img0, nv12_data

    def detect(self, im):
        im0, nv12_data = self.preprocess(im)
        outputs = self.m[0].forward(nv12_data)
        pred = outputs[0].buffer
        pred = pred[:,:,:,0]
        pred = torch.from_numpy(pred)
        # nms
        pred = non_max_suppression(pred, self.threshold, 0.4)
        pred_boxes = []
        # 这里模型的输入为384，w为640
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(
                   (384,640), det[:, :4], im0.shape).round()
                for *x, conf, cls_id in det:
                    lbl = self.names[int(cls_id)]
                    # 所需要的类别
                    if not lbl in ['person', 'car', 'truck']:
                        continue
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    pred_boxes.append(
                        (x1, y1, x2, y2, lbl, conf))

        return im, pred_boxes
```

利用yolov5检测出特定的目标框后，就可以利用deepsort的特征提取模型结合卡尔曼滤波对目标进行跟踪了，特征提取实现路径为deep_sort/deep_sort/deep/feature_extractor.py

```python
class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.net = dnn.load('deepsort_feature.bin')
        self.size = (64, 128)
    def _preprocess(self, im_crops):
        def _resize(im, size):
            return cv2.resize(im, size)
        ims = [_resize(im, self.size) for im in im_crops]
        return ims
    def __call__(self, im_crops):
        '''
        原代码是利用动态batch进行推理的，x3目前只能使用固定的batch，于是这个采用的是batch=1进行推理
        '''
        im_batch = self._preprocess(im_crops)
        sample = np.empty([len(im_batch),512])
        for i,im in enumerate(im_batch):
            nv12_data = bgr2nv12_opencv(im)
            outputs = self.net[0].forward(nv12_data)
            pred = outputs[0].buffer
            pred = pred[:,:,0,0]
            sample[i,:] = pred
        return sample
```

### 5、测试结果

从测试的准确度上来看应该是不错的，但是速度的话还有很大的提升空间，下面为测试结果。

![Screenshot from 2022-11-29 21-43-25](Screenshot%20from%202022-11-29%2021-43-25.png)

### 6、总结

本次在x3上实现了yolov5-deepsort多目标跟踪的算法实现，进一步加深了在x3上对算法移植的熟练度，同时又复习了一遍deepsort的实现流程，后续希望有机会能够进行更加深入的学习。



