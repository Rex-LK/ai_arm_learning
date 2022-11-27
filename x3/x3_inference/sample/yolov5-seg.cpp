
#include <iostream>
#include <string>
#include "detect.h"
#include "yolo_detect.hpp"
#include "common.hpp"
using namespace std;
using namespace cv;
using namespace detection;
using namespace tools;
using namespace matrix;

static const int INPUT_H = 640;
static const int INPUT_W = 640;
static const int segWidth = 160;
static const int segHeight = 160;
static const int segChannels = 32;
static const int CLASSES = 80;
static const int Box_col = 117;
static const int Num_box = 25200;
static const int OUTPUT_SIZE = Num_box * (CLASSES+5 + segChannels);  // det output
static const int OUTPUT_SIZE1 = segChannels * segWidth * segHeight ;//seg output

static const float CONF_THRESHOLD = 0.1;
static const float NMS_THRESHOLD = 0.5;
static const float MASK_THRESHOLD = 0.5;

void DrawPred(Mat& img,std::vector<Bbox> result) {
	std::vector<Scalar> color;
	srand(time(0));
    for (int i = 0; i < CLASSES; i++)
    {
        int b = rand() % 256;
		int g = rand() % 256;
		int r = rand() % 256;
		color.push_back(Scalar(b, g, r));
    }
    Mat mask = img.clone();
    for (int i = 0; i < result.size(); i++) {
		int left, top;
		left = result[i].box.x;
		top = result[i].box.y;
		int color_num = i;
		rectangle(img, result[i].box,color[result[i].class_label], 2, 8);
        cv::Mat c = mask(result[i].box);

        cv::Mat a = result[i].boxMask;

        c.setTo(color[result[i].class_label], a);
        std::string label = std::to_string(result[i].class_label) + ":" + std::to_string(result[i].confidence);
        int baseLine;
		Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		top = max(top, labelSize.height);
		putText(img, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 1, color[result[i].class_label], 2);
	}
	addWeighted(img, 0.5, mask, 0.5, 0, img);

};


int main(int argc, char* argv[]) {
    string image_path = "../data/images/street.jpg";

    Mat imageOri = imread(image_path);

    if(imageOri.empty()){
        cout<<"image is none"<<endl;
        return 0;
    }

    string model_root_path = "../data/model_zoo/";
    int img_oh = imageOri.rows;
    int img_ow = imageOri.cols;
    Detect *det;
    
    string model_name = "yolov5n-seg"; 

    det = new YoloDetect(CLASSES,Num_box,CLASSES + 5 + segChannels);

    det->loadModel(model_root_path + model_name + ".bin");
    float confidence_threshold = 0.4;

    Mat bgrInferMat = imageOri.clone();
    resize(bgrInferMat, bgrInferMat, Size(INPUT_W, INPUT_H));
    // nv12 推理
    imgOp::bgr_2_tensor_as_nv12(bgrInferMat,det->yolo_infer_->input_tensors.data(),INPUT_H,INPUT_W);

    vector<float*> detRes = det->inference();
    det->decodeBbox(img_ow, img_oh, confidence_threshold,true);
    vector<Bbox> boxes_res = nms(det->boxes_infer, 0.4);

    float *seg_det = detRes[1];
    vector<float> mask(seg_det, seg_det + segChannels * segWidth * segHeight);

    Matrix seg_proto(segChannels, segWidth * segHeight, mask);

    for (int i = 0; i < boxes_res.size(); ++i)
    {
        
        Matrix resSeg = (mygemm(boxes_res[i].mask_cofs,seg_proto).exp(-1) + 1.0).power(-1);

        Mat resMat(resSeg.data_);


        resMat = resMat.reshape(0, {segHeight, segWidth});
        // 如果图片预处理为直接resize,那么计算出来的resMat可以直接缩放回原图，
        // 如果是填充黑边的resize，可以参考原代码将原型mask恢复到原图大小
        resize(resMat, resMat, Size(INPUT_H,INPUT_W), INTER_NEAREST);
        // 获取原型mask里面目标框的区域
        Rect temp_rect = boxes_res[i].box;
        // 将目标框区域 大于0.5的值变为255
        cv::Mat binaryMat;
        inRange(resMat(temp_rect), 0.5, 1, binaryMat);
		boxes_res[i].boxMask = binaryMat;
        // cv::imwrite(to_string(i) + "_.jpg", b);
    }
    DrawPred(bgrInferMat, boxes_res);
    cv::imwrite("output-seg.jpg", bgrInferMat);
}
