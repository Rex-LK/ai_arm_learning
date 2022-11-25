
#include <iostream>
#include <string>
#include "detect.h"
#include "yolo_detect.hpp"
using namespace std;
using namespace cv;
using namespace detection;
using namespace tools;

// 加入重载
// 加入operateaor
// 加入矩阵相乘
Detect *crate_model(int num_classes, int rows, int cols){
    Detect* det = new YoloDetect(num_classes,rows,cols);
    return det;
}

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
    
    // yolov5、v7 为 25200 、yolov6为8400
    string model_name = "yolov5n"; 
    int num_classes = 80;
    int cols = 85;
    int rows = 0;
    if (model_name == "yolov5n" || model_name == "yolov5s" || model_name == "yolov7-tiny")
    {
         rows = 25200;
    }
    else if(model_name == "yolov6n") {
         rows = 8400;
    }
    // det = new YoloDetect(num_classes,rows,cols);

    det = crate_model(num_classes, rows, cols);

    det->loadModel(model_root_path + model_name + ".bin");
    float confidence_threshold = 0.4;

    int input_w = 640;
    int input_h = 640;
    cv::Mat bgrInferMat = imageOri.clone();
    resize(bgrInferMat, bgrInferMat, cv::Size(input_w, input_h));
    // nv12 推理
    imgOp::bgr_2_tensor_as_nv12(bgrInferMat,det->yolo_infer_->input_tensors.data(),input_h,input_w);

    float* detRes = det->inference();

    det->decodeBbox(img_ow, img_oh,confidence_threshold);

    std::vector<Bbox> boxes_res = nms(det->boxes_infer, 0.4);
    std::string save_det_path ="test.png";

    for (auto& box : boxes_res) {
        std::string label_name = coco_classes[box.class_label];
        cv::rectangle(bgrInferMat, cv::Point(box.left, box.top),
                      cv::Point(box.right, box.bottom), cv::Scalar(0, 255, 0), 2);
        cv::putText(bgrInferMat, cv::format("%.2f", box.confidence),
                    cv::Point(box.left, box.top - 10), 0, 0.8,
                    cv::Scalar(0, 0, 255), 2, 2);
        cv::putText(bgrInferMat, label_name, cv::Point(box.left, box.top + 10), 0,
                    0.8, cv::Scalar(0, 0, 255), 2, 2);
    }
    cv::imwrite(save_det_path, bgrInferMat);
    return 0;
    }
