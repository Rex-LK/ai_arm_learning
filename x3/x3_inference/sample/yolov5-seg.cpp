
#include <iostream>
#include <string>
#include "detect.h"
#include "yolo_detect.hpp"
using namespace std;
using namespace cv;
using namespace detection;
using namespace tools;


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
    int num_classes = 80;
    int cols = 117;
    int rows = 25200;

    det = new YoloDetect(num_classes,rows,cols);

    det->loadModel(model_root_path + model_name + ".bin");
    float confidence_threshold = 0.4;

    int input_w = 640;
    int input_h = 640;
    Mat bgrInferMat = imageOri.clone();
    resize(bgrInferMat, bgrInferMat, Size(input_w, input_h));
    // nv12 推理
    imgOp::bgr_2_tensor_as_nv12(bgrInferMat,det->yolo_infer_->input_tensors.data(),input_h,input_w);

    vector<float*> detRes = det->inference();
    det->decodeBbox(img_ow, img_oh, confidence_threshold);
    vector<Bbox> boxes_res = nms(det->boxes_infer, 0.4);

    

    // string save_det_path ="test.png";
    // for (auto& box : boxes_res) {
    //     string label_name = coco_classes[box.class_label];
    //     rectangle(bgrInferMat, Point(box.left, box.top),
    //                   Point(box.right, box.bottom), Scalar(0, 255, 0), 2);
    //     putText(bgrInferMat, format("%.2f", box.confidence),
    //                 Point(box.left, box.top - 10), 0, 0.8,
    //                 Scalar(0, 0, 255), 2, 2);
    //     putText(bgrInferMat, label_name, Point(box.left, box.top + 10), 0,
    //                 0.8, Scalar(0, 0, 255), 2, 2);
    // }
    // imwrite(save_det_path, bgrInferMat);
    // return 0;
}
