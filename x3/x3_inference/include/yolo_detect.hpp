#ifndef __YOLOV5_DETECT__
#define __YOLOV5_DETECT__
#include <math.h>
#include <algorithm>
#include "hb_dnn.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "detect.h"

using namespace detection;

class YoloDetect : public Detect{
 public:
    YoloDetect(int num_classes,int rows, int cols);
    virtual void loadModel(std::string modelPath) override;
    virtual std::vector<float*> inference() override;
    virtual void decodeBbox(int imgo_w, int imgo_h,float confidence_threshold,bool with_seg) override;
};
#endif

YoloDetect::YoloDetect(int num_classes,int rows, int cols){
    std::cout << "YoloDetect  initialize" << std::endl;
    this->num_classes = num_classes;
    this->cols = cols;
    this->rows = rows;

  };

void YoloDetect::loadModel(std::string modelPath) {
  yolo_infer_ = new Inference(modelPath);
  det_input_h = yolo_infer_->input_h;
  det_input_w = yolo_infer_->input_w;
}

std::vector<float*> YoloDetect::inference() {
  detRes_ = yolo_infer_->inference();
  return detRes_;
}

void YoloDetect::decodeBbox(int imgo_w, int imgo_h,float confidence_threshold,bool with_seg) {
  boxes_infer.clear();
  for (int i = 0; i < rows; ++i) {
    float* pitem = detRes_[0] + i * cols;
    float objness = pitem[4];
    if (objness < confidence_threshold)
        continue;
    
    float *pclass = pitem + 5;
    int label = std::max_element(pclass, pclass + num_classes) - pclass;
    float prob = pclass[label];
    float confidence = prob * objness;

    if (confidence < confidence_threshold) continue;
    
    float cx     = pitem[0];
    float cy     = pitem[1];
    float width  = pitem[2];
    float height = pitem[3];
    float left = (cx - width * 0.5);
    float top    = (cy - height * 0.5);
    float right  = (cx + width * 0.5);
    float bottom = (cy + height * 0.5);

    if(with_seg){
        std::vector<float> temp_proto(pitem + 5 + num_classes, pitem + 5 + num_classes + 32);
        Matrix tmp_cof(1, 32, temp_proto);
        boxes_infer.emplace_back(left, top, right, bottom, confidence, (float)label);
    }
    else{
        boxes_infer.emplace_back(left, top, right, bottom, confidence, (float)label);
    }


  }
  
}


