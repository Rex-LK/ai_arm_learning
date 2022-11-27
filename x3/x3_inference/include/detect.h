#ifndef __DETECT__
#define __DETECT__

#include<iostream>
#include<string>
#include<vector>
#include "inference.hpp"
#include "common.hpp"

class Detect {
 public:
    Detect(){};
    virtual void   loadModel(std::string modelPath) = 0;
    virtual std::vector<float*>  inference() = 0;
    virtual void   decodeBbox(int imgo_w, int imgo_h,float confidence_threshold) = 0;
    int det_input_h;
    int det_input_w;

    int num_classes;
    int cols;
    int rows;
    std::vector<detection::Bbox> boxes_infer;
    Inference *yolo_infer_;
    std::vector<float *>detRes_;
};
#endif

