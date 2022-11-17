#ifndef __COMMON__
#define __COMMON__

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <queue>
#include <utility>
#include <unistd.h>
#include <stdio.h>
#include <sys/time.h>

#include <math.h>
#include <algorithm>

#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "hb_dnn.h"

using namespace std;
using namespace cv;


namespace tools
{   
    float Sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
    float Tanh(float x) { return 2.0f / (1.0f + exp(-2 * x)) - 1; }
    
} 


namespace Det
{
    struct bbox {
    float left, top, right, bottom, confidence;
    int class_label;

    bbox() = default;

    bbox(float left, float top, float right, float bottom, float confidence,
            int class_label)
        : left(left),
            top(top),
            right(right),
            bottom(bottom),
            confidence(confidence),
            class_label(class_label) {}
    };

    auto iou = [](const bbox& a, const bbox& b) {
        float cross_left = std::max(a.left, b.left);
        float cross_top = std::max(a.top, b.top);
        float cross_right = std::min(a.right, b.right);
        float cross_bottom = std::min(a.bottom, b.bottom);

        float cross_area = std::max(0.0f, cross_right - cross_left) *
                            std::max(0.0f, cross_bottom - cross_top);
        float union_area =
            std::max(0.0f, a.right - a.left) * std::max(0.0f, a.bottom - a.top) +
            std::max(0.0f, b.right - b.left) * std::max(0.0f, b.bottom - b.top) -
            cross_area;
        if (cross_area == 0 || union_area == 0) return 0.0f;
        return cross_area / union_area;
    };
    vector<bbox> decodeBbox(float*detRes_ , int rows,int cols, float confidence_threshold,int num_classes) {
            vector<bbox> boxes;
            for(int i = 0; i < rows; ++i){
                float* pitem = detRes_ + i * cols;
                float objness = pitem[4];
                if(objness < confidence_threshold)
                    continue;

                float* pclass = pitem + 5;
                int label     = std::max_element(pclass, pclass + num_classes) - pclass;
                float prob    = pclass[label];
                float confidence = prob * objness;
                if(confidence < confidence_threshold)
                    continue;

                float cx     = pitem[0];
                float cy     = pitem[1];
                float width  = pitem[2];
                float height = pitem[3];

                // 通过反变换恢复到图像尺度
                float left   = (cx - width * 0.5) ;
                float top    = (cy - height * 0.5);
                float right  = (cx + width * 0.5);
                float bottom = (cy + height * 0.5);
                boxes.emplace_back(left, top, right, bottom, confidence, (float)label);
            }
        return boxes;
    }

    vector<bbox> nms(vector<bbox>boxes,float nms_threshold) {
        sort(boxes.begin(), boxes.end(),
                    [](bbox& a, bbox& b) { return a.confidence > b.confidence; });
        vector<bbox>box_result;
        vector<bool> remove_flags(boxes.size());
        box_result.reserve(boxes.size());
        for (int i = 0; i < boxes.size(); ++i) {
            if (remove_flags[i]) continue;
            auto& ibox = boxes[i];
            box_result.emplace_back(ibox);
            for (int j = i + 1; j < boxes.size(); ++j) {
            if (remove_flags[j]) continue;
            auto& jbox = boxes[j];
            if (ibox.class_label == jbox.class_label) {
                if (iou(ibox, jbox) >= nms_threshold) remove_flags[j] = true;
            }
            }
        }
        return box_result;
    }
} 


namespace Seg{

    std::vector<int> _classes_colors = 
    {
        0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 
        128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 
        64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 12
    };



    void render(cv::Mat& image, const cv::Mat& prob, const cv::Mat& iclass)
    {
        auto pimage = image.ptr<cv::Vec3b>(0);
        auto pprob  = prob.ptr<float>(0);
        auto pclass = iclass.ptr<uint8_t>(0);
        //0~512*512
        for(int i = 0; i < image.cols*image.rows; ++i, ++pimage, ++pprob, ++pclass){

            int iclass        = *pclass;
            float probability = *pprob;
            auto& pixel       = *pimage;
            float foreground  = min(0.6f + probability * 0.2f, 0.8f);
            float background  = 1 - foreground;
            for(int c = 0; c < 3; ++c){
                auto value = pixel[c] * background + foreground * _classes_colors[iclass * 3 + 2-c];
                pixel[c] = min((int)value, 255);
            }
        }
    }


    tuple<cv::Mat, cv::Mat> post_process(float* output, int output_width, int output_height, int num_class, int ibatch){
        // output 1*(numclass)*512*512）
        cv::Mat output_prob(output_height, output_width, CV_32F);
        cv::Mat output_index(output_height, output_width, CV_8U);
        //从第几个batch开始
        //每次加一个numclass 重复512*512次
        float* pnet   = output + ibatch * output_width * output_height * num_class;
        float* prob   = output_prob.ptr<float>(0);
        uint8_t* pidx = output_index.ptr<uint8_t>(0);

        for(int k = 0; k < output_prob.cols * output_prob.rows; ++k, pnet+=num_class, ++prob, ++pidx){
            //找到num_class中得分最大的值
            int ic = std::max_element(pnet, pnet + num_class) - pnet;
            *prob  = pnet[ic];
            *pidx  = ic;
        }
        return make_tuple(output_prob, output_index);
    }

}



#endif