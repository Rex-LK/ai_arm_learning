#ifndef __COMMON__
#define __COMMON__

#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <vector>
#include "hb_dnn.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "matrix.hpp"


namespace matrix{
    Matrix mygemm(const Matrix& a, const Matrix& b){
    Matrix c(a.rows(), b.cols());
    for(int i = 0; i < c.rows(); ++i){
        for(int j = 0; j < c.cols(); ++j){
            float summary = 0;
            for(int k = 0; k < a.cols(); ++k)
                summary += a(i, k) * b(k, j);

            c(i, j) = summary;
        }
    }
    return c;
}
}
namespace tools{
    long get_current_time()
    {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        long timestamp = tv.tv_sec * 1000 + tv.tv_usec / 1000;
        return timestamp;
    }
}
namespace detection{
    std::vector<std::string> coco_classes = {
        "person",        "bicycle",      "car",
        "motorcycle",    "airplane",     "bus",
        "train",         "truck",        "boat",
        "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench",        "bird",
        "cat",           "dog",          "horse",
        "sheep",         "cow",          "elephant",
        "bear",          "zebra",        "giraffe",
        "backpack",      "umbrella",     "handbag",
        "tie",           "suitcase",     "frisbee",
        "skis",          "snowboard",    "sports ball",
        "kite",          "base    // 需要恢复到原图大小 glass",   "cup",
        "fork",          "knife",        "spoon",
        "bowl",          "banana",       "apple",
        "sandwich",      "orange",       "broccoli",
        "carrot",        "hot dog",      "pizza",
        "donut",         "cake",         "chair",
        "couch",         "potted plant", "bed",
        "dining table",  "toilet",       "tv",
        "laptop",        "mouse",        "remote",
        "keyboard",      "cell phone",   "microwave",
        "oven",          "toaster",      "sink",
        "refrigerator",  "book",         "clock",
        "vase",          "scissors",     "teddy bear",
        "hair drier",    "toothbrush"};

        float Sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
        float Tanh(float x) { return 2.0f / (1.0f + exp(-2 * x)) - 1; }
        struct Bbox{
            float left, top, right, bottom, confidence;
            int class_label;
            cv::Rect box;
            cv::Mat boxMask;
            Matrix mask_cofs;
            Bbox() = default;
            Bbox(float left, float top, float right, float bottom, float confidence, int class_label, Matrix mask_cofs, cv::Rect box)
                : left(left), top(top), right(right), bottom(bottom), confidence(confidence), class_label(class_label), mask_cofs(mask_cofs), box(box){}
        };

        auto iou = [](const Bbox& a, const Bbox& b) {
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
    std::vector<Bbox> nms(std::vector<Bbox> &boxes_infer,float nms_threshold) {
        std::sort(boxes_infer.begin(), boxes_infer.end(),
                    [](Bbox& a, Bbox& b) { return a.confidence > b.confidence; });
        std::vector<bool> remove_flags(boxes_infer.size());
        std::vector<Bbox> boxes_result;
        boxes_result.reserve(boxes_infer.size());
        for (int i = 0; i < boxes_infer.size(); ++i) {
            if (remove_flags[i]) continue;
            auto& ibox = boxes_infer[i];
            boxes_result.emplace_back(ibox);
            for (int j = i + 1; j < boxes_infer.size(); ++j) {
            if (remove_flags[j]) continue;
            auto& jbox = boxes_infer[j];
            if (ibox.class_label == jbox.class_label) {
                if (iou(ibox, jbox) >= nms_threshold) remove_flags[j] = true;
            }
            }
        }
        return boxes_result;
    }
};

// Mat(brg) 转 nv12
namespace imgOp {
int32_t bgr_2_tensor_as_nv12(cv::Mat& bgr_mat, hbDNNTensor* input_tensor,
                             int input_h, int input_w) {
  hbDNNTensor* input = input_tensor;
  hbDNNTensorProperties Properties = input->properties;
  if (Properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
    input_h = Properties.validShape.dimensionSize[2];
    input_w = Properties.validShape.dimensionSize[3];
  }

  if (bgr_mat.empty()) {
    std::cout << "image is empty" << std::endl;
    return -1;
  }
  if (input_h % 2 || input_w % 2) {
    std::cout << "convert to YUV420 filed" << std::endl;
    return -1;
  }
  cv::Mat yuv_mat;
  cv::cvtColor(bgr_mat, yuv_mat, cv::COLOR_BGR2YUV_I420);
  uint8_t* nv12_data = yuv_mat.ptr<uint8_t>();
  // copy y data
  auto data = input->sysMem[0].virAddr;
  int32_t y_size = input_h * input_w;
  memcpy(reinterpret_cast<uint8_t*>(data), nv12_data, y_size);

  // copy uv data
  int32_t uv_height = input_h / 2;
  int32_t uv_width = input_w / 2;
  uint8_t* nv12 = reinterpret_cast<uint8_t*>(data) + y_size;
  uint8_t* u_data = nv12_data + y_size;
  uint8_t* v_data = u_data + uv_height * uv_width;

  for (int32_t i = 0; i < uv_width * uv_height; i++) {
    *nv12++ = *u_data++;
    *nv12++ = *v_data++;
  }
  return 0;
}

int32_t rgb_2_tensor_as_nv12(cv::Mat& rgb_mat, hbDNNTensor* input_tensor,
                             int input_h, int input_w) {
  hbDNNTensor* input = input_tensor;
  hbDNNTensorProperties Properties = input->properties;
  if (Properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
    input_h = Properties.validShape.dimensionSize[2];
    input_w = Properties.validShape.dimensionSize[3];
  }

  if (rgb_mat.empty()) {
      std::cout << "image is empty" << std::endl;
      return -1;
  }
  // resize
  cv::Mat mat;
  mat.create(input_h, input_w, rgb_mat.type());
  // cv::resize(bgr_mat, mat, mat.size(), 0, 0);
  // convert to YUV420
  if (input_h % 2 || input_w % 2) {
      std::cout << "convert to YUV420 filed" << std::endl;
      return -1;
  }

  cv::Mat bgr_mat;
  cv::cvtColor(rgb_mat, bgr_mat, cv::COLOR_RGB2BGR);

  cv::Mat yuv_mat;
  cv::cvtColor(bgr_mat, yuv_mat, cv::COLOR_BGR2YUV_I420);
  uint8_t* nv12_data = yuv_mat.ptr<uint8_t>();

  // copy y data
  auto data = input->sysMem[0].virAddr;
  int32_t y_size = input_h * input_w;
  memcpy(reinterpret_cast<uint8_t*>(data), nv12_data, y_size);

  // copy uv data
  int32_t uv_height = input_h / 2;
  int32_t uv_width = input_w / 2;
  uint8_t* nv12 = reinterpret_cast<uint8_t*>(data) + y_size;
  uint8_t* u_data = nv12_data + y_size;
  uint8_t* v_data = u_data + uv_height * uv_width;

  for (int32_t i = 0; i < uv_width * uv_height; i++) {
    *nv12++ = *u_data++;
    *nv12++ = *v_data++;
  }
  return 0;
}

int32_t yuv420_2_tensor_as_nv12(cv::Mat& yuv_mat, hbDNNTensor* input_tensor,
                                int input_h, int input_w) {
  hbDNNTensor* input = input_tensor;
  hbDNNTensorProperties Properties = input->properties;
  if (Properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
    input_h = Properties.validShape.dimensionSize[2];
    input_w = Properties.validShape.dimensionSize[3];
  }

  if (yuv_mat.empty()) {
      std::cout << "image is empty" << std::endl;
      return -1;
  }
  if (input_h % 2 || input_w % 2) {
      std::cout << "convert to YUV420 filed" << std::endl;
      return -1;
  }

  uint8_t* nv12_data = yuv_mat.ptr<uint8_t>();
  // copy y data
  auto data = input->sysMem[0].virAddr;
  int32_t y_size = input_h * input_w;
  memcpy(reinterpret_cast<uint8_t*>(data), nv12_data, y_size);

  // copy uv data
  int32_t uv_height = input_h / 2;
  int32_t uv_width = input_w / 2;
  uint8_t* nv12 = reinterpret_cast<uint8_t*>(data) + y_size;
  uint8_t* u_data = nv12_data + y_size;
  uint8_t* v_data = u_data + uv_height * uv_width;

  for (int32_t i = 0; i < uv_width * uv_height; i++) {
    *nv12++ = *u_data++;
    *nv12++ = *v_data++;
  }
  return 0;
}

}

#endif