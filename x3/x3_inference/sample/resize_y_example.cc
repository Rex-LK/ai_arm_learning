// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include <iostream>
#include <string>

#include "hb_dnn.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv.hpp"
#include "time.h"
/**
 * Align by 16
 */


#define ALIGNED_2E(w, alignment) ((w) + (alignment - 1)) & (~(alignment - 1))
#define ALIGN_16(w) ALIGNED_2E(w, 16)

/**
 * Prepare input tensor
 * @param[in] image_data: image y data
 * @param[in] image_height: image height
 * @param[in] image_width: image width
 * @param[out] tensor: tensor to be prepared and filled
 * @return: 0 if success, and -1 if failed
 */
int prepare_y_tensor(uint8_t *image_data,
                     int image_height,
                     int image_width,
                     hbDNNTensor *tensor);

/**
 * Prepare output tensor
 * @param[in] image_height: image height
 * @param[in] image_width: image width
 * @param[out] tensor: tensor to be prepared
 * @return: 0 if success, and -1 if failed
 */
int prepare_y_tensor(int image_height, int image_width, hbDNNTensor *tensor);

int main(int argc, char **argv) {
  // Parsing command line arguments


  // Prepare input tensor
  cv::Mat gray = cv::imread("../images/kite.jpg");
  cv::Mat yuv;
  cvtColor(gray, yuv, cv::COLOR_BGR2YUV_I420);

  cv::imwrite("yuv.png",yuv);
  
  hbDNNTensor input_tensor;
      prepare_y_tensor(gray.data, gray.rows, gray.cols, &input_tensor);

  // Prepare output tensor
  hbDNNTensor output_tensor;
      prepare_y_tensor(224, 224, &output_tensor);
  hbDNNResizeCtrlParam ctrl;
  HB_DNN_INITIALIZE_RESIZE_CTRL_PARAM(&ctrl);
  // Resize
  hbDNNTaskHandle_t task_handle;
  clock_t start, end;
  start = clock();

  hbDNNResize(&task_handle, &output_tensor, &input_tensor, nullptr, &ctrl);
  hbDNNWaitTaskDone(task_handle, 0);


  hbDNNReleaseTask(task_handle);

  end = clock();
  auto resized_shape = output_tensor.properties.alignedShape.dimensionSize;
  cv::Mat y_mat(resized_shape[2], resized_shape[3], CV_8UC1);
  memcpy(y_mat.data,
         output_tensor.sysMem[0].virAddr,
         resized_shape[2] * resized_shape[3]);
  cv::imwrite("2.png", y_mat);

  // Release tensor
  hbSysFreeMem(&(input_tensor.sysMem[0]));
  hbSysFreeMem(&(output_tensor.sysMem[0]));
  return 0;
}

int prepare_y_tensor(uint8_t *image_data,
                     int image_height,
                     int image_width,
                     hbDNNTensor *tensor) {
  auto &properties = tensor->properties;
  properties.tensorType = HB_DNN_IMG_TYPE_Y;
  properties.tensorLayout = HB_DNN_LAYOUT_NCHW;
  auto &valid_shape = properties.validShape;
  valid_shape.numDimensions = 4;
  valid_shape.dimensionSize[0] = 1;
  valid_shape.dimensionSize[1] = 1;
  valid_shape.dimensionSize[2] = image_height;
  valid_shape.dimensionSize[3] = image_width;

  // Align by 16 bytes
  int stride = ALIGN_16(image_width);
  auto &aligned_shape = properties.alignedShape;
  aligned_shape.numDimensions = 4;
  aligned_shape.dimensionSize[0] = 1;
  aligned_shape.dimensionSize[1] = 1;
  aligned_shape.dimensionSize[2] = image_height;
  aligned_shape.dimensionSize[3] = stride;

  int image_length = aligned_shape.dimensionSize[1] *
                     aligned_shape.dimensionSize[2] *
                     aligned_shape.dimensionSize[3];
  hbSysAllocCachedMem(&tensor->sysMem[0], image_length);
  uint8_t *data0 = reinterpret_cast<uint8_t *>(tensor->sysMem[0].virAddr);
  for (int h = 0; h < image_height; ++h) {
    auto *raw = data0 + h * stride;
    for (int w = 0; w < image_width; ++w) {
      *raw++ = *image_data++;
    }
  }

  hbSysFlushMem(&(tensor->sysMem[0]), HB_SYS_MEM_CACHE_CLEAN);
  return 0;
}

int prepare_y_tensor(int image_height, int image_width, hbDNNTensor *tensor) {
  auto &properties = tensor->properties;
  properties.tensorType = HB_DNN_IMG_TYPE_Y;
  properties.tensorLayout = HB_DNN_LAYOUT_NCHW;

  auto &valid_shape = properties.validShape;
  valid_shape.numDimensions = 4;
  valid_shape.dimensionSize[0] = 1;
  valid_shape.dimensionSize[1] = 1;
  valid_shape.dimensionSize[2] = image_height;
  valid_shape.dimensionSize[3] = image_width;

  auto &aligned_shape = properties.alignedShape;
  aligned_shape = valid_shape;
  int image_length = image_height * image_width;
  hbSysAllocCachedMem(&tensor->sysMem[0], image_length);
  return 0;
}
