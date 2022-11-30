

#include <iostream>

#include "hb_dnn.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"

#include "common.hpp"


/**
 * Prepare input tensor
 * @param[in] image_data: bgr data
 * @param[in] image_height: image height
 * @param[in] image_width: image width
 * @param[out] tensor: tensor to be prepared and filled
 * @return: 0 if success, and -1 if failed
 */
int32_t prepare_bgr_tensor(uint8_t *image_data,
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
int32_t prepare_bgr_tensor(int image_height,
                           int image_width,
                           hbDNNTensor *tensor);

/**
 * Release bgr tensor
 * @param[in] tensor: tensor to be released
 * @return: 0 if success, and -1 if failed
 */
int32_t free_bgr_tensor(hbDNNTensor *tensor);

int main(int argc, char **argv) {
  // Prepare input tensor
  cv::Mat ori_bgr = cv::imread("../../data/images/kite.jpg", 1);

  hbDNNTensor input_tensor;
  prepare_bgr_tensor(
                       ori_bgr.data, ori_bgr.rows, ori_bgr.cols, &input_tensor);

  // Prepare output tensor
  hbDNNTensor output_tensor;
  prepare_bgr_tensor(
                       416, 416, &output_tensor);


  hbDNNResizeCtrlParam ctrl;
  HB_DNN_INITIALIZE_RESIZE_CTRL_PARAM(&ctrl);

  hbDNNTaskHandle_t task_handle;
  hbDNNRoi roi;
  if (1) {
    
        hbDNNResize(
            &task_handle, &output_tensor, &input_tensor, nullptr, &ctrl);
  } else {
    roi = {1, 1, 1, 1};
        hbDNNResize(&task_handle, &output_tensor, &input_tensor, &roi, &ctrl);
  }
  hbDNNWaitTaskDone(task_handle, 0);
  auto resized_shape = output_tensor.properties.alignedShape.dimensionSize;
  cv::Mat crop_resized_bgr(resized_shape[1], resized_shape[2], CV_8UC3);
  memcpy(crop_resized_bgr.data,
         output_tensor.sysMem[0].virAddr,
         resized_shape[1] * resized_shape[2] * resized_shape[3]);
  cv::imwrite("test.png", crop_resized_bgr);

  // Release tensor
  free_bgr_tensor(&input_tensor);
  free_bgr_tensor(&output_tensor);
}

int32_t prepare_bgr_tensor(uint8_t *image_data,
                           int image_height,
                           int image_width,
                           hbDNNTensor *tensor) {
  auto &properties = tensor->properties;
  properties.tensorType = HB_DNN_IMG_TYPE_BGR;
  properties.tensorLayout = HB_DNN_LAYOUT_NHWC;

  auto &valid_shape = properties.validShape;
  valid_shape.numDimensions = 4;
  valid_shape.dimensionSize[0] = 1;
  valid_shape.dimensionSize[1] = image_height;
  valid_shape.dimensionSize[2] = image_width;
  valid_shape.dimensionSize[3] = 3;

  auto &aligned_shape = properties.alignedShape;
  aligned_shape = valid_shape;

  int image_length = aligned_shape.dimensionSize[1] *
                     aligned_shape.dimensionSize[2] *
                     aligned_shape.dimensionSize[3];
 // 申请缓存的bpu内存
  hbSysAllocCachedMem(&tensor->sysMem[0], image_length);
 // data0是bpu内存的 虚拟地址
 // 将image的内容拷贝到bpu中
  void *data0 = tensor->sysMem[0].virAddr;
  memcpy(data0, image_data, image_length);
  // 对缓存的BPU进行刷新
  hbSysFlushMem(&(tensor->sysMem[0]), HB_SYS_MEM_CACHE_CLEAN);

  return 0;
}

int32_t prepare_bgr_tensor(int image_height,
                           int image_width,
                           hbDNNTensor *tensor) {
  auto &properties = tensor->properties;
  properties.tensorType = HB_DNN_IMG_TYPE_BGR;
  properties.tensorLayout = HB_DNN_LAYOUT_NHWC;

  auto &valid_shape = properties.validShape;
  valid_shape.numDimensions = 4;
  valid_shape.dimensionSize[0] = 1;
  valid_shape.dimensionSize[1] = image_height;
  valid_shape.dimensionSize[2] = image_width;
  valid_shape.dimensionSize[3] = 3;

  auto &aligned_shape = properties.alignedShape;
  aligned_shape = valid_shape;
  int image_length = aligned_shape.dimensionSize[1] *
                     aligned_shape.dimensionSize[2] *
                     aligned_shape.dimensionSize[3];
  (hbSysAllocCachedMem(&tensor->sysMem[0], image_length),
                   "hbSysAllocCachedMem failed");
  return 0;
}

int32_t free_bgr_tensor(hbDNNTensor *tensor) {
  (hbSysFreeMem(&(tensor->sysMem[0])), "hbSysFreeMem failed");
  return 0;
}
