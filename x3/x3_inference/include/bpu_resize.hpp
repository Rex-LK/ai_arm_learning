#ifndef __BPU_RESIZE__
#define __BPU_RESIZE__

#include <iostream>
#include <string>
#include "hb_dnn.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
using namespace std;

#define ALIGNED_2E(w, alignment) ((w) + (alignment - 1)) & (~(alignment - 1))
#define ALIGN_16(w) ALIGNED_2E(w, 16)

int32_t prepare_input_tensor(
                           int image_height,
                           int image_width,
                           hbDNNTensor *tensor,int& stride,int& input_image_length);
int32_t prepare_output_tensor(int image_height,
                           int image_width,
                           hbDNNTensor *tensor,int& output_image_length);
int32_t free_tensor(hbDNNTensor *tensor);


enum class imageType : int {
    BRG = 0,
    RGB = 1,
    yuv420 = 2,
    NV12 = 3,
};

class BpuResize{
// 固定尺寸 支持 brg rgb yuv420 nv12

public:
    BpuResize(int input_w,int input_h,int output_w,int output_h,imageType imgType){
        this->imgType = imgType;
        inputW = input_w;   
        outputW = output_w; 
        if(imgType == imageType::yuv420){
            inputH = input_h * 1.5;
            outputH = output_h * 1.5;
            dataType = HB_DNN_IMG_TYPE_Y;
            layout = HB_DNN_LAYOUT_NCHW;
        }
        prepare_input_tensor(inputH, inputW, &input_tensor, stride, input_image_length);
        prepare_output_tensor(outputH, outputW, &output_tensor,output_image_length);

    }
    ~BpuResize(){
        free_tensor(&input_tensor);
        free_tensor(&output_tensor);
    }
    void copy_image_2_input_tensor(uint8_t *yuv_data,hbDNNTensor *tensor){
        uint8_t *data0 = reinterpret_cast<uint8_t *>(tensor->sysMem[0].virAddr);
        memcpy(data0, yuv_data, input_image_length);
        hbSysFlushMem(&(tensor->sysMem[0]), HB_SYS_MEM_CACHE_CLEAN);
    }

    void YuvResize(cv::Mat ori_yuv,cv::Mat resizedYuvMat){
        HB_DNN_INITIALIZE_RESIZE_CTRL_PARAM(&ctrl);
        copy_image_2_input_tensor(ori_yuv.data,&input_tensor);
        if (!useCrop)
        {
            hbDNNResize(&task_handle, &output_tensor, &input_tensor, nullptr, &ctrl);
        }
        else
        {
            hbDNNResize(&task_handle, &output_tensor, &input_tensor, &roi, &ctrl);
        }

        hbDNNWaitTaskDone(task_handle, 0);
        hbDNNReleaseTask(task_handle);
        memcpy(resizedYuvMat.data,output_tensor.sysMem[0].virAddr,outputW * outputH);
        return;
    }
private:
    int inputW;
    int inputH;
    int outputW;
    int outputH;

    imageType imgType = imageType::yuv420;
    hbDNNDataType dataType;
    hbDNNTensorLayout layout;

    hbDNNTensor input_tensor;
    hbDNNTensor output_tensor;
    hbDNNResizeCtrlParam ctrl;
    hbDNNTaskHandle_t task_handle;
    hbDNNRoi roi = {0, 0, 0, 0};
    bool useCrop = false;
    int input_image_length;
    int output_image_length;

    int stride;
};
#endif



int32_t prepare_input_tensor(int image_height,
                           int image_width,
                           hbDNNTensor *tensor,int& stride,int&input_image_length) {
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
  stride = ALIGN_16(image_width);
  auto &aligned_shape = properties.alignedShape;
  aligned_shape.numDimensions = 4;
  aligned_shape.dimensionSize[0] = 1;
  aligned_shape.dimensionSize[1] = 1;
  aligned_shape.dimensionSize[2] = image_height;
  aligned_shape.dimensionSize[3] = stride;

  input_image_length = aligned_shape.dimensionSize[1] *
                     aligned_shape.dimensionSize[2] *
                     aligned_shape.dimensionSize[3];
  hbSysAllocCachedMem(&tensor->sysMem[0], input_image_length);
  return 0;
}

int32_t prepare_output_tensor(int image_height,
                           int image_width,
                           hbDNNTensor *tensor,int&output_image_length) {
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
  output_image_length = image_height * image_width;
  hbSysAllocCachedMem(&tensor->sysMem[0], output_image_length);
  return 0;
}





int32_t free_tensor(hbDNNTensor *tensor) {
  hbSysFreeMem(&(tensor->sysMem[0]));
  return 0;
}