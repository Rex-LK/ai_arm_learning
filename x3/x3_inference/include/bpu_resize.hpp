#ifndef __BPU_RESIZE__
#define __BPU_RESIZE__

#include <iostream>
#include <string>
#include <initializer_list>
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
    BGR = 0,
    RGB = 1,
    YUV420 = 2,
    NV12 = 3,
};

class BpuResize{
// 固定尺寸 支持 brg rgb yuv420 nv12

public:
    BpuResize(const BpuResize& other) = delete; 
    BpuResize& operator = (const BpuResize& other) = delete;
    explicit BpuResize(const int input_w, const int input_h, const int output_w, const int output_h, const imageType imgType);
    ~BpuResize();
    void copy_image_2_input_tensor(uint8_t *image_data,hbDNNTensor *tensor);
    float *Resize(cv::Mat ori_img, const std::initializer_list<int> crop);
    int32_t prepare_input_tensor();
    int32_t prepare_output_tensor();

private:
    int inputW;
    int inputH;
    int outputW;
    int outputH;
    int Dimensions;
    int shape_0, shape_1, shape_2, shape_3;
    imageType imgType;
    hbDNNDataType dataType;
    hbDNNTensorLayout layout;

    hbDNNTensor input_tensor;
    hbDNNTensor output_tensor;
    hbDNNResizeCtrlParam ctrl;
    hbDNNTaskHandle_t task_handle;
    hbDNNRoi roi;
    bool useCrop = false;
    int input_image_length;
    int output_image_length;

    int stride;
};
#endif

BpuResize::BpuResize(const int input_w,const int input_h,const int output_w,const int output_h,const imageType imgType){
    this->imgType = imgType;
    inputW = input_w;   
    outputW = output_w; 
    if(imgType == imageType::YUV420){
        inputH = input_h * 1.5;
        outputH = output_h * 1.5;
        dataType = HB_DNN_IMG_TYPE_Y;
        layout = HB_DNN_LAYOUT_NCHW;
        Dimensions = 4;
        shape_0 = 1;
        shape_1 = 1;
        shape_2 = inputH;
        shape_3 = inputW;

    }
    else if(imgType == imageType::BGR || imgType == imageType::RGB){
        dataType = this->imgType==imageType::BGR ? HB_DNN_IMG_TYPE_BGR:HB_DNN_IMG_TYPE_RGB;
        inputH = input_h;
        outputH = output_h;
        layout = HB_DNN_LAYOUT_NHWC;
        Dimensions = 4;
        shape_0 = 1;
        shape_1 = inputH;
        shape_2 = inputW;
        shape_3 = 3;

    }
    prepare_input_tensor();
    prepare_output_tensor();
}
BpuResize::~BpuResize(){
    hbSysFreeMem(&input_tensor.sysMem[0]);
    hbSysFreeMem(&output_tensor.sysMem[0]);
}

int32_t BpuResize::prepare_input_tensor() {
        
    auto &properties = this->input_tensor.properties;
    auto &aligned_shape = properties.alignedShape;
    auto &valid_shape = properties.validShape;

    if(this->imgType == imageType::YUV420){
        properties.tensorType = HB_DNN_IMG_TYPE_Y;
        properties.tensorLayout = HB_DNN_LAYOUT_NCHW;
        valid_shape.numDimensions = Dimensions;
        valid_shape.dimensionSize[0] = shape_0;
        valid_shape.dimensionSize[1] = shape_1;
        valid_shape.dimensionSize[2] = shape_2;
        valid_shape.dimensionSize[3] = shape_3;
        stride = ALIGN_16(inputW);
        aligned_shape.numDimensions = Dimensions;
        aligned_shape.dimensionSize[0] = shape_0;
        aligned_shape.dimensionSize[1] = shape_1;
        aligned_shape.dimensionSize[2] = shape_2;
        aligned_shape.dimensionSize[3] = stride;
    }
    else if (this->imgType == imageType::BGR||this->imgType == imageType::RGB){
            properties.tensorType = this->imgType==imageType::BGR ?HB_DNN_IMG_TYPE_BGR:HB_DNN_IMG_TYPE_RGB;
            properties.tensorLayout = HB_DNN_LAYOUT_NHWC;
            valid_shape.numDimensions = Dimensions;
            valid_shape.dimensionSize[0] = shape_0;
            valid_shape.dimensionSize[1] = shape_1;
            valid_shape.dimensionSize[2] = shape_2;
            valid_shape.dimensionSize[3] = shape_3;
    }
    aligned_shape = valid_shape;
    input_image_length = aligned_shape.dimensionSize[1]*
                        aligned_shape.dimensionSize[2]*
                        aligned_shape.dimensionSize[3];
    hbSysAllocCachedMem(&this->input_tensor.sysMem[0], input_image_length);
    return 0;
}


int32_t BpuResize::prepare_output_tensor()
{
    auto &properties = this->output_tensor.properties;
    auto &valid_shape = properties.validShape;
    auto &aligned_shape = properties.alignedShape;
    if (this->imgType == imageType::YUV420)
    {
        properties.tensorType = HB_DNN_IMG_TYPE_Y;
        properties.tensorLayout = HB_DNN_LAYOUT_NCHW;
        valid_shape.numDimensions = Dimensions;
        valid_shape.dimensionSize[0] = shape_0;
        valid_shape.dimensionSize[1] = shape_1;
        valid_shape.dimensionSize[2] = outputH;
        valid_shape.dimensionSize[3] = outputW;
    }
    else if (this->imgType == imageType::BGR || this->imgType == imageType::RGB)
    {
        properties.tensorType = this->imgType == imageType::BGR ? HB_DNN_IMG_TYPE_BGR : HB_DNN_IMG_TYPE_RGB;
        properties.tensorLayout = HB_DNN_LAYOUT_NHWC;
        valid_shape.numDimensions = Dimensions;
        valid_shape.dimensionSize[0] = shape_0;
        valid_shape.dimensionSize[1] = outputH;
        valid_shape.dimensionSize[2] = outputW;
        valid_shape.dimensionSize[3] = shape_3;
    }
    aligned_shape = valid_shape;
    output_image_length = aligned_shape.dimensionSize[1] *
                          aligned_shape.dimensionSize[2] *
                          aligned_shape.dimensionSize[3];
    hbSysAllocCachedMem(&this->output_tensor.sysMem[0], output_image_length);
    return 0;
}
float* BpuResize::Resize(cv::Mat ori_img,const std::initializer_list<int> crop={}){
    if(crop.size()>0){
        assert(crop.size()==4);
        vector<int>tmp = crop;
        useCrop = true;
        roi= {tmp[0],tmp[1],tmp[2],tmp[3]};
    }
    HB_DNN_INITIALIZE_RESIZE_CTRL_PARAM(&ctrl);
    copy_image_2_input_tensor(ori_img.data,&input_tensor);
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
    float* res  = reinterpret_cast<float *>(output_tensor.sysMem[0].virAddr);
    return res;
}

void BpuResize::copy_image_2_input_tensor(uint8_t *image_data,hbDNNTensor *tensor){
    uint8_t *data0 = reinterpret_cast<uint8_t *>(tensor->sysMem[0].virAddr);
    memcpy(data0, image_data, input_image_length);
    hbSysFlushMem(&(tensor->sysMem[0]), HB_SYS_MEM_CACHE_CLEAN);
}