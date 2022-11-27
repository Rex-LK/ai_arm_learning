#ifndef __INFERENCE__
#define __INFERENCE__

#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

#include <iomanip>
#include <iostream>
#include <queue>
#include <vector>
#include "hb_dnn.h"

class Inference {
 public:
  Inference(std::string modelPath);
  ~Inference();
  std::vector<float*> inference();

 private:
  int prepare_tensor(hbDNNTensor *input_tensor, hbDNNTensor *output_tensor,
                     hbDNNHandle_t dnn_handle);

 public:
  std::vector<hbDNNTensor> input_tensors;
  std::vector<hbDNNTensor> output_tensors;
  int input_w;
  int input_h;

  hbPackedDNNHandle_t packed_dnn_handle;
  hbDNNHandle_t dnn_handle;
  const char **model_name_list;
  hbDNNTensor *output;
  hbDNNInferCtrlParam infer_ctrl_param;
  hbDNNTaskHandle_t task_handle;
  hbDNNTensorProperties Properties;
  float *result;
  int input_count = 0;
  int output_count = 0;
  int model_count = 0;
};
#endif

// 加载模型，分配模型输入输出内存
Inference::Inference(std::string modelPath) {
  auto modelFileName = modelPath.c_str();
  hbDNNInitializeFromFiles(&packed_dnn_handle, &modelFileName, 1);
  hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle);
  hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]);
  hbDNNGetInputCount(&input_count, dnn_handle);
  hbDNNGetOutputCount(&output_count, dnn_handle);
  input_tensors.resize(input_count);
  output_tensors.resize(output_count);
  prepare_tensor(input_tensors.data(), output_tensors.data(), dnn_handle);
  HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);
  Properties = input_tensors[0].properties;
  input_h = Properties.validShape.dimensionSize[2];
  input_w = Properties.validShape.dimensionSize[3];
}

// 推理
std::vector<float*> Inference::inference() {
  task_handle = nullptr;
  output = output_tensors.data();

  for (int i = 0; i < input_count; i++) {
    hbSysFlushMem(&input_tensors[i].sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
  }
  hbDNNInfer(&task_handle, &output, input_tensors.data(), dnn_handle,
             &infer_ctrl_param);
  // hbDNNRoiInfer(&task_handle, &output, input_tensors.data(), rois.data(), 1,
  //               dnn_handle, &infer_ctrl_param);

  hbDNNWaitTaskDone(task_handle, 0);
  hbDNNReleaseTask(task_handle);

  std::vector<float*> res;
  for (int i = 0; i < output_count; i++) {
    hbSysFlushMem(&output_tensors[i].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
    result = reinterpret_cast<float *>(output_tensors[i].sysMem[0].virAddr);
    // result = reinterpret_cast<float *>(output->sysMem[0].virAddr);
    res.push_back(result);
  }
  return res;
}

Inference::~Inference() {
  for (int i = 0; i < input_count; i++) {
    hbSysFreeMem(&(input_tensors[i].sysMem[0]));
  }
  for (int i = 0; i < output_count; i++) {
    hbSysFreeMem(&(output_tensors[i].sysMem[0]));
  }
  hbDNNRelease(packed_dnn_handle);
}

int Inference::prepare_tensor(hbDNNTensor *input_tensor,
                             hbDNNTensor *output_tensor,
                             hbDNNHandle_t dnn_handle) {
  int input_count = 0;
  int output_count = 0;
  hbDNNGetInputCount(&input_count, dnn_handle);
  hbDNNGetOutputCount(&output_count, dnn_handle);

  hbDNNTensor *input = input_tensor;
  for (int i = 0; i < input_count; i++) {
    hbDNNGetInputTensorProperties(&input[i].properties, dnn_handle, i);
    int input_memSize = input[i].properties.alignedByteSize;
    hbSysAllocCachedMem(&input[i].sysMem[0], input_memSize);
    input[i].properties.alignedShape = input[i].properties.validShape;
    const char *input_name;
    hbDNNGetInputName(&input_name, dnn_handle, i);
  }

  hbDNNTensor *output = output_tensor;
  for (int i = 0; i < output_count; i++) {
    hbDNNGetOutputTensorProperties(&output[i].properties, dnn_handle, i);
    int output_memSize = output[i].properties.alignedByteSize;
    hbSysAllocCachedMem(&output[i].sysMem[0], output_memSize);

    const char *output_name;
    hbDNNGetOutputName(&output_name, dnn_handle, i);
  }
  return 0;
}