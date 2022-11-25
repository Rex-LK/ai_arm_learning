
#include "common.h"
using namespace std;
using namespace Det;
using namespace Seg;
using namespace cv;

#define EMPTY ""

enum VLOG_LEVEL {
  EXAMPLE_SYSTEM = 0,
  EXAMPLE_REPORT = 1,
  EXAMPLE_DETAIL = 2,
  EXAMPLE_DEBUG = 3
};


#define HB_CHECK_SUCCESS(value, errmsg)                              \
  do {                                                               \
    /*value can be call of function*/                                \
    auto ret_code = value;                                           \
    if (ret_code != 0) {                                             \
      VLOG(EXAMPLE_SYSTEM) << errmsg << ", error code:" << ret_code; \
      return ret_code;                                               \
    }                                                                \
  } while (0);

int prepare_tensor(hbDNNTensor *input_tensor,
                   hbDNNTensor *output_tensor,
                   hbDNNHandle_t dnn_handle);

int32_t read_image_2_tensor_as_nv12(Mat &image,
                                    hbDNNTensor *input_tensor);
 

/**
 * Step1: get model handle
 * Step2: prepare input and output tensor
 * Step3: set input data to input tensor
 * Step4: run inference
 * Step5: do postprocess with output data
 * Step6: release resources
 */

long get_current_time(){
    struct timeval tv;
    gettimeofday(&tv, NULL);
    long timestamp = tv.tv_sec * 1000 + tv.tv_usec / 1000;
    return timestamp;
}

int main(int argc, char **argv) {
  // Parsing command line arguments

  hbPackedDNNHandle_t packed_dnn_handle;
  hbDNNHandle_t dnn_handle;
  const char **model_name_list;
  // auto modelFileName = FLAGS_model_file.c_str();
  
  string model_path = "../models/yolov6-sim.bin";
  auto modelFileName = model_path.c_str();
  string image_path = "../images/det.png";
  Mat image0 = imread(image_path);
  Mat image = image0.clone();

  string detect_type = "detect";  //"detect" "classfily"  "segment"


  int image_h = image.rows;
  int image_w = image.cols;

  int model_count = 0;
  // Step1: get model handle
  {
    hbDNNInitializeFromFiles(&packed_dnn_handle, &modelFileName, 1);
    hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle);
    hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]);
  }
  // Show how to get dnn version
  vector<hbDNNTensor> input_tensors;
  vector<hbDNNTensor> output_tensors;
  int input_count = 0;
  int output_count = 0;
  // Step2: prepare input and output tensor
  {
    hbDNNGetInputCount(&input_count, dnn_handle);
    hbDNNGetOutputCount(&output_count, dnn_handle);
    input_tensors.resize(input_count);
    output_tensors.resize(output_count);
    prepare_tensor(input_tensors.data(), output_tensors.data(), dnn_handle);
  }

  // Step3: set input data to input tensor
  {
        read_image_2_tensor_as_nv12(image, input_tensors.data());
  }

  hbDNNTaskHandle_t task_handle = nullptr;
  hbDNNTensor *output = output_tensors.data();

  // Step4: run inference
  {
    // make sure memory data is flushed to DDR before inference
    for (int i = 0; i < input_count; i++) {
      hbSysFlushMem(&input_tensors[i].sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
    }

    hbDNNInferCtrlParam infer_ctrl_param;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);
    
    
    hbDNNInfer(&task_handle,&output,
                                input_tensors.data(),
                                dnn_handle,
                                &infer_ctrl_param);

    // wait task done
    hbDNNWaitTaskDone(task_handle, 0),
                     "hbDNNWaitTaskDone failed";
  long t2 = get_current_time();


  }


  // // make sure CPU read data from DDR before using output tensor data
  for (int i = 0; i < output_count; i++) {
    hbSysFlushMem(&output_tensors[i].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
  }

  auto data = reinterpret_cast<float *>(output->sysMem[0].virAddr);

  if(detect_type=="segment"){
    int *shape = output->properties.validShape.dimensionSize;
    int out_h = shape[1];
    int out_w = shape[2];
    int out_nclass = shape[3];
    
    Mat unet_prob, iclass;

    
    resize(image,image,Size(out_w,out_h));
    tie(unet_prob, iclass) = post_process(data, out_w, out_h, out_nclass, 0);
    imwrite("unet_prob.jpg", unet_prob);  
    cout<<"Done, Save to image-prob.jpg"<<endl;

    imwrite("unet_mask.jpg", iclass);  
    cout<<"Done, Save to image-mask.jpg"<<endl;
    
    render(image, unet_prob, iclass);
    resize(image,image,Size(image_w,image_h));
    imwrite("unet-rgb.jpg", image);
    cout<<"Done, Save to unet-rgb.jpg"<<endl;

  }
  else if(detect_type == "detect"){
      int input_w = 640;
      int input_h = 640;
      
      // 特征图大小
      int cols = 85;
      int num_classes = cols - 5;
      int rows = 8400;
      float confidence_threshold = 0.4;
      float nms_threshold = 0.6;
      vector<bbox> BBoxes = decodeBbox(data,rows,cols,confidence_threshold,num_classes);
      vector<bbox> resBBoxes = nms(BBoxes,nms_threshold);

      Mat drawMat = image0.clone();
      Mat show_img;
      resize(image0,show_img,Size(640,640));

      for (auto& box : resBBoxes) {

          // string label_name = coco_classes[box.class_label];
          rectangle(show_img, Point(box.left, box.top),
                        Point(box.right, box.bottom), Scalar(0, 255, 0), 2);
          putText(show_img, format("%.2f", box.confidence),
                      Point(box.left, box.top - 10), 0, 0.8,
                      Scalar(0, 0, 255), 2, 2);
          // putText(detBgrMat, label_name, Point(box.left, box.top + 10), 0,
          //             0.8, Scalar(0, 0, 255), 2, 2);
      }
        imwrite("det.jpg",show_img);
        cout<<"Done, Save to det.jpg"<<endl;
  }



  // Step6: release resources
  {
    // release task handle
    hbDNNReleaseTask(task_handle);
    // free input mem
    for (int i = 0; i < input_count; i++) {
      hbSysFreeMem(&(input_tensors[i].sysMem[0]));
    }
    // free output mem
    for (int i = 0; i < output_count; i++) {
      hbSysFreeMem(&(output_tensors[i].sysMem[0]));
    }
    // release model
    hbDNNRelease(packed_dnn_handle);
  }

  return 0;
}

int prepare_tensor(hbDNNTensor *input_tensor,
                   hbDNNTensor *output_tensor,
                   hbDNNHandle_t dnn_handle) {
  int input_count = 0;
  int output_count = 0;
  hbDNNGetInputCount(&input_count, dnn_handle);
  hbDNNGetOutputCount(&output_count, dnn_handle);

  /** Tips:
   * For input memory size:
   * *   input_memSize = input[i].properties.alignedByteSize
   * For output memory size:
   * *   output_memSize = output[i].properties.alignedByteSize
   */
  hbDNNTensor *input = input_tensor;
  for (int i = 0; i < input_count; i++) { 
        hbDNNGetInputTensorProperties(&input[i].properties, dnn_handle, i);
    int input_memSize = input[i].properties.alignedByteSize;
    hbSysAllocCachedMem(&input[i].sysMem[0], input_memSize);
    /** Tips:
     * For input tensor, aligned shape should always be equal to the real
     * shape of the user's data. If you are going to set your input data with
     * padding, this step is not necessary.
     * */
    input[i].properties.alignedShape = input[i].properties.validShape;

    // Show how to get input name
    const char *input_name;
    hbDNNGetInputName(&input_name, dnn_handle, i);
  }

  hbDNNTensor *output = output_tensor;
  for (int i = 0; i < output_count; i++) {
    hbDNNGetOutputTensorProperties(&output[i].properties, dnn_handle, i);
    int output_memSize = output[i].properties.alignedByteSize;
    hbSysAllocCachedMem(&output[i].sysMem[0], output_memSize);

    // Show how to get output name
    const char *output_name;
    hbDNNGetOutputName(&output_name, dnn_handle, i);
  }
  return 0;
}

/** You can define read_image_2_tensor_as_other_type to prepare your data **/
int32_t read_image_2_tensor_as_nv12(Mat &image,
                                    hbDNNTensor *input_tensor) {
  hbDNNTensor *input = input_tensor;
  hbDNNTensorProperties Properties = input->properties;
  int tensor_id = 0;
  int input_h = Properties.validShape.dimensionSize[1];
  int input_w = Properties.validShape.dimensionSize[2];
  if (Properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
    input_h = Properties.validShape.dimensionSize[2];
    input_w = Properties.validShape.dimensionSize[3];
  }

  Mat bgr_mat = image;
  if (bgr_mat.empty()) {
      cout << "image file not exist!" << endl;
      return -1;
  }
  // resize
  Mat mat;
  mat.create(input_h, input_w, bgr_mat.type());
  resize(bgr_mat, mat, mat.size(), 0, 0);
  // convert to YUV420
  if (input_h % 2 || input_w % 2) {
      cout << "input img height and width must aligned by 2!" << endl;
      return -1;
  }
  Mat yuv_mat;
  cvtColor(mat, yuv_mat, COLOR_BGR2YUV_I420);
  uint8_t *nv12_data = yuv_mat.ptr<uint8_t>();

  // copy y data
  auto data = input->sysMem[0].virAddr;
  int32_t y_size = input_h * input_w;
  memcpy(reinterpret_cast<uint8_t *>(data), nv12_data, y_size);

  // copy uv data
  int32_t uv_height = input_h / 2;
  int32_t uv_width = input_w / 2;
  uint8_t *nv12 = reinterpret_cast<uint8_t *>(data) + y_size;
  uint8_t *u_data = nv12_data + y_size;
  uint8_t *v_data = u_data + uv_height * uv_width;

  for (int32_t i = 0; i < uv_width * uv_height; i++) {
    *nv12++ = *u_data++;
    *nv12++ = *v_data++;
  }
  return 0;
}











