#ifndef __RUN__
#define __RUN__

#include <math.h>
#include <unistd.h>

#include <algorithm>
#include <set>
#include <string>
#include <unordered_map>
#include "aiShm.h"
#include "common.h"
#include "date_time.h"
#include "json_cpp.h"
#include "log.h"
#include "shm_handle.h"

#include "detect.h"
#include "loadImage.h"
#include "segment.h"
#include "x3m_resize.h"
/**
 * @brief ai模块 实际项目推理
 *
 */

class Run {
 public:
  Run(std::string aiConfigPath);
  ~Run(){};
  void doRun(cv::Mat& imageOri, uint64_t startLoadTime,int imgo_w, int imgo_h);

  void doSeg(cv::Mat& yuvMat);
  void doDet(cv::Mat& image);
  void writeResult();
  void doX3mResize(cv::Mat& yuvMat,cv::Mat&resizedYuvMat);

 private:
  void loadConfigParam(std::string& aiConfigPath);
  void segPostProcess();
  void updateModelParam() {
    seg->update_param();
    segNclass = seg->seg_nclass;
    segOutH = seg->seg_output_h;
    segOutW = seg->seg_output_w;
  }

 private:
  float cameraImageW = 640;
  float cameraImageH = 360;

  //resize
  bool useX3mResize = true;
  // yuv resize
  x3mResize* x3m_resize;
  cv::Mat resizedYuvMat;
  int resizeOutW = 2048;
  int resizeOutH = 1024;

  // seg inference
  bool useSeg = true;
  std::string segModelPath;
  Segment* seg;
  bool segUseArgmax;
  int segInputW = 2048;
  int segInputH = 1024;
  int segOutW = 256;
  int segOutH = 128;

  float outRatioW;
  float outRatioH;

  int segNclass = 5;
  // pre
  std::string configClassId;
  std::vector<int> classId;
  std::string configClassName;
  std::vector<std::string> className;
  std::unordered_map<int,std::string> classMap;
  std::string configDropThresh;
  std::vector<int> dropThresh;
  std::unordered_map<int,int> dropMap;

  // infer
  float* segRes;
  cv::Mat prob;

  // process
  std::set<int> resClass;
  cv::Mat tmpRangeMat;
  cv::Mat modelOutMat;
  cv::Mat modelOutDownMat;
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  int labelScale;
  int skip_stride = 1;

  // det inference
  bool useDet = false;
  std::string detModelPath;
  Detect* det;
  float* detRes;

  //共享内存
  std::string aiShmName;
  uint32_t aiShmSize;

  // write result
  json_t aiResult;
  aiShm* writeShm;

  // sys
  bool debugSaveSeg = false;
  bool debugSaveDet = false;
  uint64_t startTime;
  uint64_t resTimestamp;
  int timeSleep;
  string debugDir;
  ofstream ofs;

  // prog
  uint64_t tokenTime;

};
#endif
Run::Run(std::string aiConfigPath) {
  // 加载模型路径、尺寸等参数
  loadConfigParam(aiConfigPath);

  // 读写内存
  writeShm = new aiShm(aiShmName, aiShmSize);

  // 分割类
  seg = new Segment(segModelPath);

  segInputH = seg->seg_input_h;
  segInputW = seg->seg_input_w;
  segNclass = seg->seg_nclass;

  modelOutMat.create(segInputH, segInputW, CV_8UC1);
  modelOutDownMat.create(segInputH / skip_stride, segInputW / skip_stride, CV_8UC1);

    // 硬件resize
  x3m_resize = new x3mResize(cameraImageW,cameraImageH * 1.5,resizeOutW,resizeOutH * 1.5);

  if (useDet) {
    // 检测类
    det = new Detect(detModelPath);
  }
};

/**
 * @brief 从配置文件读取参数
 *
 */

void Run::loadConfigParam(std::string& aiConfigPath) {
  json_t aiParam;
  if (JsonCpp::ReadFile(aiConfigPath, aiParam)) {
    // modelParam
    // seg
    useSeg = aiParam["segParam"]["useSeg"];
    segModelPath = aiParam["segParam"]["segModelPath"];

    segModelPath = aiParam["segParam"]["segModelPath"];
    segOutW = aiParam["segParam"]["segOutW"];
    segOutH = aiParam["segParam"]["segOutH"];

    segUseArgmax = aiParam["segParam"]["segUseArgmax"];

    configClassName = aiParam["segParam"]["configClassName"];
    className = tools::param2vector_string(configClassName);

    configDropThresh = aiParam["segParam"]["configDropThresh"];
    dropThresh = tools::param2vector_int(configDropThresh);

    labelScale = aiParam["segParam"]["labelScale"];
    configClassId = aiParam["segParam"]["configClassId"];
    classId = tools::param2vector_int(configClassId);
    for (int k = 0; k < classId.size(); k++) {
      classId[k] *= labelScale;
      classMap[classId[k]] = className[k];
      dropMap[classId[k]] = dropThresh[k];
    }
    skip_stride = aiParam["segParam"]["skipStride"];
    debugSaveSeg = aiParam["segParam"]["debugSaveSeg"];

    //resize
    resizeOutW = aiParam["resizeParam"]["resizeOutW"];
    resizeOutH = aiParam["resizeParam"]["resizeOutH"];
    resizedYuvMat.create(resizeOutH * 1.5, resizeOutW, CV_8UC1);

    // det
    detModelPath = aiParam["detParam"]["detModelPath"];
    debugSaveDet = aiParam["detParam"]["debugSaveDet"];
    useDet = aiParam["detParam"]["useDet"];
    // shmParam
    aiShmName = aiParam["shmParam"]["aiShmName"];
    aiShmSize = aiParam["shmParam"]["aiShmSize"];
    bool shmExists = tools::dir_file_exists("/tmp/shm", 1);
    // sysParam
    timeSleep = aiParam["sysParam"]["timeSleep"];
    debugDir = aiParam["sysParam"]["saveDebugDir"];

    if (debugSaveSeg || debugSaveDet) {
      debugDir = debugDir + to_string(GetCurrentTickMs());
      LOGD("debugDir:%s", debugDir.c_str());
      std::string command = "mkdir -p " + debugDir;
      system(command.c_str());
    }

    // 将模型输出的尺度恢复到 原图大小 

    outRatioW = cameraImageW / segOutW;
    outRatioH = cameraImageH / segOutH;
  }
}
/**
 * @brief 进行推理、找边界、并写结果
 */

void Run::doRun(cv::Mat& image, uint64_t startLoadTime,int imgo_w = 640, int imgo_h = 360) {
  aiResult.clear();
  tokenTime = startLoadTime;
  aiResult["tokenTime"] = tokenTime;
  //是否进行硬件resize  
  if(useX3mResize){
    doX3mResize(image,resizedYuvMat);
    // 查看resize之后的图片
    // uint64_t resized_time = GetCurrentTickMs();
    // cv::Mat ResizebrgMat;
    // uint64_t t1 = GetCurrentTickMs();
    // cvtColor(resizedYuvMat,ResizebrgMat, cv::COLOR_YUV2BGR_I420);
    // cv::imwrite(to_string(t1) + "_resized_brg.png",ResizebrgMat);
  }
  // 是否进行分割
  if (useSeg) {
    doSeg(resizedYuvMat);
  }
  // if (useDet) {
  //   doDet(image);
  // }

  // 存原图
  if (debugSaveSeg || debugSaveDet) {
    //原图 yuv420 转 bgr 保存
    std::string save_oimg_path =
        debugDir + "/seg_oimg_" + to_string(tokenTime) + ".jpg";
    cv::Mat save_oimg;
    cv::cvtColor(image, save_oimg, CV_YUV2BGR_I420);
    cv::imwrite(save_oimg_path, save_oimg);
    LOGD("save ori img");
  }
  // 将分割结果写入到共享内存中
  writeResult();
  // 将分割结果写到文件中,仅调试使用
  if (debugSaveSeg) {
      LOGD("write result 2 txt");
      ofs.open(debugDir + "/aiResult.json", ios::app);
      ofs << tokenTime << endl;
      ofs << aiResult.dump() << endl;
      ofs << "-----------" << endl;
      ofs.close();
  }

  uint64_t endTime = GetCurrentTickMs();

  // 是否要进行sleep,以较少cpu的占用率
  // 根据发送结果的频率进sleep
  // sleep_for
  if (timeSleep && timeSleep > (endTime - tokenTime)) {
    Sleep(timeSleep - (endTime - tokenTime));
  }
  uint64_t allTime = GetCurrentTickMs();
  float fps = 1000.0f / (allTime - tokenTime);
  LOGD("fps = %f ", fps );

  // read
//   auto readShm = aiShm(aiShmName);
//   std::string readRes = readShm.Read(ShmSize);
}

void Run::doSeg(cv::Mat& yuvMat) {

  imgOp::yuv420_2_tensor_as_nv12(yuvMat, seg->infer_->input_tensors.data(),
                                 segInputH, segInputW);
  uint64_t t1 =  GetCurrentTickMs();
  //推理
  segRes = seg->inference();
  uint64_t t2 =  GetCurrentTickMs();
  LOGD("infer time:%d ms",t2 - t1)
  updateModelParam();

  uint64_t t3 =  GetCurrentTickMs();

  if (!segUseArgmax) {
    // useArgmax表示模型不能在 bpu上进行argmax
    if(debugSaveSeg){
      // debug 模式下 每个像素乘以一个比例, 便于观察结果
       modelOutDownMat = segmentation::decodeMaskImage(segRes, segOutW, segOutH, segNclass, resClass,labelScale);
      }
    else{
       modelOutDownMat = segmentation::decodeMaskImage(segRes, segOutW, segOutH, segNclass, resClass);
    }
  } else {
    if (debugSaveSeg) {
      // debug 模式下 每个像素乘以一个比例, 便于观察结果
      decodeMaskImage(segRes, modelOutDownMat, segOutW, skip_stride, resClass,
                      labelScale);
    } else {
      decodeMaskImage(segRes, modelOutDownMat, segOutW, skip_stride, resClass);
    }
  }
  uint64_t t4 = GetCurrentTickMs();
  LOGD("decode time:%d ms",t4 - t3)
  if (debugSaveSeg) {
    std::string save_mask_path =
        debugDir + "/seg_mask_" + to_string(tokenTime) + ".png";
    cv::imwrite(save_mask_path, modelOutDownMat);
    LOGD("save mask img");
  }
  // 分割模型后处理
  uint64_t t5 = GetCurrentTickMs();
  segPostProcess();
  uint64_t t6 = GetCurrentTickMs();
  LOGD("post time:%d ms",t6 - t5);
  LOGD("all time:%d ms",t6 - tokenTime);
}


/**
 * @brief 遍历mask出现的  种类 0~5之间
 */

//todo 还需要将 128 * 256 上的结果 重新 缩放会640 * 360 的尺度
void Run::segPostProcess() {
  for (int k = 0; k < classId.size(); ++k) {
    /*
        curClass -> int , 代表某一个类别
    */
    int curClass = classId[k];
    std::string curLabel = classMap[curClass];

    if(resClass.find(curClass) == resClass.end()){
      // 目前的类别没出现在 当前mask中
      std::vector<std::vector<std::vector<int>>> empty_all_edge;
      aiResult[curLabel] = empty_all_edge;
      continue;
    }

    // curClass == 0 为背景类
    if (curClass == 0) continue;
    contours.clear();
    hierarchy.clear();
    cv::inRange(modelOutDownMat, curClass, curClass, tmpRangeMat);
    // cv::imwrite(curLabel + ".png",tmpRangeMat);
    // 找到 该类别物体的轮廓  cv::RETR_CCOMP  cv::RETR_EXTERNAL
    cv::findContours(tmpRangeMat, contours, hierarchy, cv::RETR_CCOMP,
                     cv::CHAIN_APPROX_SIMPLE);

    // 如果mask被缩放,那这里的坐标还需要恢复到原来的尺寸(360,640) 乘以一个高宽系数
    std::vector<cv::Rect> boundRect(contours.size());
    std::vector<std::vector<std::vector<int>>> all_edge;
    std::vector<std::vector<int>> sig_edge;
    for (int i = 0; i < contours.size(); i++) {
    // 处理边界 对于小于一定阈值的 边界需要过滤
      if(contours[i].size() < dropMap[curClass]){
          continue;
      } 
      // 
      if(curLabel == "sward"){
        // 返回草地的所有坐标点
        for (int j = 0; j < contours[i].size(); j++) {
          int cx = contours[i][j].x * outRatioW;
          int cy = contours[i][j].y * outRatioH;
          std::vector<int> tmp = {cx, cy};
          sig_edge.emplace_back(tmp);
        }
      }
      else{
        // 返回其他物体的左上右下两个坐标点
        boundRect[i] = cv::boundingRect(cv::Mat(contours[i]));
        int x = boundRect[i].x * outRatioW;
        int y = boundRect[i].y * outRatioH;
        int width = boundRect[i].width * outRatioW;
        int height = boundRect[i].height * outRatioH;

        int left   = x - width  * 0.5;
        int top    = y - height * 0.5;
        int right  = x + width  * 0.5;
        int bottom = y + height * 0.5; 
        std::vector<int> left_top = {left, top};
        sig_edge.emplace_back(left_top);
        std::vector<int> right_bottom = {right, bottom};
        sig_edge.emplace_back(right_bottom);
      }

      all_edge.emplace_back(sig_edge);

      if (debugSaveSeg) {
      // 发送物体的左上右下两个点，当物体较多时，目前不太准
      // 计算该区域的最小外接矩形
        cv::resize(modelOutDownMat,modelOutDownMat,cv::Size(segInputW,segInputW));
        boundRect[i] = cv::boundingRect(cv::Mat(contours[i]));
        int x = boundRect[i].x * outRatioW;
        int y = boundRect[i].y * outRatioH;
        int width = boundRect[i].width * outRatioW;
        int height = boundRect[i].height * outRatioH;
        // aiResult[curLabel+ "_" +to_string(j)] = {x ,y ,x + width, y + height};
        cv::rectangle(modelOutDownMat, cv::Point(x, y),
                      cv::Point(x + width, y + height), cv::Scalar(255, 0, 0),
                      2, 8, 0);
      }

    }
    aiResult[curLabel] = all_edge;
  }
  if (debugSaveSeg) {
    cv::imwrite(debugDir + "/seg_drawContous_" +
                    to_string(GetCurrentTickMs()) + ".jpg",
                modelOutDownMat);
    LOGD("Done, Save to seg_drawContous.jpg");
  }
};

void Run::doDet(cv::Mat& image) {
  // param
  int oimg_w = image.cols;
  int oimg_h = image.rows;

  cv::Mat detBgrMat;
  int input_w = 352;
  int input_h = 352;
  cv::resize(image,detBgrMat,cv::Size(input_w,input_h));

  hbDNNTensor* input = det->infer_->input_tensors.data();
  auto data = input->sysMem[0].virAddr;
  memcpy(reinterpret_cast<uint8_t*>(data), detBgrMat.ptr<uint8_t>(),
         input_h * input_w * 3);

  detRes = det->inference();
  det->decodeBbox(oimg_w, oimg_h);
  det->detNms();
  if (debugSaveDet) {
    std::string save_det_path =
        debugDir + "/seg_det_" + to_string(tokenTime) + ".png";
    for (auto& box : det->box_result) {
      std::string label_name = coco_classes[box.class_label];
      cv::rectangle(detBgrMat, cv::Point(box.left, box.top),
                    cv::Point(box.right, box.bottom), cv::Scalar(0, 255, 0), 2);
      cv::putText(detBgrMat, cv::format("%.2f", box.confidence),
                  cv::Point(box.left, box.top - 10), 0, 0.8,
                  cv::Scalar(0, 0, 255), 2, 2);
      cv::putText(detBgrMat, label_name, cv::Point(box.left, box.top + 10), 0,
                  0.8, cv::Scalar(0, 0, 255), 2, 2);
    }
    cv::imwrite(save_det_path, detBgrMat);
    LOGD("save det img");
  }
}

void Run::writeResult() { 
  writeShm->Write(aiResult); 
}

void Run::doX3mResize(cv::Mat& yuvMat,cv::Mat&resizedYuvMat){
    x3m_resize->YuvResize(yuvMat,resizedYuvMat);
}