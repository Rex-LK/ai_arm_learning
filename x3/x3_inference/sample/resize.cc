#include "bpu_resize.h"
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
int main(int argc, char **argv) {

  
  BpuResize* resizer = new BpuResize(1352,900,2048,1024);
  string image_path = "../images/kite.jpg";
  Mat brgMat = imread(image_path);
  Mat yuvMat;
  cvtColor(brgMat, yuvMat, COLOR_BGR2YUV_I420);
  void* res = resizer->doResize(yuvMat);
//   cv::Mat crop_resized_bgr(640, 640, CV_8UC3);
//   memcpy(crop_resized_bgr.data,res,640*640*3);

//   cv::Mat a;
//   cvtColor(crop_resized_bgr, a, CV_YUV2BGR_I420);
//   cv::imwrite("1.png", crop_resized_bgr);
  return 0;
}


