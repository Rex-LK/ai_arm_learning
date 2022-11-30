
#include "bpu_resize.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
using namespace std;
using namespace cv;

// 1、给定任意尺寸的图片
// 尺寸变化时，需要重新分配 输入输出的大小
// 2、给定固定尺寸的图片
// 3、resize 支持格式 bgr rgb nv12

int main(){
    string image_path = "../../images/kite.jpg";
    int oimg_w = 1920;
    int oimg_h = 1080;
    auto img = cv::imread(image_path);

    int resized_w = 640;
    int resized_h = 640;
    cv::resize(img,img,Size(oimg_w,oimg_h));
    Mat img_yuv;
    cv::cvtColor(img, img_yuv, cv::COLOR_BGR2YUV_I420);

    BpuResize* resizer = new BpuResize(oimg_w,oimg_h * 1.5,resized_w,resized_h * 1.5);
    Mat resizedYuvMat(resized_h * 1.5, resized_w, CV_8UC1);
    resizer->YuvResize(img_yuv,resizedYuvMat);

    cv::Mat ResizebrgMat;
    cv::cvtColor(resizedYuvMat,ResizebrgMat, cv::COLOR_YUV2BGR_I420);
    cv::imwrite("test_resized_brg.png", ResizebrgMat);
    return 0;
}
