
// #include "bpu_resize.h"
#include "bpu_resize.hpp"

#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "common.hpp"
using namespace std;
using namespace cv;

// 1、给定任意尺寸的图片
// 尺寸变化时，需要重新分配 输入输出的大小
// 2、给定固定尺寸的图片
// 3、resize 支持格式 bgr rgb nv12

int main(int argc,char*argv[]){
    string image_path = "../../data/images/kite.jpg";
    string test_case = argv[1];
    int oimg_w = 1920;
    int oimg_h = 1080;
    auto bgrImg = cv::imread(image_path);

    int resized_w = 640;
    int resized_h = 640;

    cv::resize(bgrImg,bgrImg,Size(oimg_w,oimg_h));

    if(test_case == "YUV420"){
        Mat yuvImg;
        cv::cvtColor(bgrImg, yuvImg, cv::COLOR_BGR2YUV_I420);
        BpuResize* resizer = new BpuResize(oimg_w,oimg_h,resized_w,resized_h,imageType::YUV420);
        long t1 = tools::get_current_time();
        float* res = resizer->Resize(yuvImg);
        long t2 = tools::get_current_time();
        cout <<"bpu resize:" <<t2 - t1 << endl;
        Mat ResizedYuvMat(resized_h * 1.5, resized_w, CV_8UC1);
        memcpy(ResizedYuvMat.data,res,resized_w * resized_h * 1.5);
        cv::Mat ResizedBgrMat;
        cv::cvtColor(ResizedYuvMat,ResizedBgrMat, cv::COLOR_YUV2BGR_I420);
        cv::imwrite("test_resized_yuv.png", ResizedBgrMat);
    }
    else if (test_case == "BGR"){
        BpuResize* resizer = new BpuResize(oimg_w,oimg_h,resized_w,resized_h,imageType::BGR);
        long t1 = tools::get_current_time();
        float* res = resizer->Resize(bgrImg);
        long t2 = tools::get_current_time();
        cout <<"bpu resize:" <<t2 - t1 << endl;
        Mat ResizedBgrMat(resized_h , resized_w, CV_8UC3);
        memcpy(ResizedBgrMat.data,res,resized_w * resized_h * 3);
        cv::imwrite("test_resized_bgr.png", ResizedBgrMat);
    }
    else if (test_case == "RGB"){
        cv::Mat rgbImg;
        cv::cvtColor(bgrImg, rgbImg, cv::COLOR_BGR2RGB);
        BpuResize* resizer = new BpuResize(oimg_w,oimg_h,resized_w,resized_h,imageType::RGB);
        long t1 = tools::get_current_time();
        float* res = resizer->Resize(rgbImg,{0,0,2000,2000});
        long t2 = tools::get_current_time();
        cout <<"bpu resize:" <<t2 - t1 << endl;
        Mat ResizedRgbMat(resized_h , resized_w, CV_8UC3);
        memcpy(ResizedRgbMat.data,res,resized_w * resized_h * 3);
        cv::Mat ResizedBgrMat;
        cv::cvtColor(ResizedRgbMat,ResizedBgrMat, cv::COLOR_RGB2BGR);
        cv::imwrite("test_resized_rgb.png", ResizedBgrMat);
    }
    return 0;
}
