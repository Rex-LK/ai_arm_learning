## [旭日x3] 动手实践之bpu_rezie以及简化cpp编译流程
### 1、前言
在x3开发者手册里面的利用bpu进行resize的操作,于是想着在板端上测试一下，对比了一下bpu-resize与opencv-resize的时间差异。同时之前一直是在docker环境下进行编译的,稍显麻烦,而cpp编译只依赖交叉编译工具和依赖文件,交叉编译工具在docker下的/opt/gcc-ubuntu-9.3.0-2020.03-x86_64-aarch64-linux-gnu下,将它复制出来到主机上,就可以在不依赖docker进行编译了,如果编译起来有问题,也可以从本文的百度云链接获得完整的依赖文件以及源代码。
本文测试代码:
百度云完整依赖文件以及源代码:
#### 2、简化cpp编译环境
在本机中export交叉编译工具的路径
```
export LD_LIBRARY_PATH=..../ai_arm_learning/x3/datagcc-ubuntu-9.3.0-2020. 03-x86_64-aarch64-linux-gnu/lib/x86_64-linux-gnu     ## ....为实际路径
```
如果不使用交叉编译工具时，即 注释掉# SET(tar x3)，在主机运行yolo_demo时会出现
```
./yolo_demo: error while loading shared libraries: libhbdk_sim_x86.so: cannot open shared object file: No such file or directory
```
同样的export对应的库路径即可
```
export LD_LIBRARY_PATH=..../ai_arm_learning/x3/data/deps/x86/dnn_x86/lib   ## ....为实际路径
```
有了上面的过程，就可以愉快的不依赖docker进行交叉编译了，下面就开始本文的正题吧。

#### 3、使用bpu进行resize
在x3_inference/sample/resize_demo.cpp中实现了三种不同图片格式的resize方法,yuv420、rgb、bgr,最终调用的api接口为 hbDNNResize
```cpp
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
```
实现resize的头文件 bpu_resize.hpp
```cpp
class BpuResize{
public:
    BpuResize(const BpuResize& other) = delete; 
    BpuResize& operator = (const BpuResize& other) = delete;
    explicit BpuResize(const int input_w, const int input_h, const int output_w, const int output_h, const imageType imgType);
    ~BpuResize();
    void copy_image_2_input_tensor(uint8_t *image_data,hbDNNTensor *tensor);
    float *Resize(cv::Mat ori_img, const std::initializer_list<int> crop);
    int32_t prepare_input_tensor();
    int32_t prepare_output_tensor();
    //...
}
```
### 4、总结
本次测试简化了cpp遍历流程,为后续的上板测试省略的一定的步骤,同时对bpu的resize的接口进行了测试,发现如果原图为1920*1080,resize后的图片大小为640\*640时，opencv的resize需要40+ms,而bpu接口的时间只需要25+ms,但目前从小图放大时,opencv的resize更快一些,不知道这是否属于正常现象。