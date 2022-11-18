## \[旭日x3] 动手实践之一个文件实现分割、检测cpp代码部署

### 1、前言

最近x3派上部署ai视觉算法，在最初接触到这x3派的时候，里面提供了一些python例子，而当需要部署cpp代码时,部署文档少之又少，应该在机器人平台上有一些案例，在x3派的cpp部署文档里面,仅发现一个图片分类的例子，相信论坛里面许多同学也遇到过类似的问题。因此经过一段时间的测试，结合一些开源的代码，实现了一个简易的cpp部署流程,目前代码没进行封装，仅供参考。
后处理代码参考:<https://github.com/shouxieai/tensorRT_Pro>
环境搭建以及yolov5模型转换参考:<https://developer.horizon.ai/forumDetail/107952931390742029>
本文测试代码参考:https://github.com/Rex-LK/ai_arm_learning
依赖库的百度云链接: https://pan.baidu.com/s/1aWFCIt1qmUzBIU8wfVZ3aA?pwd=zf59 
提取码: zf59
感谢上述大佬的开源代码
### 2、cpp部署代码

#### 2.1、代码来源

源文件来自于官方开发者horizon_xj3_open_explorer_v2.3.3_20220727/ddk/samples/ai\_toolchain/horizon\_runtime\_sample/code/00\_quick\_start文件夹,这个文件可以在官网下载， 该文件夹里面是一个cpp分类的案例，通过对这个文件的解析，可以知道整个流程大致可以分为五个部分

*   1、加载模型
*   2、分配内存
*   3、推理
*   4、后处理
*   5、释放内存
    分割和检测的区别只在于后处理过程，其他过程完全一致，于是只需要在分类的代码的基础上，加上分割和检测的代码，就可以使它能够进行分割以及检测，下面为部分后处理代码。

```cpp
// 加载模型
string model_path = "../FastestDet_nv12.bin";
auto modelFileName = model_path.c_str();
string image_path = "../det.png";
Mat image0 = imread("../det.png");
Mat image = image0.clone();
// ...
//配内存
//推理 
// 分割模型后处理
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
// yolov5s检测模型后处理
else if(detect_type == "detect"){
  int cols = 85;
  int num_classes = cols - 5;
  int rows = 25200;
  float confidence_threshold = 0.4;
  float nms_threshold = 0.6;
  vector<bbox> BBoxes = decodeBbox(data,rows,cols,confidence_threshold,num_classes);
  vector<bbox> resBBoxes = nms(BBoxes,nms_threshold);

  Mat drawMat = image0.clone();
  Mat show_img;
  resize(image0,show_img,Size(640,640));

  for (auto& box : resBBoxes) {
      rectangle(show_img, Point(box.left, box.top),
                    Point(box.right, box.bottom), Scalar(0, 255, 0), 2);
      putText(show_img, format("%.2f", box.confidence),
                  Point(box.left, box.top - 10), 0, 0.8,
                  Scalar(0, 0, 255), 2, 2);
  }
    imwrite("det.jpg",show_img);
    cout<<"Done, Save to det.jpg"<<endl;
}
// 释放内存
```

#### 2.2、CMakeLists.txt及依赖环境

编译本项目的依赖文件都可以在docker内/root/.horizon/ddk/xj3\_aarch64 以及horizon\_xj3\_open\_explorer\_v2.3.3\_20220727/ddk/samples/ai\_toolchain/horizon\_runtime\_sample/code/deps\_gcc9.3里面找到。同时也可以在上面的百度云链接进行下载。本项目支持在docker里面编译和运行、docker里面编译以及x3派上运行，通过是否注释 SET(tar x3)来进行控制。

```CMake
cmake_minimum_required(VERSION 2.8)
project(test)
# 如果在x3上运行，则不注释，如果需要在x86的容器上运行,则需要注释
SET(tar x3)

if(tar)
    message(STATUS "build arm")
    SET(CMAKE_C_COMPILER /opt/gcc-ubuntu-9.3.0-2020.03-x86_64-aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc)
    SET(CMAKE_CXX_COMPILER /opt/gcc-ubuntu-9.3.0-2020.03-x86_64-aarch64-linux-gnu/bin/aarch64-linux-gnu-g++)
else()
    message(STATUS "build x86")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ")

if(tar)
    set(DEPS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps/aarch64/) 
else()
    set(DEPS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps/x86/) 
endif()

add_definitions(-w)

if(tar)
    include_directories(
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${DEPS_DIR}/dnn/include
            ${DEPS_DIR}/glog/include
            ${DEPS_DIR}/gflags/include
            ${DEPS_DIR}/opencv/include)

    link_directories(
        ${DEPS_DIR}/dnn/lib
        ${DEPS_DIR}/appsdk/appuser/lib
        ${DEPS_DIR}/appsdk/appuser/lib/hbbpu
        ${DEPS_DIR}/glog/lib
        ${DEPS_DIR}/gflags/lib
        ${DEPS_DIR}/opencv/lib)

    include_directories(${LIB_DIR_OPENCV}/include/)
    link_directories(${LIB_DIR_OPENCV}/lib/)
    SET(LINK_libs dnn gflags glog opencv_world zlib dl rt pthread dnn)
else()
    include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${DEPS_DIR}/dnn_x86/include/dnn
        ${DEPS_DIR}/glog/include
        ${DEPS_DIR}/gflags/include
        ${DEPS_DIR}/opencv/include)

    link_directories(
        ${DEPS_DIR}/dnn_x86/lib
        ${DEPS_DIR}/glog/lib
        ${DEPS_DIR}/gflags/lib
        ${DEPS_DIR}/opencv/lib)
        SET(LINK_libs dnn hbdk_sim_x86 gflags glog opencv_world zlib dl rt pthread)
endif()


add_executable(run_x3 src/run_x3.cc)
target_link_libraries(run_x3 ${LINK_libs})

```
### 3、运行
```shell
cd x3_demo
mkdir build && cd build
cmake ..
make -j
./run_x3.
```
### 4、总结

经过一系列的以及测试，终于能愉快的在板子上进行cpp代码部署了，熟悉这个项目后，相信后续的开发流程也会变得更加简单，本文只是简单示意了一下cpp部署流程，代码还有很大的改进空间，希望以后有时间会在x3派上进行其他项目的测试。
