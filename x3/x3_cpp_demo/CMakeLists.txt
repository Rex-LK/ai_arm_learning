cmake_minimum_required(VERSION 2.8)
project(test)

# SET(tar x3j)
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
            ${DEPS_DIR}/opencv/include)

    link_directories(
        ${DEPS_DIR}/dnn/lib
        ${DEPS_DIR}/appsdk/appuser/lib
        ${DEPS_DIR}/appsdk/appuser/lib/hbbpu
        ${DEPS_DIR}/opencv/lib)

    include_directories(${LIB_DIR_OPENCV}/include/)
    link_directories(${LIB_DIR_OPENCV}/lib/)
    SET(LINK_libs dnn  opencv_world zlib dl rt pthread dnn)
else()
    include_directories(
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${DEPS_DIR}/dnn_x86/include/dnn
            ${DEPS_DIR}/opencv/include)

    link_directories(
            ${DEPS_DIR}/dnn_x86/lib
            ${DEPS_DIR}/opencv/lib
            )
    SET(LINK_libs dnn hbdk_sim_x86  opencv_world zlib dl rt pthread)

endif()


add_executable(run_x3j src/run_x3j.cc)
target_link_libraries(run_x3j ${LINK_libs})


