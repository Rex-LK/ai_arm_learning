cmake_minimum_required(VERSION 2.6)
# set (CMAKE_C_COMPILER "/usr/bin/gcc")
# set (CMAKE_CXX_COMPILER "/usr/bin/g++")

project(matrix_demo)

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -O0 -Wfatal-errors -pthread -w -g")

message(STATUS "${CMAKE_CURRENT_SOURCE_DIR}")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/OpenBLAS0.3.21/include
    include/
)

link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/OpenBLAS0.3.21/lib   
)
SET(LINK_libs openblas)

add_library(matrix STATIC src/matrix.cpp)

# add_executable(sample sample/sample.cpp)
# target_link_libraries(sample ${LINK_libs} matrix openblas)
add_executable(test1 sample/test1.cpp)
target_link_libraries(test1 ${LINK_libs} openblas)


