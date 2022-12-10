#include <iostream>
#include <vector>
#include <cstdlib>

#include "openblas/cblas.h"
#include "openblas/lapacke.h"
using namespace std;
// 随机生成20以内的给定尺寸数组
static void RandomFill(std::vector<float>& numbers,size_t size);
// 打印数组元素的函数
static void Print(const std::vector<float>& numbers);
// vector是一维的，输出是个矩阵，那输出的时候就要指定有几行几列
static void Print(const std::vector<float>& numbers, int rows, int cols);
// 寻找数组中最大的那个元素的索引和值
static void TestLevel1();
// 测试Level2里面最常用的函数：向量和矩阵的乘积
static void TestLevel2();
static void TestLevel3();


namespace Level1{
    void maxValue(vector<float> a){
        cout<<"Level1::maxValue"<<endl;
        size_t maxIndex = cblas_isamax(a.size(), a.data(), 1);
        cout <<"maxIndex:" << maxIndex << endl;
        cout <<"maxValue:" << a [maxIndex] << endl; 
    }
    // 矢量大小和(函数)
    void asum(vector<float> a){
        cout<<"Level1::sum"<<endl;
        float sum = cblas_sasum(a.size(), a.data(), 1);
        cout <<"sum:" << sum << endl;
    }
    // 标量-向量乘积(例程) b = [alpha * a] + b
    void axpy(float alpha,vector<float> a,vector<float> b){
        cout<<"Level1::axpy"<<endl;
        cblas_saxpy(a.size(),alpha, a.data(),1,b.data(),1);
        Print(b);
    }
    // 拷贝向量(例程)
    void copy(vector<float> a,int step_a ,vector<float> b,int step_b){
        cout<<"Level1::copy"<<endl;
        cblas_scopy(a.size(), a.data(), step_a ,b.data(),step_b);
        Print(b);
    }
    // 点积(函数)
    void dot(vector<float> a,int step_a ,vector<float> b,int step_b){
        cout<<"Level1::dot"<<endl;
        float res =  cblas_sdot(a.size(), a.data(),step_a,b.data(),step_b);
        cout<<"dot res::" <<res<<endl;
    }
    // 向量的2范数(欧几里得范数)(函数)
    void nrm2(vector<float> a,int step_a){
        cout<<"Level1::nrm2"<<endl;
        float res =  cblas_snrm2(a.size(), a.data(),step_a);
        cout<<"nrm2 res::" <<res<<endl;
    }
    // 向量-标量点积（例程）
    void scal(vector<float> a,float alpha,int step_a){
        cout<<"Level1::scal"<<endl;
        Print(a);
        cblas_sscal(a.size(), alpha , a.data(),step_a);
        Print(a);
    }
    
    // 向量交换(例程)
    void swap(vector<float> a,int step_a,vector<float> b,int step_b){
        cout<<"Level1::swap"<<endl;
        Print(a);
        Print(b);
        cblas_sswap(a.size(), a.data(),step_a,b.data(),step_b);
        Print(a);
        Print(b);
    }
}


int main(int argc, const char * argv[]) {

    vector<float> a = {1.0f,2.0f,3.0f,4.0f};
    vector<float> b = {4.0f,5.0f,6.0f,7.0f};
    Level1::maxValue(a);
    Level1::asum(a);
    Level1::axpy(1,a,b);
    Level1::copy(a,1,b,1);
    Level1::dot(a,1,b,1);
    Level1::nrm2(a,1);
    Level1::scal(a,2,1);
    return 0;
}

void RandomFill(std::vector<float>& numbers, size_t size) {
    numbers.resize(size);
    for (size_t i = 0; i != size; ++ i) {
        numbers[i] = static_cast<float>(rand() % 20);
    }
}

void Print(const std::vector<float>& numbers) {
    for (float number : numbers) {
        std::cout << number << ' ';
    }
    std::cout << std::endl;
}

void Print (const std:: vector<float>& numbers ,int  rows, int cols) {
    for (int row =0; row != rows; ++ row) {
        for (int col = 0; col != cols; ++ col) {
            std::cout << numbers[row * cols + col] << ' ';
        }
        std::cout << std::endl;
    }
}

static void TestLevel1() {
    const int VECTOR_SIZE = 4;
    std::vector<float> fv1;
    
    RandomFill(fv1, VECTOR_SIZE);
    Print(fv1);
    size_t maxIndex = cblas_isamax(VECTOR_SIZE, fv1.data(), 1);
    std::cout << maxIndex << std::endl;
    std::cout << fv1[maxIndex] << std::endl; 
}

static void TestLevel2()
{
    const int M = 3;
    const int N = 2;

    std::vector<float> a;
    std::vector<float> x;
    std::vector<float> y;
    
    RandomFill(a, M * N);
    RandomFill(x, N);
    RandomFill(y, M);
    
    std::cout << "A" << std::endl;
    Print(a, M, N);
    std::cout << "x" << std::endl;
    Print(x);
    std::cout << "y" << std::endl;
    Print(y);
    
    /*
     我们的目标是想计算这么一个公式：
     y := alpha * A * x + beta * y
     A:是一个矩阵，x是一个向量，所以我希望说去计算一个矩阵和向量的乘积。alpha是一个乘积的缩放，
     beta是对y的缩放，
     相当于把y里面的数字乘以beta，再加上A矩阵和向量的乘积。
     
     那这边有一个特例，假如我y里面都是0，或这beta是0的情况下，我就可以把公式看成：
    // y := alpha * A * x
     
     这个函数名称为：cblas_sgemv（）
     // s:single 单精度浮点数
     // ge: 是一个乘法
     // m: matrix
     // v: vector
     */

    /**
     参数解释：
     param CblasRowMajor 行主序还是列主序，默认行主序，何为主序：即数组存储元素的方式--按行存储还是按列存储，行主序：00，01，列主序00，10
     param CblasNoTrans 矩阵是否需要转置，不需要转置，如果需要转置的话，运算的时候它会自动做转置
     param M 矩阵的行数
     param N 矩阵的列数
     param 1.0f alpha ，我们设为1
     param a.data a矩阵的缓冲区首地址
     param lda a矩阵的列数
     param x.data x矩阵的缓冲区首地址
     param 1 x里面每次跳跃累加的个数，默认为1
     param 2.0f beta对y的缩放值
     param y.data y矩阵的缓冲区首地址
     param 1 y里面每次跳跃累加的个数，默认为1
     */
    int lda = N;
    cblas_sgemv(CblasRowMajor, CblasNoTrans, M, N, 1.0f, a.data(), lda, x.data(), 1, 2.0f, y.data(), 1);
    std::cout << "result y" << std::endl;
    Print(y);
}
static void TestLevel3() {
    // 我们希望计算两个矩阵的乘积，我们就需要定义三个参数M、N、K。
    const int M = 3;
    const int N = 2;
    const int K = 4;
    
    std::vector<float> a;
    std::vector<float> b;
    std::vector<float> c;
    
    RandomFill(a, M * K);
    RandomFill(b, K * N);
    RandomFill(c, M * N);
    
    // 输出A、B、C 三个矩阵
    std::cout << "A" << std::endl;
    Print(a, M, K);
    
    std::cout << "B" << std::endl;
    Print(b, K, N);
    
    std::cout << "C" << std::endl;
    Print(c, M, N);
    
    /*
     我们的目标是计算这么一个公式：
    
    // C := alpha * A * B + beta * C
    // 如果只想做两个矩阵的乘法，beta设成0就好了，变为如下式子：
    // C := alpha * A * B
      */

    /*
     函数释义：
     sgemm:矩阵间的单精度乘法
     s:single 单精度
     ge：general 一般的，普通的
     m：matix 矩阵
     m：multiplication 乘法
     */
    
    /**
     参数释义：
     param CblasRowMajor 行主序还是列主序，默认行主序
     param CblasNoTrans A需不需要转置，不需要
     param CblasNoTrans B需不需要转置，不需要
     param M 系数M
     param N 系数N
     param K 系数K
     param 1.0f alpha 设为1
     param a.data a的缓冲区首地址
     param lda a的列数
     param b.data b的缓冲区首地址
     param ldb b的列数
     param 1.0f beta 设为1
     param c.data c的缓冲区首地址
     param ldc c的列数
     */
  
    int lda = K;
    int ldb = N;
    int ldc = N;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, a.data(), lda, b.data(), ldb, 1.0f, c.data(), ldc);
    std::cout << "Result C" << std::endl;
    // 三行四列的矩阵 * 四行二列的矩阵 + 三行二列的矩阵，结果为一个三行二列的矩阵
    Print(c, M, N);
}
