#include "gemm.h"
#include <iostream>
#include <taichi/taichi_core.h>
#include <opencv2/opencv.hpp>
namespace taichi_aot {


void GEMM_APP::prepare() {
    if (!hasCreated_) {std::cout << "wrong in create!! "<< std::endl;}
    gemm_kernel_ = aotModule_.get_kernel("gemm");
    uint32_t m = 256;
    uint32_t n = 4096;
    uint32_t k = 4096;

    // device memory 
    A_ = runtimePtr_->allocate_ndarray<float>({m,k}, {1}, true);
    B_ = runtimePtr_->allocate_ndarray<float>({k,n}, {1}, true);
    output_ = runtimePtr_->allocate_ndarray<float>({m,n}, {1}, true);
    std::vector<float> a_vec(m * k);
    std::vector<float> b_vec(k * n);
    srand(0);

    for(int i = 0; i < a_vec.size(); i++) {
        a_vec[i] = static_cast<float>(rand()) / static_cast <float> (RAND_MAX);;
    }

    for(int i = 0; i < b_vec.size(); i++) {
        b_vec[i] = static_cast<float>(rand()) / static_cast <float> (RAND_MAX);;
    }

    // host memory
    float *aptr = static_cast<float*>(A_.map());
    float *bptr = static_cast<float*>(B_.map());

    std::memcpy(aptr, a_vec.data(), sizeof(float) * a_vec.size());
    std::memcpy(bptr, b_vec.data(), sizeof(float) * b_vec.size());

    A_.unmap();
    B_.unmap();

    gemm_kernel_[0] = output_;
    gemm_kernel_[1] = A_;
    gemm_kernel_[2] = B_;
    gemm_kernel_[3] = (int32_t)k;
    start_ = std::chrono::steady_clock::now();
}                                                                

void GEMM_APP::run () {
    if (!hasCreated_) {std::cout << "wrong in create!! "<< std::endl;}
    gemm_kernel_.launch();
    runtimePtr_->wait();
    
}

void GEMM_APP::output() {
    if (!hasCreated_) {std::cout << "wrong in create!! "<< std::endl;}
    end_ = std::chrono::steady_clock::now();
    TiNdShape a_shape = A_.shape();
    TiNdShape b_shape = B_.shape();
    TiNdShape c_shape = output_.shape();

    std::cout << "taichi aot GFLOPS :" << static_cast<float>(2) * a_shape.dims[0] * a_shape.dims[1] * b_shape.dims[1] / (end_ - start_).count() << std::endl;

    float *a_ptr = static_cast<float*>(A_.map());
    float *b_ptr = static_cast<float*>(B_.map());
    cv::Mat Amat(a_shape.dims[0], a_shape.dims[1], CV_32FC1, a_ptr);
    cv::Mat Bmat(b_shape.dims[0], b_shape.dims[1], CV_32FC1, b_ptr);
    
    start_ = std::chrono::steady_clock::now();
    cv::Mat Cmat = Amat * Bmat;
    end_ = std::chrono::steady_clock::now();
    std::cout << "opencv GFLOPS :" << static_cast<float>(2) * a_shape.dims[0] * a_shape.dims[1] * b_shape.dims[1] / (end_ - start_).count() << std::endl;

    float *cptr = static_cast<float *>(output_.map());
    cv::Mat taichiCMat(c_shape.dims[0], c_shape.dims[1], CV_32FC1, cptr);

    for(int r = 0; r < Cmat.rows; r++) {
        for(int c = 0; c < Cmat.cols; c++) {
            float diff = Cmat.at<float>(r, c) - taichiCMat.at<float>(r, c);
            if(diff > 1e-5 || diff < -1e-5) {
                // std::cout << Cmat.at<float>(r, c) << " " << taichiCMat.at<float>(r, c) << " " << diff << std::endl;
            }
        }
    }
    
    A_.unmap();
    B_.unmap();
    output_.unmap();

}

GEMM_APP::~GEMM_APP() {
}
}