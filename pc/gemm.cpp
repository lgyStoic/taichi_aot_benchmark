#include "gemm.h"
#include <taichi/taichi_core.h>
namespace taichi_aot {


void GEMM_APP::prepare() {
    if (!hasCreated_) {std::cout << "wrong in create!! "<< std::endl;}
    gemm_kernel_ = aotModule_.get_kernel("gemm");
    uint32_t m = 2;
    uint32_t n = 2;
    uint32_t k = 1;

    // device memory 
    ti::NdArray<float> A = runtimePtr_->allocate_ndarray<float>({m,k}, {1}, true);
    ti::NdArray<float> B = runtimePtr_->allocate_ndarray<float>({k,n}, {1}, true);
    output_ = runtimePtr_->allocate_ndarray<float>({m,n}, {1}, true);

    // host memory
    A.write({0.0, 1.0});
    B.write({1.0, 2.0});

    gemm_kernel_[0] = output_;
    gemm_kernel_[1] = A;
    gemm_kernel_[2] = B;
    gemm_kernel_[3] = (int32_t)k;
}

void GEMM_APP::run () {
    if (!hasCreated_) {std::cout << "wrong in create!! "<< std::endl;}
    gemm_kernel_.launch();
    runtimePtr_->wait();
}

void GEMM_APP::output() {
    if (!hasCreated_) {std::cout << "wrong in create!! "<< std::endl;}
    float *cptr = static_cast<float *>(output_.map());
    for(int i = 0; i < 4; i++) {
        std::cout << *(cptr + i) << " ";
    }
    std::cout << std::endl;
    output_.unmap();
}

GEMM_APP::~GEMM_APP() {

}
}