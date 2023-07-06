#include "reduce_sum.h"

namespace taichi_aot {

void REDUCE_APP::prepare() {
    if (!hasCreated_) {std::cout << "wrong in create!! "<< std::endl;}
    reduce_kernel_ = aotModule_.get_kernel("reduce_sum_kernel");
    uint32_t n = 10000;

    A_ = runtimePtr_->allocate_ndarray<float>({n}, {1}, true);
    std::vector<float> a_vec(n, 0);
    for(int i = 0; i < n; i++) {
        a_vec[i] = static_cast<float>(rand()) / static_cast <float> (RAND_MAX);;
    }
    float *a_ptr = static_cast<float *>(A_.map());
    std::memcpy(a_ptr, a_vec.data(), sizeof(float) * n);
    A_.unmap();
    reduce_kernel_[0] = A_;
}

void REDUCE_APP::run() {
    reduce_kernel_.launch();
    runtimePtr_->wait();
}

void REDUCE_APP::output() {
    float *a_ptr = static_cast<float *>(A_.map());
    sumValue_ = *a_ptr;
    A_.unmap();
    std::cout << "result :" << sumValue_ << std::endl;
}

}