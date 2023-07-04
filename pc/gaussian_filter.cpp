#include "gaussian_filter.h"
#include <cmath>
#include <opencv2/opencv.hpp>

namespace taichi_aot {
void GAUSSIAN_APP::prepare() {
    if (!hasCreated_) {std::cout << "wrong in create!! "<< std::endl;}
    gaussian_kernel_ = aotModule_.get_kernel("taichi_gaussian.tcm");
    uint32_t sz = 2 * radius_ + 1;
    weight_ = runtimePtr_->allocate_ndarray<float>({sz}, {1}, true);
    std::vector<float> weight(sz, 0);
    float total = 0.0;
    for(int i = 0; i < sz; i++) {
        float tmp = -0.5 * (i - radius_) / sigma_;
        float val = std::exp(tmp * tmp);
        weight[i] = val;
        total += val;
    }
    for(auto &w : weight) {
        w /= total;
    }

    float *w_ptr = static_cast<float*>(weight_.map());
    std::memcpy(w_ptr, weight.data(), weight.size());
    weight_.unmap();
    gaussian_kernel_[1] = weight_;

    // load image
    cv::Mat img = cv::imread("./mountain.jgp");
    img_ = runtimePtr_->allocate_ndarray<uint8_t>({img.total()}, {1}, true);
    uint8_t *img_ptr = static_cast<uint8_t*>(img_.map());
    std::memcpy(img_ptr, img.data(), img.total());
    img_.unmap();
    gaussian_kernel_[0] = img_;

    // tmp memory
    ti::NdArray temp = runtimePtr_->allocate_ndarray<uint8_t>({img.total()}, {1}, true);
    gaussian_kernel_[2] = temp;
}

void GAUSSIAN_APP::run() {
    if (!hasCreated_) {std::cout << "wrong in create!! "<< std::endl;}
    gaussian_kernel_.launch();
    runtimePtr_->wait();
}

void GAUSSIAN_APP::output() {

}

}
