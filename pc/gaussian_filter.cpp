#include "gaussian_filter.h"
#include <cmath>
#include <opencv2/opencv.hpp>

namespace taichi_aot {
void GAUSSIAN_APP::prepare() {
    if (!hasCreated_) {std::cout << "wrong in create!! "<< std::endl;}
    gaussian_kernel_ = aotModule_.get_kernel("gaussian_blur");
    uint32_t sz = 2 * radius_ + 1;
    weight_ = runtimePtr_->allocate_ndarray<float>({sz}, {1}, true);
    std::vector<float> weight(sz, 0);
    float total = 0.0;
    for(int i = 0; i < sz; i++) {
        float tmp = (i - radius_) / sigma_;
        float val = std::exp(-0.5 * tmp * tmp);
        weight[i] = val;
        total += val;
    }
    for(auto &w : weight) {
        w /= total;
    }

    // load image
    cv::Mat img = cv::imread("./build/assets/bench_case/mountain.jpg");
    img_ = runtimePtr_->allocate_ndarray<uint8_t>({(unsigned int)img.rows, (unsigned int)img.cols}, {(unsigned int)img.elemSize()}, true);
    uint8_t *img_ptr = static_cast<uint8_t*>(img_.map());
    std::memcpy(img_ptr, img.data, img.total() * img.elemSize());
    img_.unmap();
    gaussian_kernel_[0] = img_;
    
    float *w_ptr = static_cast<float*>(weight_.map());
    std::memcpy(w_ptr, weight.data(), weight.size() * sizeof(float));
    weight_.unmap();
    gaussian_kernel_[1] = weight_;

    // tmp memory
    blurtemp_ = runtimePtr_->allocate_ndarray<uint8_t>({(unsigned int)img.rows, (unsigned int)img.cols}, {3}, true);
    uint8_t *blur_ptr = static_cast<uint8_t*>(blurtemp_.map());
    std::memset(blur_ptr, 0, img.total() * img.elemSize());
    blurtemp_.unmap();
    gaussian_kernel_[2] = blurtemp_;
    start_ = std::chrono::steady_clock::now();
}

void GAUSSIAN_APP::run() {
    if (!hasCreated_) {std::cout << "wrong in create!! "<< std::endl;}
    gaussian_kernel_.launch();
    runtimePtr_->wait();
}

void GAUSSIAN_APP::output() {
    end_ = std::chrono::steady_clock::now();
    std::cout << "gaussian cost :" << std::chrono::duration_cast<std::chrono::milliseconds>(end_ - start_).count()/ 1000.0 << std::endl;
    TiNdShape shape =  img_.shape();
    float* img_ptr = static_cast<float*>(img_.map());
    cv::Mat img(shape.dims[0], shape.dims[1], CV_8UC3, img_ptr);
    cv::imwrite("./build/assets/bench_case/mountain_blur.jpg", img);
    img_.unmap();

    // vs opencv cost
    img = cv::imread("./build/assets/bench_case/mountain.jpg");
    cv::Mat img_cv_blur;
    start_ = std::chrono::steady_clock::now();
    int sz = 2 * radius_ + 1;
    cv::GaussianBlur(img, img_cv_blur, cv::Size(sz, sz), sigma_, sigma_);
    end_ = std::chrono::steady_clock::now();
    std::cout << "opencv gaussian cost :" << std::chrono::duration_cast<std::chrono::milliseconds>(end_ - start_).count()/ 1000.0 << std::endl;

}

GAUSSIAN_APP::~GAUSSIAN_APP() {

}

}
