#include "nlmean.h"
#include <cmath>
#include <opencv2/opencv.hpp>

namespace taichi_aot {
void NLMEAN_APP::prepare() {
    if (!hasCreated_) {std::cout << "wrong in create!! "<< std::endl;}
    nlm_kernel_ = aotModule_.get_kernel("nlmean");
    copyBorder_kernel_ = aotModule_.get_kernel("copyBorder");

    // load image
    cv::Mat img = cv::imread("./build/assets/bench_case/house.jpg");
    img_ = runtimePtr_->allocate_ndarray<uint8_t>({(unsigned int)img.rows, (unsigned int)img.cols}, {(unsigned int)img.elemSize()}, true);
    uint8_t *img_ptr = static_cast<uint8_t*>(img_.map());
    std::memcpy(img_ptr, img.data, img.total() * img.elemSize());
    img_.unmap();
    int padrows = img.rows + halfKernelSize_ + halfSearchSize_;
    int padcols = img.cols + halfKernelSize_ + halfSearchSize_;
    img_pad_ = runtimePtr_->allocate_ndarray<uint8_t>({(unsigned int)padrows, (unsigned int)padcols}, {3}, true);
    img_dest_ = runtimePtr_->allocate_ndarray<uint8_t>({(unsigned int)padrows, (unsigned int)padcols}, {3}, true);

    copyBorder_kernel_[0] = img_;
    copyBorder_kernel_[1] = img_pad_;

    nlm_kernel_[0] = img_pad_;
    nlm_kernel_[1] = img_dest_;
    nlm_kernel_[2] = halfKernelSize_;
    nlm_kernel_[3] = halfSearchSize_;
    nlm_kernel_[4] = h_;
    
    // tmp memory
    start_ = std::chrono::steady_clock::now();
}

void NLMEAN_APP::run() {
    if (!hasCreated_) {std::cout << "wrong in create!! "<< std::endl;}
    copyBorder_kernel_.launch();
    nlm_kernel_.launch();
    runtimePtr_->wait();
}

void NLMEAN_APP::output() {
    end_ = std::chrono::steady_clock::now();
    std::cout << "nlm cost :" << std::chrono::duration_cast<std::chrono::milliseconds>(end_ - start_).count()/ 1000.0 << std::endl;
    TiNdShape shape =  img_dest_.shape();
    float* img_ptr = static_cast<float*>(img_dest_.map());
    cv::Mat img(shape.dims[0], shape.dims[1], CV_8UC3, img_ptr);
    cv::imwrite("./build/assets/bench_case/house_nlm.jpg", img);
    img_dest_.unmap();
    
    // vs opencv
    start_ = std::chrono::steady_clock::now();

    img = cv::imread("./build/assets/bench_case/house.jpg");
    cv::Mat imgdst = img.clone();
    cv::fastNlMeansDenoisingColored(img, imgdst, 20, 10, 7, 31);
    end_ = std::chrono::steady_clock::now();
    cv::imwrite("./build/assets/bench_case/house_opencv_nlm.jpg", imgdst);
    std::cout << "opencv nlm cost :" << std::chrono::duration_cast<std::chrono::milliseconds>(end_ - start_).count()/ 1000.0 << std::endl;

}

NLMEAN_APP::~NLMEAN_APP() {

}

}
