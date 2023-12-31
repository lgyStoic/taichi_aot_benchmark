#ifndef __GAUSSIAN_FILTER_H__
#define __GAUSSIAN_FILTER_H__
#include "base.h"
namespace taichi_aot {
class GAUSSIAN_APP: public AOT_APP {
public:

    void prepare() override;

    void run() override;

    void output() override;

    ~GAUSSIAN_APP() override;

private:
    ti::Kernel gaussian_kernel_;
    ti::NdArray<uint8_t> img_;
    ti::NdArray<uint8_t> blurtemp_;
    ti::NdArray<float> weight_;

    const float sigma_ = 20.0;
    const int radius_ = sigma_* 3;
};
}

#endif