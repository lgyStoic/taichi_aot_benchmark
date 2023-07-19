#ifndef __NLM_H__
#define __NLM_H__
#include "base.h"
namespace taichi_aot {
class NLMEAN_APP: public AOT_APP {
public:

    void prepare() override;

    void run() override;

    void output() override;

    ~NLMEAN_APP() override;

private:
    ti::Kernel copyBorder_kernel_;
    ti::Kernel nlm_kernel_;
    ti::NdArray<uint8_t> img_;
    ti::NdArray<uint8_t> img_pad_;
    ti::NdArray<uint8_t> img_dest_;

    int halfKernelSize_ = 3;   
    int halfSearchSize_ = 15;

    float h_ = 20.0;

};
}

#endif