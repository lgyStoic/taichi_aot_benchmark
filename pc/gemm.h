#ifndef __GEMM_H__
#define __GEMM_H__
#include "base.h"
#include <string>
namespace taichi_aot {
class GEMM_APP: public AOT_APP {
public:

    void prepare() override;

    void run() override;

    void output() override;

    ~GEMM_APP() override;

private:
    ti::Kernel gemm_kernel_;
    ti::NdArray<float> output_;
    ti::NdArray<float> A_;
    ti::NdArray<float> B_;

};
}
#endif