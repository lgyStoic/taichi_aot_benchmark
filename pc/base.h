#ifndef __TAICHI_BASE_H__
#define __TAICHI_BASE_H__
#include <string>
#include <taichi/cpp/taichi.hpp>
#include <memory>

namespace taichi_aot {
struct  AOT_APP {

AOT_APP() = delete;
AOT_APP(const AOT_APP&) = delete;
AOT_APP& operator=(const AOT_APP&) = delete;

virtual AOT_APP& create_name(std::string &modulePath, TiArch arch) {
    if(!runtimePtr_) {
        runtimePtr_ = std::make_unique<ti::Runtime>(arch);
    }

    aotModule_ = runtimePtr_->load_aot_module(modulePath);
    assert(aotModule_.is_valid());
    return *this;
};

virtual void prepare() = 0;

virtual void run() = 0;

virtual ~AOT_APP();

static std::unique_ptr<ti::Runtime> runtimePtr_;
ti::AotModule aotModule_;
};
}

#endif