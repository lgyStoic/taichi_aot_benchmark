#ifndef __REDUCE_SUM_H__
#define __REDUCE_SUM_H__
#include "base.h"

namespace taichi_aot {

class REDUCE_APP:public AOT_APP {
public:

void prepare() override;

void run() override;

void output() override;
    
};

}
#endif