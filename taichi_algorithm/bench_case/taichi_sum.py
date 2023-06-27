import taichi as ti
import argparse
from enum import Enum
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--type", default="run", type=str)

args = parser.parse_args()

class RUN_TYPE(Enum):
    COMPILE = 1
    RUN_DIRECT = 2

if args.type == 'run':
    run_type = RUN_TYPE.RUN_DIRECT
elif args.type == 'compile':
    run_type = RUN_TYPE.COMPILE

ti.init(arch=ti.vulkan)
if ti.lang.impl.current_cfg().arch != ti.vulkan:
    raise RuntimeError("Vulkan is not available.")

@ti.kernel
def reduce_sum_kernel(f: ti.types.ndarray(dtype=ti.f32, ndim=1))->ti.f32:
    sum = 0.0
    for i in f:
        sum += f[i]
    return sum

if __name__ == "__main__":
    n = 10000
    A_f = ti.ndarray(dtype=ti.f32, shape=(n))
    n_arr = np.random.rand(n)
    A_f.from_numpy(n_arr)
    sum = reduce_sum_kernel(A_f)
    print(sum)

