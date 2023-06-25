import taichi as ti
import argparse
from enum import Enum

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
m = 256
n = 4096
k = 4096

A_f = ti.field(ti.f32, shape=(m, k))
B_f = ti.field(ti.f32, shape=(k, n))
C_f = ti.field(ti.f32, shape=(m, n))

    
@ti.func
def inner(i, j):
    for ke in range(k):
        C_f[i, j] += A_f[i, ke] * B_f[ke, j]
    
@ti.kernel
def gemm():
    for i,j in C_f:
        inner(i, j)

if __name__ == "__main__":
    if run_type == RUN_TYPE.RUN_DIRECT: 
        gemm()
    elif run_type == RUN_TYPE.COMPILE:
        mod = ti.aot.Module(ti.vulkan)
        mod.add_kernel(gemm)
        mod.archive("taichi_gemm.tcm")