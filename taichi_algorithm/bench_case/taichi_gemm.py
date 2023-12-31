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

@ti.kernel
def gemm(C_f: ti.types.ndarray(dtype=ti.f32, ndim=2), A_f: ti.types.ndarray(dtype=ti.f32, ndim=2), B_f: ti.types.ndarray(ti.f32, ndim=2), k:ti.int32):
    for i,j in C_f:
        for ke in range(k):
            C_f[i, j] += A_f[i, ke] * B_f[ke, j]

if __name__ == "__main__":
    if run_type == RUN_TYPE.RUN_DIRECT:
        m = 2
        n = 2
        k = 1

        A_f = ti.ndarray(dtype=ti.f32, shape=(m, k))
        B_f = ti.ndarray(dtype=ti.f32, shape=(k, n))
        C_f = ti.ndarray(dtype=ti.f32, shape=(m, n))

        A_f[0, 0] = 0.0
        A_f[1, 0] = 1.0
        B_f[0, 0] = 1.0
        B_f[0, 1] = 2.0

       # for i,j in A_f:
       #     A_f[i, j] = ti.random(float)

       # for i,j in B_f:
       #     B_f[i, j] = ti.random(float)

        gemm(C_f, A_f, B_f, k)
        ti.sync()
        print(C_f[0, 0])
        print(C_f[0, 1])
        print(C_f[1, 0])
        print(C_f[1, 1])

    elif run_type == RUN_TYPE.COMPILE:
        mod = ti.aot.Module(ti.vulkan)
        mod.add_kernel(gemm)
        mod.archive("taichi_gemm.tcm")
