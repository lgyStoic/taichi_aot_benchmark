import taichi as ti
import argparse
from enum import Enum
import numpy as np
import cv2 as cv
import time

parser = argparse.ArgumentParser()
parser.add_argument("--type", default="run", type=str)

img2d = ti.types.ndarray(dtype=ti.types.vector(3, ti.u8), ndim=2)

args = parser.parse_args()

class RUN_TYPE(Enum):
    COMPILE = 1
    RUN_DIRECT = 2

if args.type == 'run':
    run_type = RUN_TYPE.RUN_DIRECT
elif args.type == 'compile':
    run_type = RUN_TYPE.COMPILE

ti.init(arch=ti.vulkan)


@ti.kernel
def nlmean(img_pad: img2d, img_dest: img2d, halfKernelSz: ti.i32, halfSearchSz: ti.i32, h:ti.f32):
    rows:ti.i32 = img_pad.shape[0]
    cols:ti.i32 = img_pad.shape[1]
    total:ti.i32 = rows * cols
    borderSz:ti.i32 = halfKernelSz + halfSearchSz
    kernelSz:ti.i32 = 2 * halfKernelSz + 1
    h2: ti.f32 = h * h
    for idx in range(total):
        i: ti.i32 = idx / cols
        j: ti.i32 = idx % cols
        if i < borderSz or i >= rows - borderSz:
            pass
        elif j < borderSz or j >= cols - borderSz:
            pass
        else:
            newPixel = ti.math.vec3(0.0, 0.0, 0.0)
            sumw = ti.math.vec3(0.0, 0.0, 0.0)

            for sr in range(-halfSearchSz, halfSearchSz + 1):
                for sc in range(-halfSearchSz, halfSearchSz + 1):
                    # mse block
                    patchA_i: ti.i32 = sr + i
                    patchA_j: ti.i32 = sc + j
                    mse = ti.math.vec3(0.0, 0.0, 0.0)

                    for kr in range(-halfKernelSz, halfKernelSz):
                        for kc in range(-halfKernelSz, halfKernelSz):
                            diff = (img_pad[kr + patchA_i, kr + patchA_j]).cast(ti.f32) - (img_pad[kr + i, kc + j]).cast(ti.f32)
                            mse += diff * diff
                    mse /= (kernelSz * kernelSz)
                    weight  = ti.math.exp(-mse / h2)
                    newPixel += weight * (img_pad[i + sr, j + sc].cast(ti.f32))
                    sumw += weight
            img_dest[i, j] = (newPixel / sumw).cast(ti.u8)
           
# img pad with copy border
@ti.kernel
def copyBorder(img_src: img2d, img_pad: img2d, top: ti.int32, down: ti.int32, left: ti.int32, right: ti.int32):
    rows = img_src.shape[0]
    cols = img_src.shape[1]
    # copy major
    for i, j in img_src:
        img_pad[i + top, j + left] = img_src[i, j]
    # pad left
    for j in range(0, left):
        for i in range(top, top + rows):
            img_pad[i, j] = img_src[i, 0]
    # pad right 
    for j in range(left + cols - 1, left + cols + right):
        for i in range(top, top + rows):
            img_pad[i, j] = img_src[i, left + cols - 1]
    # pad top
    for i in range(0, top):
        for j in range(0, left + cols + right):
            img_pad[i, j] = img_pad[top, j]
    # pad down
    for i in range(top + rows, top + rows + down):
        for j in range(0, left + cols + right):
            img_pad[i, j] = img_pad[top + rows - 1, j]
    
        
if __name__ == "__main__":
    if run_type == RUN_TYPE.RUN_DIRECT:
        img = cv.imread('./house.jpg')
        halfKernelSz = 5
        halfSearchSz = 15
        h = 20
        img_dest = ti.ndarray(dtype=ti.types.vector(3, ti.u8), shape=(img.shape[0] + 2 * (halfSearchSz + halfKernelSz), img.shape[1] + 2 * (halfKernelSz + halfSearchSz)))
        img_pad = ti.ndarray(dtype=ti.types.vector(3, ti.u8), shape=(img.shape[0] + 2 * (halfSearchSz + halfKernelSz), img.shape[1] + 2 * (halfKernelSz + halfSearchSz)))
        padsize = halfSearchSz + halfKernelSz
        start = time.time()
        copyBorder(img, img_pad, padsize, padsize, padsize, padsize)
        nlmean(img, img_dest, halfKernelSz, halfSearchSz, h)
        print(f"nlmean imge process %s second" % (time.time() - start))

        cv.imwrite("./house_nlm.jpg", img_dest.to_numpy())
    elif run_type == RUN_TYPE.COMPILE:
        mod = ti.aot.Module(ti.vulkan, caps=[ti.DeviceCapability.spirv_has_int8])
        mod.add_kernel(copyBorder)
        mod.add_kernel(nlmean)
        mod.archive("taichi_nlm.tcm")

