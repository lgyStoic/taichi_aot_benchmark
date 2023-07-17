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
            pixel = img_pad[i, j].cast(ti.f32)
            newPixel = ti.math.vec3(0.0, 0.0, 0.0)
            sumw = ti.math.vec3(0.0, 0.0, 0.0)

            for sr in range(-halfSearchSz, halfSearchSz):
                for sc in range(-halfSearchSz, halfSearchSz):
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
           
@ti.kernel
def copyBorder():
    pass
        
if __name__ == "__main__":
    if run_type == RUN_TYPE.RUN_DIRECT:
        img = cv.imread('./house.jpg')
        halfKernelSz = 5
        halfSearchSz = 15
        h = 20
        img = cv.copyMakeBorder(img, halfKernelSz + halfSearchSz, halfSearchSz + halfKernelSz, halfSearchSz + halfKernelSz, halfSearchSz + halfKernelSz, 0)
        img_dest = ti.ndarray(dtype=ti.types.vector(3, ti.u8), shape=(img.shape[0], img.shape[1]))
        start = time.time()
        nlmean(img, img_dest, halfKernelSz, halfSearchSz, h)
        print(f"nlmean imge process %s second" % (time.time() - start))

        cv.imwrite("./house_nlm.jpg", img_dest.to_numpy())

