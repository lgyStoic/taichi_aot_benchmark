import cv2
import taichi as ti
import taichi.math as tm
import math
import numpy as np
import time
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

ti.init(arch=ti.vulkan, debug=True)

img2d = ti.types.ndarray(dtype=ti.types.vector(3, ti.u8), ndim=2)

def compute_weights(radius, sigma):
    weights = np.zeros((2 * radius + 1), dtype=np.float32)
    total = 0.0
    for i in range(0, 2 * radius + 1):
        val = ti.exp(-0.5 * ((i - radius) / sigma)**2)
        weights[i] = val
        total += val

    for i in range(0, 2 * radius + 1):
        weights[i] /= total
    return weights


@ti.kernel
def gaussian_blur(img: img2d, weights: ti.types.ndarray(dtype=ti.f32, ndim=1), img_blurred: img2d):
    n, m = img.shape[0], img.shape[1]
    blur_radius = int(weights.shape[0] / 2)

    for i, j in ti.ndrange(n, m):
        l_begin, l_end = max(0, i - blur_radius), min(n, i + blur_radius + 1)
        total_rgb = tm.vec3(0.0)
        total_weight = 0.0
        for l in range(l_begin, l_end):
            w = weights[i - l + blur_radius]
            total_rgb += ti.cast(img[l, j], ti.f32) * w
            total_weight += w

        img_blurred[i, j] = (total_rgb / total_weight).cast(ti.u8)

    for i, j in ti.ndrange(n, m):
        l_begin, l_end = max(0, j - blur_radius), min(m, j + blur_radius + 1)
        total_rgb = tm.vec3(0.0)
        total_weight = 0.0
        for l in range(l_begin, l_end):
            w = weights[j - l + blur_radius]
            total_rgb += ti.cast(img_blurred[i, l], ti.f32) * w
            total_weight += w

        img[i, j] = (total_rgb / total_weight).cast(ti.u8)

if __name__ == "__main__":
    if run_type == RUN_TYPE.RUN_DIRECT:
        img = cv2.imread('./mountain.jpg')
        sigma = 20.0
        radius = math.ceil(sigma * 3)
        win_sz = 2 * radius + 1
        weights = compute_weights(radius, sigma) 
        weightsNdarray = ti.ndarray(dtype=ti.f32, shape=(win_sz))
        weightsNdarray.from_numpy(weights)
        img_blurred = ti.ndarray(dtype=ti.types.vector(3, ti.u8), shape=(img.shape[0], img.shape[1]))

        start = time.time()
        gaussian_blur(img, weightsNdarray, img_blurred)
    
        print(f"imge process %s second" % (time.time() - start))

        cv2.imwrite("./mountain_blur.jpg", img)

        img = cv2.imread('./mountain.jpg')
        start = time.time()
        img = cv2.GaussianBlur(img, (win_sz, win_sz), sigmaX=sigma, sigmaY=sigma)
        print(f"opencv imge process %s second" % (time.time() - start))
        cv2.imwrite("./mountain_blur_opencv.jpg", img)
    else:
        mod = ti.aot.Module(ti.vulkan, caps=[ti.DeviceCapability.spirv_has_int8])
        mod.add_kernel(gaussian_blur)
        mod.archive("taichi_gaussian.tcm")