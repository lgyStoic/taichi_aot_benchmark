import cv2
import taichi as ti
import taichi.math as tm
import math
import numpy as np

ti.init(arch=ti.vulkan)

img2d = ti.types.ndarray(dtype=ti.math.vec3, ndim=2)

def compute_weights(radius, sigma):
    weights = np.zeros((2 * radius + 1), dtype=np.float32)
    total = 0.0
    # Not much computation here - serialize the for loop to save two more GPU kernel launch costs
    for i in range(0, 2 * radius + 1):
        # Drop the normal distribution constant coefficients since we need to normalize later anyway
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
            total_rgb += img[l, j] * w
            total_weight += w

        img_blurred[i, j] = (total_rgb / total_weight).cast(ti.u8)

    for i, j in ti.ndrange(n, m):
        l_begin, l_end = max(0, j - blur_radius), min(m, j + blur_radius + 1)
        total_rgb = tm.vec3(0.0)
        total_weight = 0.0
        for l in range(l_begin, l_end):
            w = weights[j - l + blur_radius]
            total_rgb += img_blurred[i, l] * w
            total_weight += w

        img[i, j] = (total_rgb / total_weight).cast(ti.u8)

if __name__ == "__main__":
    img = cv2.imread('./mountain.jpg')
    cv2.imshow('input', img)
    sigma = 10
    radius = math.ceil(sigma * 3)
    win_sz = 2 * radius + 1
    weights = compute_weights(radius, sigma) 
    weightsNdarray = ti.ndarray(dtype=ti.f32, shape=(win_sz))
    weightsNdarray.from_numpy(weights)
    img_blurred = ti.ndarray(dtype=ti.math.vec3, shape=(img.shape[0], img.shape[1]))

    gaussian_blur(img, weightsNdarray, img_blurred)

    cv2.imwrite("./mountain_blur.jpg", img)
