import taichi as ti

ti.init(arch=ti.vulkan)
if ti.lang.impl.current_cfg().arch != ti.vulkan:
    raise RuntimeError("Vulkan is not available.")

@ti.kernel
def paint(n: ti.u32, t: ti.f32, pixels: ti.types.ndarray(dtype=ti.f32, ndim=2)):
    for i, j in pixels:  # Parallelized over all pixels
        c = ti.Vector([-0.8, ti.cos(t) * 0.2])
        z = ti.Vector([i / n - 1, j / n - 0.5]) * 2
        iterations = 0
        while z.norm() < 20 and iterations < 50:
            z = ti.Vector([z[0]**2 - z[1]**2, z[1] * z[0] * 2]) + c
            iterations += 1
        pixels[i, j] = 1 - iterations * 0.02


mod = ti.aot.Module(ti.vulkan)
mod.add_kernel(paint)
mod.archive("module.tcm")
