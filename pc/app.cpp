#include <fstream>
#include <taichi/cpp/taichi.hpp>

void save_ppm(const float* pixels, uint32_t w, uint32_t h, const char* path) {
  std::fstream f(path, std::ios::out | std::ios::trunc);
  f << "P3\n" << w << ' ' << h << "\n255\n";
  for (int j = h - 1; j >= 0; --j) {
    for (int i = 0; i < w; ++i) {
      f << static_cast<uint32_t>(255.999 * pixels[i * h + j]) << ' '
        << static_cast<uint32_t>(255.999 * pixels[i * h + j]) << ' '
        << static_cast<uint32_t>(255.999 * pixels[i * h + j]) << '\n';
    }
  }
  f.flush();
  f.close();
}

int main(int argc, const char** argv) {
  ti::Runtime runtime(TI_ARCH_VULKAN);
  ti::AotModule aot_module = runtime.load_aot_module("module.tcm");
  ti::Kernel kernel_paint = aot_module.get_kernel("paint");

  int n = 320;
  float t = 0.0f;
  ti::NdArray<float> pixels = runtime.allocate_ndarray<float>({(uint32_t)(2 * n), (uint32_t)n}, {1}, true);

  kernel_paint[0] = n;
  kernel_paint[1] = t;
  kernel_paint[2] = pixels;
  kernel_paint.launch();
  runtime.wait();

  auto pixels_data = (const float*)pixels.map();
  save_ppm(pixels_data, 2 * n, n, "result.ppm");
  pixels.unmap();

  return 0;
}

