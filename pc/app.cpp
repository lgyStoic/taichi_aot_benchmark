#include <fstream>
#include "gemm.h"

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
  std::shared_ptr<taichi_aot::AOT_APP> app = std::make_shared<taichi_aot::GEMM_APP>();
  std::string gemm_module = "./build/assets/bench_case/taichi_gemm.tcm";
  app->create_name(gemm_module, TI_ARCH_METAL);
  app->prepare();
  app->run();
  app->output();
  return 0;
}

