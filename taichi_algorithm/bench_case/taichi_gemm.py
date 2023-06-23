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

@ti.func
def gemm():
    pass


if __name__ == "__main__":
    if run_type == RUN_TYPE.RUN_DIRECT: 
        gemm()
    elif run_type == RUN_TYPE.COMPILE:
        pass