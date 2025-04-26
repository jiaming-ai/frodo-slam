import os
import torch
from torch.utils.cpp_extension import load

ROOT = os.path.dirname(__file__)
src_dir = os.path.join(ROOT, "src")
include_dirs = [os.path.join(ROOT, "include"), os.path.join(ROOT, "../../thirdparty/eigen")]

_backends = load(
    name="mast3r_slam_backends",
    sources=[
        os.path.join(src_dir, "gn.cpp"),
        os.path.join(src_dir, "gn_kernels.cu"),
        os.path.join(src_dir, "matching_kernels.cu"),
    ],
    extra_include_paths=include_dirs,
    extra_cflags=["-O3"],
    extra_cuda_cflags=[ "-O3"],
    with_cuda=torch.cuda.is_available(),
    verbose=False,
)

# expose it
# gauss_newton_points   = _backends.gauss_newton_points
# gauss_newton_rays     = _backends.gauss_newton_rays
# gauss_newton_calib    = _backends.gauss_newton_calib
# iter_proj             = _backends.iter_proj
# refine_matches        = _backends.refine_matches
mast3r_slam_backends = _backends