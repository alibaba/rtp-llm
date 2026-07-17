workspace(name = "rtp_llm")

load("//3rdparty/cuda_config:cuda_configure.bzl", "cuda_configure")
load("//3rdparty/gpus:rocm_configure.bzl", "rocm_configure")
load("//3rdparty/py:python_configure.bzl", "python_configure")
load("//3rdparty/gpus:xpu_configure.bzl", "xpu_configure")
load("//3rdparty/gpus:torch_xpu_configure.bzl", "torch_xpu_configure")

cuda_configure(name = "local_config_cuda")

rocm_configure(name = "local_config_rocm")

python_configure(name = "local_config_python")

xpu_configure(name = "local_config_xpu")

torch_xpu_configure(
    name = "torch_xpu",
    build_file = "//:BUILD.pytorch",
)

local_repository(
    name = "rtp_deps",
    path = "deps",
)

local_repository(
    name = "arch_config",
    path = "arch_config",
)

load("@rtp_deps//:http.bzl", "http_deps")

http_deps()

load("@rtp_deps//:git.bzl", "git_deps")

git_deps()

load("@rules_python//python:repositories.bzl", "py_repositories")

py_repositories()

load("@rtp_deps//:pip.bzl", "pip_deps")

pip_deps()

load("@pip_cpu_torch//:requirements.bzl", pip_cpu_torch_install_deps = "install_deps")
pip_cpu_torch_install_deps()

load("@pip_arm_torch//:requirements.bzl", pip_arm_torch_install_deps = "install_deps")
pip_arm_torch_install_deps()

load("@pip_ppu_torch//:requirements.bzl", pip_ppu_torch_install_deps = "install_deps")
pip_ppu_torch_install_deps()

load("@pip_gpu_cuda12_torch//:requirements.bzl", pip_gpu_cuda12_torch_install_deps = "install_deps")
pip_gpu_cuda12_torch_install_deps()

load("@pip_gpu_cuda12_9_torch//:requirements.bzl", pip_gpu_cuda12_9_torch_install_deps = "install_deps")
pip_gpu_cuda12_9_torch_install_deps()

load("@pip_cuda12_arm_torch//:requirements.bzl", pip_cuda12_arm_torch_install_deps = "install_deps")
pip_cuda12_arm_torch_install_deps()

load("@pip_gpu_rocm_torch//:requirements.bzl", pip_gpu_rocm_torch_install_deps = "install_deps")
pip_gpu_rocm_torch_install_deps()

# XPU wheel installation is gated on TF_NEED_XPU (set by --config=xpu) via the
# xpu_pip_gate repo declared in pip_deps(). On non-XPU containers and plain
# `bazel sync`, install_deps() is a no-op, so the pip_xpu_torch_* whl_library
# repos are never declared and XPU-only pins (e.g. scikit-learn==1.8.0, which
# Requires-Python>=3.11) are never resolved under Python 3.10.
load("@xpu_pip_gate//:requirements.bzl", pip_xpu_torch_install_deps = "install_deps")
pip_xpu_torch_install_deps()

load("//:def.bzl", "read_release_version")
read_release_version(name = "release_version")
