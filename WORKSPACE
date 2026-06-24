workspace(name = "rtp_llm")

load("//3rdparty/cuda_config:cuda_configure.bzl", "cuda_configure")
load("//3rdparty/gpus:rocm_configure.bzl", "rocm_configure")
load("//3rdparty/py:python_configure.bzl", "python_configure")

cuda_configure(name = "local_config_cuda")

rocm_configure(name = "local_config_rocm")

python_configure(name = "local_config_python")

local_repository(
    name = "rtp_deps",
    path = "deps",
)

# Always points to the opensource `deps/` directory. `rtp_deps` gets swapped to
# `internal_source/deps/` by `--override_repository` in `.internal_bazelrc`, but
# the internal overlay still needs to read opensource requirements via `-r`
# includes and shared http_archive declarations — this repo is the stable
# handle for that.
local_repository(
    name = "rtp_opensource_deps",
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

load("@pip_ppu_torch//:requirements.bzl", pip_ppu_torch_install_deps = "install_deps")
pip_ppu_torch_install_deps()

load("@pip_gpu_cuda12_9_torch//:requirements.bzl", pip_gpu_cuda12_9_torch_install_deps = "install_deps")
pip_gpu_cuda12_9_torch_install_deps()

# TODO(pip_unify): Backwards-compatible install for internal source that still
# references @pip_gpu_cuda12_torch. Remove once internal overlays are migrated.
load("@pip_gpu_cuda12_torch//:requirements.bzl", pip_gpu_cuda12_torch_install_deps = "install_deps")
pip_gpu_cuda12_torch_install_deps()

load("@pip_cuda12_arm_torch//:requirements.bzl", pip_cuda12_arm_torch_install_deps = "install_deps")
pip_cuda12_arm_torch_install_deps()

load("@pip_gpu_rocm_torch//:requirements.bzl", pip_gpu_rocm_torch_install_deps = "install_deps")
pip_gpu_rocm_torch_install_deps()

load("//:def.bzl", "read_release_version")
read_release_version(name = "release_version")
