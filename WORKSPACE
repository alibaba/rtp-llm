workspace(name = "rtp_llm")

load("//3rdparty/cuda_config:cuda_configure.bzl", "cuda_configure")
load("//3rdparty/gpus:rocm_configure.bzl", "rocm_configure")
load("//3rdparty/py:python_configure.bzl", "python_configure")

cuda_configure(name = "local_config_cuda")

rocm_configure(name = "local_config_rocm")

python_configure(name = "local_config_python")

load("//deps:http.bzl", "http_deps")

http_deps()

load("//deps:git.bzl", "git_deps")

git_deps()

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

maybe(
    new_git_repository,
    name = "xgrammar",
    remote = "https://github.com/mlc-ai/xgrammar.git",
    commit = "36998a7abfb6a8fb79057aef110a6e93d0fd634c",  # v0.2.2
    init_submodules = False,
    patch_cmds = [
        "git submodule update --init --depth=1 3rdparty/dlpack 3rdparty/picojson",
    ],
    build_file = "//3rdparty/xgrammar:xgrammar.BUILD",
)

load("@rules_python//python:repositories.bzl", "py_repositories")

py_repositories()

load("//deps:pip.bzl", "pip_deps")

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

load("//:def.bzl", "read_release_version")
read_release_version(name = "release_version")
