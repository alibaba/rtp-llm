workspace(name = "rtp_llm")

# C++ 构建配置
load("//3rdparty/cuda_config:cuda_configure.bzl", "cuda_configure")
load("//3rdparty/gpus:rocm_configure.bzl", "rocm_configure")
load("//3rdparty/py:python_configure.bzl", "python_configure")

cuda_configure(name = "local_config_cuda")

rocm_configure(name = "local_config_rocm")

python_configure(name = "local_config_python")

# HTTP 依赖
load("//deps:http.bzl", "http_deps")

http_deps()

# Git 依赖
load("//deps:git.bzl", "git_deps")

git_deps()

# 版本信息
load("//:def.bzl", "read_release_version", "torch_local_repository")
read_release_version(name = "release_version")

torch_local_repository(
    name = "torch",
    build_file = "//:BUILD.pytorch",
)