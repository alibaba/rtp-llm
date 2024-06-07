
load("//3rdparty/cuda_config:cuda_configure.bzl", "cuda_configure")
load("//3rdparty/gpus:rocm_configure.bzl", "rocm_configure")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")
load("//3rdparty/py:python_configure.bzl", "python_configure")

def clean_dep(dep):
    return str(Label(dep))

def workspace():
    cuda_configure(name = "local_config_cuda")
    rocm_configure(name = "local_config_rocm")
    python_configure(name = "local_config_python")
    http_dependency()
    git_dependency()
