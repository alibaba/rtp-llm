workspace(name = "maga_transformer")

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

load("//deps:pip.bzl", "pip_deps")
pip_deps()
