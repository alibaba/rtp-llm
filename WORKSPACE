workspace(name = "rtp_llm")

load("//3rdparty/cuda_config:cuda_configure.bzl", "cuda_configure")
load("//3rdparty/gpus:rocm_configure.bzl", "rocm_configure")
load("//3rdparty/py:python_configure.bzl", "python_configure")

cuda_configure(name = "local_config_cuda")

rocm_configure(name = "local_config_rocm")

python_configure(name = "local_config_python")

# @rtp_deps — Starlark deps (http.bzl / git.bzl). In the internal monorepo, sources live in
# internal_source/deps (sibling of github-opensource/). Use path="../internal_source/deps":
# relying on a github-opensource/deps symlink out of the workspace can make Bazel resolve the
# repo to github-opensource/internal_source/deps (missing) instead of following "..".
# OSS-only clones without internal_source should vendor deps under github-opensource/deps locally.
local_repository(
    name = "rtp_deps",
    path = "../internal_source/deps",
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

load("//:def.bzl", "read_release_version", "torch_local_repository")
read_release_version(name = "release_version")

torch_local_repository(
    name = "torch",
    build_file = "//:BUILD.pytorch",
)
