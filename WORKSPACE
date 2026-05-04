workspace(name = "rtp_llm")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Required by @havenask//.../bundle.bzl (providers.bzl).
http_archive(
    name = "rules_pkg",
    sha256 = "d250924a2ecc5176808fc4c25d5cf5e9e79e6346d79d5ab1c493e289e722d1d0",
    # Prefer mirror.bazel.build first: UrlRewriter rewrites github.com to internal OSS mirrors;
    # a bad mirror response can yield an incomplete rules_pkg (missing providers.bzl) and break
    # @havenask//.../bundle.bzl analysis. Same tarball as GitHub (sha256 verified).
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_pkg/releases/download/0.10.1/rules_pkg-0.10.1.tar.gz",
        "https://github.com/bazelbuild/rules_pkg/releases/download/0.10.1/rules_pkg-0.10.1.tar.gz",
    ],
    # rules_pkg 0.9+ moved providers under //pkg:; @havenask still loads @rules_pkg//:providers.bzl.
    patches = ["//patches/rules_pkg:0001-add-providers-root-shim-for-havenask.patch"],
    patch_args = ["-p1"],
)

load("@rules_pkg//:deps.bzl", "rules_pkg_dependencies")

rules_pkg_dependencies()

load("//3rdparty/cuda_config:cuda_configure.bzl", "cuda_configure")
load("//3rdparty/gpus:rocm_configure.bzl", "rocm_configure")
load("//3rdparty/py:python_configure.bzl", "python_configure")

cuda_configure(name = "local_config_cuda")

rocm_configure(name = "local_config_rocm")

python_configure(name = "local_config_python")

# @rtp_deps — OSS default resolves to github-opensource/deps/ (public URLs).
# Internal monorepo overrides to ../internal_source/deps via
#     common --override_repository=rtp_deps=../internal_source/deps
# in internal_source/.internal_bazelrc. The two deps trees declare the same
# http_deps() / git_deps() symbol surface; the override swaps in internal-
# mirror URLs plus internal-only RDMA/EIC RPMs and PPU cutlass/flashinfer.
# Keep the relative-path form: %workspace%/.. in try-imported rcs mis-resolves
# on Bazel 6.x for --override_repository.
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

load("//:def.bzl", "read_release_version", "torch_local_repository")
read_release_version(name = "release_version")

torch_local_repository(
    name = "torch",
    build_file = "//:BUILD.pytorch",
)
