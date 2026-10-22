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

# @rtp_deps — OSS default resolves to github-opensource/deps/ (public URLs).
# Internal monorepo overrides to ../internal_source/deps via
#     common --override_repository=rtp_deps=../internal_source/deps
# in internal_source/.internal_bazelrc. The two deps trees declare the same
# http_deps() / git_deps() symbol surface; the override swaps in internal-
# mirror URLs plus internal-only accelerator and transport dependencies.
# Keep the relative-path form: %workspace%/.. in try-imported rcs mis-resolves
# on Bazel 6.x for --override_repository.
local_repository(
    name = "rtp_deps",
    path = "deps",
)

load("@rtp_deps//:git.bzl", "rules_python_deps")

rules_python_deps()

load("@rules_pkg//:deps.bzl", "rules_pkg_dependencies")

rules_pkg_dependencies()

http_archive(
    name = "io_opentelemetry_cpp",
    # v1.21.0 tag commit: b9cf499ff5715433848b316059714b5c59af1f2c
    sha256 = "d020f3aa595a9e0cb8db468c07383e8771744cfe8d0257af4aa721f82c5b4220",
    strip_prefix = "opentelemetry-cpp-b9cf499ff5715433848b316059714b5c59af1f2c",
    patch_args = ["-p1"],
    # See the patch header for the trace-only compatibility contract and the
    # conditions under which the upstream metrics inputs must be restored.
    patches = ["//patches/opentelemetry_cpp:0001-trace-only-otlp-recordable.patch"],
    repo_mapping = {
        "@zlib": "@zlib_archive",
    },
    urls = [
        "https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/third_party/opentelemetry-cpp/opentelemetry-cpp-b9cf499ff5715433848b316059714b5c59af1f2c.tar.gz",
        "https://github.com/open-telemetry/opentelemetry-cpp/archive/b9cf499ff5715433848b316059714b5c59af1f2c.tar.gz",
    ],
)

load("//3rdparty/cuda_config:cuda_configure.bzl", "cuda_configure")
load("//3rdparty/gpus:rocm_configure.bzl", "rocm_configure")
load("//3rdparty/py:python_configure.bzl", "python_configure")

cuda_configure(name = "local_config_cuda")

rocm_configure(name = "local_config_rocm")

python_configure(name = "local_config_python")

local_repository(
    name = "arch_config",
    path = "arch_config",
)

load("@rtp_deps//:http.bzl", "http_deps")

http_deps()

load("@rtp_deps//:git.bzl", "git_deps")

git_deps()

load("//3rdparty/xgrammar:repositories.bzl", "xgrammar_deps")

xgrammar_deps()

load("@io_opentelemetry_cpp//bazel:repository.bzl", "opentelemetry_cpp_deps")

opentelemetry_cpp_deps()

load("@rules_python//python:repositories.bzl", "py_repositories")

py_repositories()

load("@rtp_deps//:pip.bzl", "pip_deps")

pip_deps()

load("@rules_python//python:pip.bzl", "pip_parse")

# Test-only, platform-independent pytest closure; do not alter runtime locks.
pip_parse(
    name = "pip_dsv4_test",
    requirements_lock = "//rtp_llm/test/pytest:requirements_lock.txt",
    python_interpreter = "/opt/conda310/bin/python3",
    extra_pip_args = ["--extra-index-url=https://mirrors.aliyun.com/pypi/simple/"],
    timeout = 3600,
)

load("@pip_dsv4_test//:requirements.bzl", pip_dsv4_test_install_deps = "install_deps")
pip_dsv4_test_install_deps()


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

load("@pip_gpu_cuda13_torch//:requirements.bzl", pip_gpu_cuda13_torch_install_deps = "install_deps")
pip_gpu_cuda13_torch_install_deps()

load("@pip_cuda12_arm_torch//:requirements.bzl", pip_cuda12_arm_torch_install_deps = "install_deps")
pip_cuda12_arm_torch_install_deps()

load("@pip_cuda13_arm_torch//:requirements.bzl", pip_cuda13_arm_torch_install_deps = "install_deps")
pip_cuda13_arm_torch_install_deps()

load("@pip_gpu_rocm_torch//:requirements.bzl", pip_gpu_rocm_torch_install_deps = "install_deps")
pip_gpu_rocm_torch_install_deps()

load("//:def.bzl", "read_release_version", "torch_local_repository")
read_release_version(name = "release_version")

torch_local_repository(
    name = "torch",
    build_file = "//:BUILD.pytorch.native",
)
