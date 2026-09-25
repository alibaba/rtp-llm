workspace(name = "rtp_llm")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

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
load("//3rdparty/gpus:dcu_configure.bzl", "dcu_configure")
load("//3rdparty/py:python_configure.bzl", "python_configure")

cuda_configure(name = "local_config_cuda")

rocm_configure(name = "local_config_rocm")

dcu_configure(name = "local_config_dcu")

python_configure(name = "local_config_python")

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

load("//3rdparty/xgrammar:repositories.bzl", "xgrammar_deps")

xgrammar_deps()

# node169 mirror: pre-declare platforms so opentelemetry maybe() does not hit github
http_archive(
    name = "platforms",
    sha256 = "218efe8ee736d26a3572663b374a253c012b716d8af0c07e842e82f238a0a7ee",
    urls = ["file:///home/bazel_mirrors/platforms-0.0.10.tar.gz"],
)

# node169 mirror: opentelemetry-proto 1.6.0
http_archive(
    name = "com_github_opentelemetry_proto",
    build_file = "@io_opentelemetry_cpp//bazel:opentelemetry_proto.BUILD",
    sha256 = "92682778affe8d00cd36f68308b49295db34fce379bef0a781c50837eccbc3c0",
    strip_prefix = "opentelemetry-proto-1.6.0",
    urls = ["file:///home/bazel_mirrors/opentelemetry-proto-v1.6.0.tar.gz"],
)

# node169 mirrors for opentelemetry_cpp_deps (github unreachable)

http_archive(
    name = "com_github_grpc_grpc",
    sha256 = "f40bde4ce2f31760f65dc49a2f50876f59077026494e67dccf23992548b1b04f",
    strip_prefix = "grpc-1.62.0",
    urls = ["file:///home/bazel_mirrors/grpc_grpc-f40bde4c.tar.gz"],
)

http_archive(
    name = "github_nlohmann_json",
    build_file = "@io_opentelemetry_cpp//bazel:nlohmann_json.BUILD",
    sha256 = "b8cb0ef2dd7f57f18933997c9934bb1fa962594f701cd5a8d3c2c80541559372",
    urls = ["file:///home/bazel_mirrors/github_nlohmann_json-b8cb0ef2.zip"],
)

http_archive(
    name = "com_github_jupp0r_prometheus_cpp",
    sha256 = "ac6e958405a29fbbea9db70b00fa3c420e16ad32e1baf941ab233ba031dd72ee",
    strip_prefix = "prometheus-cpp-1.3.0",
    urls = ["file:///home/bazel_mirrors/jupp0r_prometheus_cpp-ac6e9584.tar.gz"],
)

http_archive(
    name = "com_github_opentracing",
    sha256 = "5b170042da4d1c4c231df6594da120875429d5231e9baa5179822ee8d1054ac3",
    strip_prefix = "opentracing-cpp-1.6.0",
    urls = ["file:///home/bazel_mirrors/opentracing-5b170042.tar.gz"],
)

http_archive(
    name = "com_github_google_benchmark",
    sha256 = "6bc180a57d23d4d9515519f92b0c83d61b05b5bab188961f36ac7b06b0d9e9ce",
    strip_prefix = "benchmark-1.8.3",
    urls = ["file:///home/bazel_mirrors/google_benchmark-6bc180a5.tar.gz"],
)

http_archive(
    name = "build_bazel_apple_support",
    sha256 = "c4bb2b7367c484382300aee75be598b92f847896fb31bbd22f3a2346adf66a80",
    urls = ["file:///home/bazel_mirrors/apple_support-c4bb2b73.tar.gz"],
)

http_archive(
    name = "build_bazel_rules_apple",
    sha256 = "b4df908ec14868369021182ab191dbd1f40830c9b300650d5dc389e0b9266c8d",
    urls = ["file:///home/bazel_mirrors/rules_apple-b4df908e.tar.gz"],
)

http_archive(
    name = "rules_foreign_cc",
    sha256 = "69023642d5781c68911beda769f91fcbc8ca48711db935a75da7f6536b65047f",
    strip_prefix = "rules_foreign_cc-0.6.0",
    urls = ["file:///home/bazel_mirrors/rules_foreign_cc-69023642.tar.gz"],
)

http_archive(
    name = "rules_proto",
    sha256 = "8e7d59a5b12b233be5652e3d29f42fba01c7cbab09f6b3a8d0a57ed6d1e9a0da",
    strip_prefix = "rules_proto-7e4afce6fe62dbff0a4a03450143146f9f2d7488",
    urls = ["file:///home/bazel_mirrors/rules_proto-7e4afce6.tar.gz"],
)

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

load("@pip_gpu_dcu_torch//:requirements.bzl", pip_gpu_dcu_torch_install_deps = "install_deps")
pip_gpu_dcu_torch_install_deps()

load("//:def.bzl", "read_release_version")
read_release_version(name = "release_version")
