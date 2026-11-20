package(default_visibility = ["//visibility:public"])
load("@rules_python//python:packaging.bzl", "py_package")
load("@rules_python//python:pip.bzl", "compile_pip_requirements")
load(":pip.bzl", "PIP_EXTRA_ARGS")
exports_files([
    "http.bzl",
    "pip.bzl",
    "requirements_base.txt",
    "requirements_lock_rocm.txt",
    "requirements_rocm.txt",
])

py_package(
    name = "extension_package",
    deps = [
    ],
    packages = [
        "rtp_llm"
    ],
)


py_package(
    name = "extension_package_frontend",
    deps = [
    ],
    packages = [
        "rtp_llm"
    ],
)

py_library(
    name = "extension_frontend_lib",
    deps = [],
    visibility = ["//visibility:public"],
)

py_library(
    name = "extension_lib",
    deps = [],
    visibility = ["//visibility:public"],
)

compile_pip_requirements(
    name = "requirements_torch_cpu",
    src = "requirements_torch_cpu.txt",
    extra_args = PIP_EXTRA_ARGS,
    extra_data = ["//:requirements_base.txt"],
    requirements_txt = "requirements_lock_torch_cpu.txt",
    tags = ["manual"],
)

compile_pip_requirements(
    name = "requirements_torch_gpu_cuda12",
    src = "requirements_torch_gpu_cuda12.txt",
    extra_args = PIP_EXTRA_ARGS,
    extra_data = ["//:requirements_base.txt"],
    requirements_txt = "requirements_lock_torch_gpu_cuda12.txt",
    tags = ["manual"],
)

compile_pip_requirements(
    name = "requirements_torch_gpu_cuda12_9",
    src = "requirements_torch_gpu_cuda12_9.txt",
    extra_args = PIP_EXTRA_ARGS,
    extra_data = ["//:requirements_base.txt"],
    requirements_txt = "requirements_lock_torch_gpu_cuda12_9.txt",
    tags = ["manual"],
)

sh_binary(
    name = "requirements_torch_gpu_cuda13.update",
    srcs = ["update_cuda13_locks.sh"],
    args = ["x86"],
    data = [
        "requirements_base.txt",
        "requirements_cuda13_overrides.txt",
        "requirements_torch_gpu_cuda13.txt",
    ],
    tags = ["manual"],
)

sh_binary(
    name = "requirements_torch_gpu_cuda13_sm90.update",
    srcs = ["update_cuda13_locks.sh"],
    args = ["sm90-x86"],
    data = [
        "requirements_base.txt",
        "requirements_cuda13_sm90_overrides.txt",
        "requirements_torch_gpu_cuda13_sm90.txt",
    ],
    tags = ["manual"],
)

compile_pip_requirements(
    name = "requirements_cuda12_arm",
    src = "requirements_cuda12_arm.txt",
    extra_args = PIP_EXTRA_ARGS,
    extra_data = ["//:requirements_base.txt"],
    requirements_txt = "requirements_lock_cuda12_arm.txt",
    tags = ["manual"],
)

sh_binary(
    name = "requirements_cuda13_arm.update",
    srcs = ["update_cuda13_locks.sh"],
    args = ["arm"],
    data = [
        "requirements_base.txt",
        "requirements_cuda13_arm.txt",
        "requirements_cuda13_overrides.txt",
    ],
    tags = ["manual"],
)

sh_test(
    name = "update_cuda13_locks_test",
    srcs = ["update_cuda13_locks_test.sh"],
    data = ["update_cuda13_locks.sh"],
)

compile_pip_requirements(
    name = "requirements_rocm",
    src = "requirements_rocm.txt",
    extra_args = PIP_EXTRA_ARGS,
    extra_data = ["//:requirements_base.txt"],
    requirements_txt = "requirements_lock_rocm.txt",
    tags = ["manual"],
)

# Standalone CPU validation of both dependency-repository annotations.
py_test(
    name = "cutlass_sm90_annotations_test",
    srcs = ["cutlass_sm90_annotations_test.py"],
    data = [
        ":pip.bzl",
        "@rtp_deps//:pip.bzl",
        "@rtp_llm//patches/cutlass:0001-bazel-split-package-paths.patch",
        "@rtp_llm//patches/cutlass:0002-bazel-toolkit-library-paths.patch",
        "@rtp_llm//patches/cutlass:0003-sm90-481-split-package-paths.patch",
        "@rtp_llm//patches/cutlass:0004-sm90-481-toolkit-library-paths.patch",
    ],
    env = {"GPU_COUNT": "0", "WORLD_SIZE": "1"},
)
