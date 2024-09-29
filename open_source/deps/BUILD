package(default_visibility = ["//visibility:public"])
load("@rules_python//python:packaging.bzl", "py_package")
load("@rules_python//python:pip.bzl", "compile_pip_requirements")
load(":pip.bzl", "PIP_EXTRA_ARGS")
exports_files(["requirements_base.txt"])

py_package(
    name = "extension_package",
    deps = [
    ],
    packages = [
        "maga_transformer"
    ],
)

py_library(
    name = "extension_lib",
    deps = [],
)

compile_pip_requirements(
    name = "requirements_torch_cpu",
    src = "requirements_torch_cpu.txt",
    extra_args = PIP_EXTRA_ARGS,
    extra_data = ["//open_source/deps:requirements_base.txt"],
    requirements_txt = "requirements_lock_torch_cpu.txt",
    tags = ["manual"],
)

compile_pip_requirements(
    name = "requirements_torch_gpu",
    src = "requirements_torch_gpu.txt",
    extra_args = PIP_EXTRA_ARGS,
    extra_data = ["//open_source/deps:requirements_base.txt"],
    requirements_txt = "requirements_lock_torch_gpu.txt",
    tags = ["manual"],
)

compile_pip_requirements(
    name = "requirements_torch_gpu_cuda12",
    src = "requirements_torch_gpu_cuda12.txt",
    extra_args = PIP_EXTRA_ARGS,
    extra_data = ["//open_source/deps:requirements_base.txt"],
    requirements_txt = "requirements_lock_torch_gpu_cuda12.txt",
    tags = ["manual"],
)

compile_pip_requirements(
    name = "requirements_rocm",
    src = "requirements_rocm.txt",
    extra_args = PIP_EXTRA_ARGS,
    extra_data = ["//open_source/deps:requirements_base.txt"],
    requirements_txt = "requirements_lock_rocm.txt",
    tags = ["manual"],
)

compile_pip_requirements(
    name = "requirements_cpu_arm",
    src = "requirements_cpu_arm.txt",
    extra_args = PIP_EXTRA_ARGS,
    extra_data = ["//open_source/deps:requirements_base.txt"],
    requirements_txt = "requirements_lock_torch_arm.txt",
    tags = ["manual"],
)
