load("@rules_python//python:packaging.bzl", "py_package")
load("@rules_python//python:pip.bzl", "compile_pip_requirements")
load(":pip.bzl", "PIP_EXTRA_ARGS")

package(default_visibility = ["//visibility:public"])

py_package(
    name = "extension_package",
    packages = [
        "internal_source",
        "rtp_llm",
    ],
    deps = [
        ":extension_lib",
    ],
)

py_package(
    name = "extension_package_frontend",
    packages = [
        "internal_source",
        "rtp_llm",
    ],
    deps = [
        ":extension_frontend_lib",
    ]
)

py_library(
    name = "extension_frontend_lib",
    deps = [
        "//internal_source/rtp_llm:internal_tokenizers",
        "//internal_source/rtp_llm:internal_renderers",
        "//internal_source/rtp_llm:internal_utils",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "extension_lib",
    deps = [
        ":extension_frontend_lib",
        "//internal_source/rtp_llm:internal_models",
    ],
    visibility = ["//visibility:public"],
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
    name = "requirements_cpu_arm",
    src = "requirements_cpu_arm.txt",
    extra_args = PIP_EXTRA_ARGS,
    extra_data = ["//open_source/deps:requirements_base.txt"],
    requirements_txt = "requirements_lock_torch_arm.txt",
    tags = ["manual"],
)

compile_pip_requirements(
    name = "requirements_torch_ppu",
    src = "//internal_source/RTP_LLM-PPU/rtp_llm:requirements_torch_ppu.txt",
    extra_args = PIP_EXTRA_ARGS,
    extra_data = ["//open_source/deps:requirements_base.txt"],
    requirements_txt = "//internal_source/RTP_LLM-PPU/rtp_llm:requirements_lock_torch_ppu.txt",
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
    name = "requirements_torch_gpu_cuda12_9",
    src = "requirements_torch_gpu_cuda12_9.txt",
    extra_args = PIP_EXTRA_ARGS,
    extra_data = ["//open_source/deps:requirements_base.txt"],
    requirements_txt = "requirements_lock_torch_gpu_cuda12_9.txt",
    tags = ["manual"],
)

compile_pip_requirements(
    name = "requirements_torch_gpu_cuda13",
    src = "requirements_torch_gpu_cuda13.txt",
    extra_args = PIP_EXTRA_ARGS,
    extra_data = ["//open_source/deps:requirements_base.txt"],
    requirements_txt = "requirements_lock_torch_gpu_cuda13.txt",
    tags = ["manual"],
)

compile_pip_requirements(
    name = "requirements_cuda12_arm",
    src = "requirements_cuda12_arm.txt",
    extra_args = PIP_EXTRA_ARGS,
    extra_data = ["//open_source/deps:requirements_base.txt"],
    requirements_txt = "requirements_lock_cuda12_arm.txt",
    tags = ["manual"],
)

compile_pip_requirements(
    name = "requirements_cuda13_arm",
    src = "requirements_cuda13_arm.txt",
    extra_args = PIP_EXTRA_ARGS,
    extra_data = ["//open_source/deps:requirements_base.txt"],
    requirements_txt = "requirements_lock_cuda13_arm.txt",
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
