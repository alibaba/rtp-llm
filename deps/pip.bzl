load("@rules_python//python:pip.bzl", "package_annotation", "pip_parse")

PIP_EXTRA_ARGS = [
    "--cache-dir=~/.cache/pip",
    "--extra-index-url=https://mirrors.aliyun.com/pypi/simple/",
    "--verbose",
]

# FlashInfer 0.6.9 plans decode on the CPU; avoid Python scalar-Tensor iteration.
# Apply directly to the Python wheel used by both CUDA 13 architectures.
FLASHINFER_069_ANNOTATIONS = {
    "flashinfer-python": package_annotation(
        whl_patches = {
            "@rtp_llm//patches/flashinfer:0016-python-decode-max-kv-len.patch": json.encode({
                "whls": [
                    "flashinfer_python-0.6.9-py3-none-any.whl",
                    "flashinfer_python-0.6.9+8c4f4dcf-py3-none-any.whl",
                ],
                "patch_strip": 1,
            }),
        },
    ),
}

CUDA13_ANNOTATIONS = dict(FLASHINFER_069_ANNOTATIONS)
CUDA13_ANNOTATIONS["nvidia-cutlass-dsl-libs-core"] = package_annotation(
    whl_patches = {
        "@rtp_llm//patches/cutlass:0001-bazel-split-package-paths.patch": json.encode({
            "whls": ["nvidia_cutlass_dsl_libs_core-4.7.1-py3-none-any.whl"],
            "patch_strip": 1,
        }),
    },
)
CUDA13_ANNOTATIONS["nvidia-cutlass-dsl-libs-base"] = package_annotation(
    whl_patches = {
        "@rtp_llm//patches/cutlass:0002-bazel-toolkit-library-paths.patch": json.encode({
            "whls": [
                "nvidia_cutlass_dsl_libs_base-4.7.1-cp310-cp310-manylinux_2_28_x86_64.whl",
                "nvidia_cutlass_dsl_libs_base-4.7.1-cp310-cp310-manylinux_2_28_aarch64.whl",
            ],
            "patch_strip": 1,
        }),
    },
)
CUDA13_ANNOTATIONS["flash-attn-4"] = package_annotation(
    whl_patches = {
        "@rtp_llm//patches/flash_attn4:0001-quack-packed-sub.patch": json.encode({
            "whls": ["flash_attn_4-4.0.0b21-py3-none-any.whl"],
            "patch_strip": 1,
        }),
    },
)

def pip_deps():
    pip_parse(
        name = "pip_cpu_torch",
        requirements_lock = "@rtp_deps//:requirements_lock_torch_cpu.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS,
        timeout = 3600,
    )

    pip_parse(
        name = "pip_arm_torch",
        requirements_lock = "@rtp_deps//:requirements_lock_torch_arm.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS,
        timeout = 3600,
    )

    pip_parse(
        name = "pip_ppu_torch",
        requirements_lock = "@rtp_deps//:requirements_lock_torch_gpu_cuda12.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS,
        timeout = 3600,
    )

    pip_parse(
        name = "pip_gpu_cuda12_torch",
        requirements_lock = "@rtp_deps//:requirements_lock_torch_gpu_cuda12.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS,
        timeout = 3600,
        quiet = False,
    )

    pip_parse(
        name = "pip_gpu_cuda12_9_torch",
        annotations = FLASHINFER_069_ANNOTATIONS,
        requirements_lock = "@rtp_deps//:requirements_lock_torch_gpu_cuda12_9.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS,
        timeout = 3600,
        quiet = False,
    )

    pip_parse(
        name = "pip_gpu_cuda13_torch",
        annotations = CUDA13_ANNOTATIONS,
        requirements_lock = "@rtp_deps//:requirements_lock_torch_gpu_cuda13.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS + ["--quiet"],
        timeout = 3600,
        quiet = False,
    )

    pip_parse(
        name = "pip_cuda12_arm_torch",
        requirements_lock = "@rtp_deps//:requirements_lock_cuda12_arm.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS,
        timeout = 3600,
        quiet = False,
    )

    pip_parse(
        name = "pip_cuda13_arm_torch",
        annotations = CUDA13_ANNOTATIONS,
        requirements_lock = "@rtp_deps//:requirements_lock_cuda13_arm.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS,
        timeout = 3600,
        quiet = False,
    )

    pip_parse(
        name = "pip_gpu_rocm_torch",
        requirements_lock = "@rtp_deps//:requirements_lock_rocm.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS,
        timeout = 12000,
    )
