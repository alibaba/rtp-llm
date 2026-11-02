load("@rules_python//python:pip.bzl", "package_annotation", "pip_parse")

# Index selection is caller-injected, never file content. Locks carry no
# --index-url lines, so these arguments fully own where by-name packages
# resolve from. rules_python runs pip with --isolated (which ignores PIP_*
# environment variables), so the envsubst channel is the only working
# mechanism: ${VAR:-default} is expanded at repository-rule time into the
# pip command line.
#
#   public default : pypi.org; closed checkouts override both via
#                    .internal_bazelrc (the artlab PyPI mirror + the private
#                    rtp_diffusion wheel index).
#
# Note: on Bazel 6/7, envsubst variable changes do not trigger refetches.
# That is benign for hash-pinned locks (changing the index cannot change the
# resolved bytes), but switching indexes mid-cache requires
# `bazel clean --expunge` or removing the pip repository cache.
PIP_EXTRA_ARGS = [
    "--cache-dir=~/.cache/pip",
    "--verbose",
]

PIP_INDEX_ARGS = [
    "--index-url=${PIP_INDEX_URL:-https://pypi.org/simple/}",
    "--extra-index-url=${PIP_EXTRA_INDEX_URL:-https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com/rtp_llm/simple/}",
]

PIP_INDEX_ENVSUBST = [
    "PIP_INDEX_URL",
    "PIP_EXTRA_INDEX_URL",
]

# FlashInfer 0.6.9 plans decode on the CPU; avoid Python scalar-Tensor iteration.
# Apply directly to the Python wheels used by the CUDA 12.9/13 lines (both
# the 8c4f4dcf and e87d610f fork builds plus upstream stock).
FLASHINFER_069_ANNOTATIONS = {
    "flashinfer-python": package_annotation(
        whl_patches = {
            "@rtp_llm//patches/flashinfer:0016-python-decode-max-kv-len.patch": json.encode({
                "whls": [
                    "flashinfer_python-0.6.9-py3-none-any.whl",
                    "flashinfer_python-0.6.9+e87d610f-py3-none-any.whl",
                    "flashinfer_python-0.6.9+8c4f4dcf-py3-none-any.whl",
                ],
                "patch_strip": 1,
            }),
        },
    ),
}

def pip_deps():
    pip_parse(
        name = "pip_cpu_torch",
        requirements_lock = "@rtp_pip_deps//:requirements_lock_torch_cpu.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS + PIP_INDEX_ARGS,
        envsubst = PIP_INDEX_ENVSUBST,
        timeout = 3600,
    )

    pip_parse(
        name = "pip_arm_torch",
        requirements_lock = "@rtp_pip_deps//:requirements_lock_torch_arm.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS + PIP_INDEX_ARGS,
        envsubst = PIP_INDEX_ENVSUBST,
        timeout = 3600,
    )

    pip_parse(
        name = "pip_ppu_torch",
        requirements_lock = "@rtp_pip_deps//:requirements_lock_torch_ppu.txt",
        python_interpreter = "/opt/conda312/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS + PIP_INDEX_ARGS,
        envsubst = PIP_INDEX_ENVSUBST,
        timeout = 3600,
    )

    pip_parse(
        name = "pip_gpu_cuda12_torch",
        requirements_lock = "@rtp_pip_deps//:requirements_lock_torch_gpu_cuda12.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS + PIP_INDEX_ARGS,
        envsubst = PIP_INDEX_ENVSUBST,
        timeout = 3600,
        quiet = False,
    )

    pip_parse(
        name = "pip_gpu_cuda12_9_torch",
        annotations = FLASHINFER_069_ANNOTATIONS,
        requirements_lock = "@rtp_pip_deps//:requirements_lock_torch_gpu_cuda12_9.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS + PIP_INDEX_ARGS + ["--quiet"],
        envsubst = PIP_INDEX_ENVSUBST,
        timeout = 3600,
        quiet = False,
    )

    pip_parse(
        name = "pip_gpu_cuda13_torch",
        annotations = FLASHINFER_069_ANNOTATIONS,
        requirements_lock = "@rtp_pip_deps//:requirements_lock_torch_gpu_cuda13.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS + PIP_INDEX_ARGS + ["--quiet"],
        envsubst = PIP_INDEX_ENVSUBST,
        timeout = 3600,
        quiet = False,
    )

    pip_parse(
        name = "pip_cuda12_arm_torch",
        annotations = FLASHINFER_069_ANNOTATIONS,
        requirements_lock = "@rtp_pip_deps//:requirements_lock_cuda12_arm.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS + PIP_INDEX_ARGS,
        envsubst = PIP_INDEX_ENVSUBST,
        timeout = 3600,
        quiet = False,
    )

    pip_parse(
        name = "pip_cuda13_arm_torch",
        annotations = FLASHINFER_069_ANNOTATIONS,
        requirements_lock = "@rtp_pip_deps//:requirements_lock_cuda13_arm.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS + PIP_INDEX_ARGS,
        envsubst = PIP_INDEX_ENVSUBST,
        timeout = 3600,
        quiet = False,
    )

    pip_parse(
        name = "pip_gpu_rocm_torch",
        requirements_lock = "@rtp_pip_deps//:requirements_lock_rocm.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS + PIP_INDEX_ARGS,
        envsubst = PIP_INDEX_ENVSUBST,
        timeout = 12000,
    )

    # Havenask compatibility parse: internal BUILD files (e.g. internal rdma)
    # load("@pip//..."), so the name must stay "pip". Repository creation is
    # lazy; public builds never reference these labels, so the parse stays
    # dormant there.
    pip_parse(
        name = "pip",
        requirements_lock = "@rtp_pip_deps//:requirements_lock_torch_gpu_cuda12.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS + PIP_INDEX_ARGS,
        envsubst = PIP_INDEX_ENVSUBST,
        timeout = 12000,
    )
