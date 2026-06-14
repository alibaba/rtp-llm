load("@rules_python//python:pip.bzl", "pip_parse")

PIP_EXTRA_ARGS = [
    "--cache-dir=~/.cache/pip",
    "--extra-index-url=https://mirrors.aliyun.com/pypi/simple/",
    "--verbose",
]

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
        requirements_lock = "@rtp_deps//:requirements_lock_torch_gpu_cuda12_9.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS,
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
        name = "pip_gpu_rocm_torch",
        requirements_lock = "@rtp_deps//:requirements_lock_rocm.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS,
        timeout = 12000,
    )

    # NOTE: The XPU lockfile was generated with Python 3.12 (PyTorch XPU wheels
    # require ==3.12). In the XPU Docker image, /opt/conda310/bin/python3 is a
    # symlink to /opt/venv/bin/python3 (Python 3.12), so the interpreter version
    # matches the lockfile. In non-XPU containers this path is Python 3.10, but
    # pip_parse with --generate-hashes lockfiles does not re-resolve dependencies,
    # so the version mismatch is safe in practice.
    pip_parse(
        name = "pip_xpu_torch",
        requirements_lock = "@rtp_deps//:requirements_lock_xpu.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS + ["--extra-index-url=https://download.pytorch.org/whl/xpu"],
        timeout = 3600,
    )
