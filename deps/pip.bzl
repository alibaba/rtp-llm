load("@rules_python//python:pip.bzl", "pip_parse")

PIP_EXTRA_ARGS = [
    "--cache-dir=~/.cache/pip",
    "--index-url=https://mirrors.aliyun.com/pypi/simple/",
    "--extra-index-url=https://pypi.tuna.tsinghua.edu.cn/simple/",
    "--find-links=/home/wheels",
    "--timeout=30",
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
        name = "pip_gpu_cuda13_torch",
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

    pip_parse(
        name = "pip_gpu_dcu_torch",
	requirements_lock = "@rtp_deps//:requirements_lock_dcu.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS,
        timeout = 12000,
        # aiter ships triton autotune configs whose file names contain ':' --
        # invalid in bazel labels, so keep them out of the generated data glob
        # (runtime reads them from the pip-installed wheel instead).
        pip_data_exclude = ["site-packages/aiter/ops/triton/configs/**"],
    )
