load("@rules_python//python:pip.bzl", "pip_parse")

PIP_EXTRA_ARGS = [
    "--cache-dir=~/.cache/pip",
    "--index-url=http://artlab.alibaba-inc.com/1/pypi/rtp_diffusion",
    "--extra-index-url=https://artlab.alibaba-inc.com/1/pypi/huiwa_rtp_internal",
    "--extra-index-url=https://artlab.alibaba-inc.com/1/PYPI/simple/",
    "--trusted-host=artlab.alibaba-inc.com",
    "--verbose",
]

def pip_deps():
    pip_parse(
        name = "pip_cpu_torch",
        requirements_lock = "//internal_source/deps:requirements_lock_torch_cpu.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS,
        timeout = 3600,
    )

    pip_parse(
        name = "pip_arm_torch",
        requirements_lock = "//internal_source/deps:requirements_lock_torch_arm.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS,
        timeout = 3600,
    )

    pip_parse(
        name = "pip_ppu_torch",
        requirements_lock = "//internal_source/RTP_LLM-PPU/rtp_llm:requirements_lock_torch_ppu.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS,
        timeout = 3600,
    )

    pip_parse(
        name = "pip_gpu_cuda12_torch",
        requirements_lock = "//internal_source/deps:requirements_lock_torch_gpu_cuda12.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS,
        timeout = 3600,
        quiet = False,
    )

    pip_parse(
        name = "pip_gpu_cuda12_9_torch",
        requirements_lock = "//internal_source/deps:requirements_lock_torch_gpu_cuda12_9.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS + ["--quiet"],
        timeout = 3600,
        quiet = False,
    )

    pip_parse(
        name = "pip_gpu_cuda13_torch",
        requirements_lock = "//internal_source/deps:requirements_lock_torch_gpu_cuda13.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS + ["--quiet"],
        timeout = 3600,
        quiet = False,
    )

    pip_parse(
        name = "pip_cuda12_arm_torch",
        requirements_lock = "//internal_source/deps:requirements_lock_cuda12_arm.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS,
        timeout = 3600,
        quiet = False,
    )

    pip_parse(
        name = "pip_cuda13_arm_torch",
        requirements_lock = "//internal_source/deps:requirements_lock_cuda13_arm.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS,
        timeout = 3600,
        quiet = False,
    )

    pip_parse(
        name = "pip_gpu_rocm_torch",
        requirements_lock = "//internal_source/deps:requirements_lock_rocm.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS,
        timeout = 12000,
    )

    # for alibaba compile defs.bzl
    pip_parse(
        # to keep compat with havenask
        # must use the name "pip"
        # otherwise, we should modify lots of BUILD files in havenask
        # check load("@pip//xxx")
        name = "pip",
        requirements_lock = "//internal_source/deps:requirements_lock_torch_gpu_cuda12.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS,
        timeout = 12000,
    )
