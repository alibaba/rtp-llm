load("@rules_python//python:pip.bzl", "pip_install")

def pip_deps():    
    pip_install(
        name = "pip",
        requirements = "//maga_transformer:requirements.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = [
            "--index-url=https://rtp-pypi-mirrors.alibaba-inc.com/root/pypi/+simple/",
            "--cache-dir=~/.cache/pip",
            "--log=pip.log",
        ]
    )

    pip_install(
    name = "pip_gpu_torch",
    requirements = "//maga_transformer:requirements_torch_gpu.txt",
    python_interpreter = "/opt/conda310/bin/python3",
    extra_pip_args = [
        "--index-url=https://rtp-pypi-mirrors.alibaba-inc.com/root/pypi/+simple/",
        "--cache-dir=~/.cache/pip",
        "--log=pip.log",
    ]
    )

    pip_install(
        name = "pip_gpu_cuda12_torch",
        requirements = "//maga_transformer:requirements_torch_gpu_cuda12.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = [
            "--index-url=https://rtp-pypi-mirrors.alibaba-inc.com/root/pypi/+simple/",
            "--cache-dir=~/.cache/pip",
            "--log=pip.log",
        ]
    )