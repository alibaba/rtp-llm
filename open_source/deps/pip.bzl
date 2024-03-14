load("@rules_python//python:pip.bzl", "pip_install")

def pip_deps():
    pip_install(
        name = "pip_gpu_torch",
        requirements = ["//open_source/deps:requirements_torch_gpu.txt", "//open_source/deps:requirements_base.txt"],
        python_interpreter = "/opt/conda310/bin/python3",
    )

    pip_install(
        name = "pip_gpu_cuda12_torch",
        requirements = ["//open_source/deps:requirements_torch_gpu_cuda12.txt", "//open_source/deps:requirements_base.txt"],
        python_interpreter = "/opt/conda310/bin/python3",
    )
