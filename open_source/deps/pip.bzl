load("@rules_python//python:pip.bzl", "pip_parse")

def pip_deps():
    pip_parse(
        name = "pip_cpu_torch",
        requirements_lock = "//open_source/deps:requirements_lock_torch_cpu.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        timeout = 3600,
    )

    pip_parse(
        name = "pip_gpu_torch",
        requirements_lock = "//open_source/deps:requirements_lock_torch_gpu.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        timeout = 3600,
    )

    pip_parse(
        name = "pip_gpu_cuda12_torch",
        requirements_lock = "//open_source/deps:requirements_lock_torch_gpu_cuda12.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        timeout = 3600,
        quiet = False,
    )

    pip_parse(
        name = "pip_gpu_rocm_torch",
        requirements_lock = "//open_source/deps:requirements_lock_rocm.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        timeout = 12000,
    )

    pip_install(
        name = "pip_cpu_arm_torch",
        requirements = ["//open_source/deps:requirements_cpu_arm.txt", "//open_source/deps:requirements_base.txt"],
        python_interpreter = "/opt/conda310/bin/python3",
        timeout=12000,
    )
