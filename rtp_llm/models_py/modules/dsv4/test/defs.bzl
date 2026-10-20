"""DSV4 test execution on the CUDA13 pools with an explicit pytest entrypoint."""

load("@pip_dsv4_test//:requirements.bzl", test_requirement = "requirement")
load("//rtp_llm/test/utils:cuda13_sm100_test.bzl", "cuda13_sm100_py_test")

def dsv4_py_test(name, deps, src = None, gpu_count = 1, tags = [], args = [], **kwargs):
    source = src or name + ".py"
    runner = "//rtp_llm/test/pytest:pytest_main.py"
    cuda13_sm100_py_test(
        name = name,
        srcs = [source, runner],
        main = runner,
        args = ["--cuda-devices", str(gpu_count), "$(location :" + source + ")"] + args,
        deps = deps + ["//rtp_llm:jit_cache_env", test_requirement("pytest")],
        gpu_count = gpu_count,
        tags = ["dsv4_cuda13"] + tags,
        legacy_create_init = 0,
        **kwargs
    )

def dsv4_sm100_py_test(name, deps, src = None, extra_srcs = [], gpu_count = 1, tags = [], args = [], **kwargs):
    """Keep the shared ARM/x86 target names with the strict pytest runner."""
    source = src or name + ".py"
    runner = "//rtp_llm/test/pytest:pytest_main.py"
    cuda13_sm100_py_test(
        name = name,
        srcs = [source, runner] + extra_srcs,
        main = runner,
        args = ["--cuda-devices", str(gpu_count), "$(location :" + source + ")"] + args,
        deps = deps + ["//rtp_llm:jit_cache_env", test_requirement("pytest")],
        gpu_count = gpu_count,
        tags = ["dsv4_cuda13"] + tags,
        legacy_create_init = 0,
        **kwargs
    )
