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

def dsv4_sm120_py_test(name, deps, src = None, extra_srcs = [], gpu_count = 1, tags = [], args = [], env = {}, sm120_build_only = False, **kwargs):
    """Run on the SM120 pool; plain cuda13 builds run these as blocking tests.

    sm120_build_only keeps the exact cuda13_sm120 build for tests that need the
    forked DeepGEMM/FlashInfer providers or more than two GPUs.
    """
    source = src or name + ".py"
    # SM120 lanes need the --cuda-major/--cuda-minor gate; that launcher
    # lives in the sibling pytest_main_sm120.py so pytest_main.py stays
    # unmodified.
    runner = "//rtp_llm/test/pytest:pytest_main_sm120.py"
    main_runner = "//rtp_llm/test/pytest:pytest_main.py"
    test_env = dict(env)
    test_env["GPU_COUNT"] = str(gpu_count)
    # using_cuda13_x86 also matches --config=cuda13_sm120 (strict superset).
    compatible_setting = "@//:using_cuda13_sm120" if sm120_build_only else "@//:using_cuda13_x86"
    # SM120_CU13 lets the opt-in aone lane select these tests by
    # --test_tag_filters like every other lane; under plain cuda13
    # sm120_build_only targets are still skipped via target_compatible_with.
    lane_tags = ["SM120_CU13"] if sm120_build_only else []
    native.py_test(
        name = name,
        srcs = [source, runner, main_runner] + extra_srcs,
        main = runner,
        args = ["--cuda-devices", str(gpu_count), "--cuda-major", "12", "--cuda-minor", "0", "$(location :" + source + ")"] + args,
        deps = deps + ["//rtp_llm:jit_cache_env", test_requirement("pytest")],
        env = test_env,
        tags = ["dsv4_cuda13", "RTX_5000_PRO_CU13"] + lane_tags + tags,
        exec_properties = {"gpu": "RTX_5000_PRO_CU13", "gpu_count": str(gpu_count)},
        target_compatible_with = select({
            compatible_setting: [],
            "//conditions:default": ["@platforms//:incompatible"],
        }),
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
