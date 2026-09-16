"""CUDA 13 SM100 tests with separate ARM and x86 execution targets."""

def cuda13_sm100_py_test(name, srcs, main = None, gpu_count = 1, tags = [], env = {}, **kwargs):
    test_env = dict(env)
    test_env["GPU_COUNT"] = str(gpu_count)
    # Internal CUDA13 CI maps L20D_TEST to B300 DGX (SM 10.3), not L20.
    # SM100_ARM_CU13 is the separate CUDA13 GB200 pool.
    for suffix, config, gpu in [
        ("", "@//:using_cuda13_arm", "SM100_ARM_CU13"),
        ("_x86", "@//:using_cuda13_x86", "L20D_TEST"),
    ]:
        native.py_test(
            name = name + suffix,
            srcs = srcs,
            main = main or name + ".py",
            env = test_env,
            tags = tags + [gpu],
            exec_properties = {"gpu": gpu, "gpu_count": str(gpu_count)},
            target_compatible_with = select({
                config: [],
                "//conditions:default": ["@platforms//:incompatible"],
            }),
            **kwargs
        )
