def cuda13_sm100_py_test(name, srcs, main = None, gpu_count = 1, tags = [], env = {}, **kwargs):
    test_env = dict(env)
    test_env["GPU_COUNT"] = str(gpu_count)
    for suffix, config, gpu in [
        ("", "@//:using_cuda13_arm", "SM100_ARM_CU13"),
        ("_x86", "@//:using_cuda13_x86", "SM103_CU13"),
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
