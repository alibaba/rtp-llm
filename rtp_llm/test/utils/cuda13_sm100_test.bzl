"""CUDA 13 SM100 tests with separate ARM and x86 execution targets."""

_CUDA13_TARGETS = [
    ("", "@//:using_cuda13_arm", "SM100_ARM_CU13", "10.0"),
    ("_x86", "@//:using_cuda13_x86", "L20D_TEST", "10.3"),
]

def cuda13_sm100_py_test(name, srcs, main = None, gpu_count = 1, tags = [], env = {}, **kwargs):
    # Keep the existing pool tag and the remote execution property identical:
    # CI uses the tag for selection and the property for worker placement.
    for suffix, config, hardware_tag, compute_capability in _CUDA13_TARGETS:
        test_env = dict(env)
        test_env["EXPECTED_CUDA_COMPUTE_CAPABILITY"] = compute_capability
        test_env["EXPECTED_CUDA_MAJOR"] = "13"
        test_env["GPU_COUNT"] = str(gpu_count)
        native.py_test(
            name = name + suffix,
            srcs = srcs,
            main = main or name + ".py",
            env = test_env,
            tags = tags + [hardware_tag],
            exec_properties = {"gpu": hardware_tag, "gpu_count": str(gpu_count)},
            target_compatible_with = select({
                config: [],
                "//conditions:default": ["@platforms//:incompatible"],
            }),
            **kwargs
        )
