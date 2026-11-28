"""CUDA 13 SM120 test rule mirroring cuda13_sm100_test.bzl."""


def cuda13_sm120_py_test(
    name, srcs, main=None, gpu_count=1, tags=[], env={}, **kwargs
):
    test_env = dict(env)
    test_env["GPU_COUNT"] = str(gpu_count)
    extra_tags = ["multi_device"] if gpu_count > 1 and "multi_device" not in tags else []
    native.py_test(
        name=name,
        srcs=srcs,
        main=main or name + ".py",
        env=test_env,
        tags=["RTX_5000_PRO_CU13"] + extra_tags + tags,
        exec_properties={"gpu": "RTX_5000_PRO_CU13", "gpu_count": str(gpu_count)},
        target_compatible_with=select(
            {
                "@//:using_cuda13_x86": [],
                "//conditions:default": ["@platforms//:incompatible"],
            }
        ),
        **kwargs
    )
