load("@arch_config//:arch_select.bzl", "torch_deps")


def device_test_envs():
    return select({
        "@//:using_cuda": {
            "TEST_USING_DEVICE": "CUDA",
            "LD_PRELOAD": "libtorch_cpu.so",
        },
        "@//:using_rocm": {
            "TEST_USING_DEVICE": "ROCM",
        },
        "//conditions:default": {
            "TEST_USING_DEVICE": "CUDA",
            "LD_PRELOAD": "libtorch_cpu.so",
        },
    })

def device_impl_target():
    return select({
        "@//:using_cuda": [
            "//rtp_llm/models_py/bindings/core:sampling_ops_test_impls",
        ],
        "@//:using_rocm": [
            "//rtp_llm/models_py/bindings/core:sampling_ops_test_impls",
        ],
        "//conditions:default": [],
    })
