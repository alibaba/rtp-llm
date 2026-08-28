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
        "@//:using_ascend": {
            "TEST_USING_DEVICE": "ASCEND",
        },
        "//conditions:default": {
            "TEST_USING_DEVICE": "CUDA",
            "LD_PRELOAD": "libtorch_cpu.so",
        },
    })

def device_impl_target():
    return select({
        "@//:using_cuda": [
            "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        ],
        "@//:using_ascend": [
            # Phase 5: "//rtp_llm/models_py/bindings/ascend/ops:ascend_impl",
        ],
        "//conditions:default": [],
    })
