load("//bazel:arch_select.bzl", "torch_deps")


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
            "//rtp_llm/cpp/cuda/ops:cuda_impl",
        ],
        "//conditions:default": [],
    })
