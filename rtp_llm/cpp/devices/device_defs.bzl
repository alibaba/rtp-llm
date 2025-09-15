load("//bazel:arch_select.bzl", "torch_deps")


def device_test_envs():
    return select({
        "@//:using_cuda": {
            "TEST_USING_DEVICE": "CUDA",
            "LD_PRELOAD": "libtorch_cpu.so",
        },
        "@//:using_rocm": {
            "TEST_USING_DEVICE": "ROCM",
            # "LD_PRELOAD": "/opt/conda310/lib/python3.10/site-packages/torch/lib/libtorch_cpu.so",
        },
        "@//:using_arm": {
            "TEST_USING_DEVICE": "ARM",
        },
        "//conditions:default": {
            "TEST_USING_DEVICE": "CPU",
            # NOTE: libxfastertransformer.so has conflict of std::regex related symbols with torch,
            # which causes SIGABRT on munmap_chunk() called via std::regex compiler.
            # a related discussion: https://github.com/apache/tvm/issues/9362
            # force preloading torch so to avoid the conflict.
            "LD_PRELOAD": "libtorch_cpu.so",
        },
    })

def device_impl_target():
    return select({
        "@//:using_cuda": [
            "//rtp_llm/cpp/devices/cuda_impl:cuda_impl",
        ],
        "@//:using_rocm": [
            "//rtp_llm/cpp/devices/rocm_impl:rocm_impl",
        ],
        "@//:using_arm": [
            "//rtp_llm/cpp/devices/arm_impl:arm_cpu_impl",
        ],
        "//conditions:default": [
            "//rtp_llm/cpp/devices/cpu_impl:cpu_impl"
        ],
    })
