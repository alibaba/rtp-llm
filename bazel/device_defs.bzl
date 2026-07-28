load("@arch_config//:arch_select.bzl", "torch_deps")


def cuda_test_envs():
    return {
        "TEST_USING_DEVICE": "CUDA",
        "LD_PRELOAD": "libtorch_cpu.so",
    }

def rocm_test_envs():
    return {
        "TEST_USING_DEVICE": "ROCM",
    }

def device_test_envs():
    return select({
        "@//:using_cuda": cuda_test_envs(),
        "@//:using_rocm": rocm_test_envs(),
        "//conditions:default": cuda_test_envs(),
    })

def device_runtime_deps():
    return select({
        "@//:using_cuda": [
            "@local_config_cuda//cuda:cuda_headers",
            "@local_config_cuda//cuda:cudart",
        ],
        "@//:using_rocm": [
            "@local_config_rocm//rocm:rocm_headers",
            "@local_config_rocm//rocm:hip",
        ],
        "//conditions:default": [
            "@local_config_cuda//cuda:cuda_headers",
            "@local_config_cuda//cuda:cudart",
        ],
    })

def device_impl_target():
    return select({
        "@//:using_cuda": [
            "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        ],
        # ROCm has no equivalent Python-op registration target. Tests using this
        # helper exercise the C++ device path without a static op registry.
        "@//:using_rocm": [],
        "//conditions:default": [],
    })
