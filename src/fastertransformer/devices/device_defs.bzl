load("//:def.bzl", "copts", "cuda_copts", "torch_deps")

def device_linkopts():
    return select({
        "//:using_cuda": [
            "-L/usr/local/cuda/lib64",
            "-lcudart",
            "-lcuda",
            "-lnccl",
            "-lnvToolsExt",
        ],
        "//conditions:default": [
        ],
    })

def device_copts():
    return select({
        "//:using_cuda": {
            "TEST_USING_DEVICE": "CUDA",
        },
        "//conditions:default": {
            "TEST_USING_DEVICE": "CPU",
        },
    })

def device_test_envs():
    return select({
        "//:using_cuda": {
            "TEST_USING_DEVICE": "CUDA",
        },
        "//conditions:default": {
            "TEST_USING_DEVICE": "CPU",
        },
    })

def device_impl_target():
    return select({
        "//:using_cuda": [
            "//src/fastertransformer/devices/cuda_impl:cuda_impl"
        ],
        "//conditions:default": [
            "//src/fastertransformer/devices/cpu_impl:cpu_impl"
        ],
    })
