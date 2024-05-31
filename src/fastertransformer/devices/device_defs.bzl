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
            # NOTE: libxfastertransformer.so has conflict of std::regex related symbols with torch,
            # which causes SIGABRT on munmap_chunk() called via std::regex compiler.
            # a related discussion: https://github.com/apache/tvm/issues/9362
            # force preloading torch so to avoid the conflict.
            "LD_PRELOAD": "libtorch_cpu.so",
        },
    })

def device_impl_target():
    return select({
        "//:using_cuda": [
            "//src/fastertransformer/devices/cuda_impl:cuda_impl",
            "//3rdparty/contextFusedMultiHeadAttention:trt_fmha_impl",
            "//3rdparty/trt_fused_multihead_attention:trt_fused_multihead_attention_impl",
            "//3rdparty/flash_attention2:flash_attention2_impl",
        ],
        "//conditions:default": [
            "//src/fastertransformer/devices/cpu_impl:cpu_impl"
        ],
    })
