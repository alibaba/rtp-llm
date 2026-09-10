load("//:def.bzl", "rocm_copts")

cc_test(
    name = "trtllm_allreduce_sync_test",
    srcs = ["trtllm_allreduce_sync_test.cu"],
    copts = rocm_copts(),
    deps = [
        "//rtp_llm/models_py/bindings/rocm/kernels:rocm_trtllm_allreduce_sync",
        "@com_google_googletest//:gtest_main",
        "@local_config_rocm//rocm:hip",
        "@local_config_rocm//rocm:rocm_headers",
    ],
    env = {"GPU_COUNT": "2"},
    tags = ["rocm"],
    timeout = "moderate",
    exec_properties = {
        "gpu": "MI308X-ROCM7",
        "gpu_count": "2",
    },
)
