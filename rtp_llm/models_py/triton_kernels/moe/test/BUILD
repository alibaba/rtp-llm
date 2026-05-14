load("@arch_config//:arch_select.bzl", "requirement")

requirement(["triton", "torch"])

py_test(
    name = "test_remap_local_ids",
    srcs = ["test_remap_local_ids.py"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:moe",
    ] + [":triton", ":torch"],
    env = {"GPU_COUNT": "1"},
    tags = ["rocm"],
    exec_properties = {'gpu': 'MI308X-ROCM7', 'gpu_count': '1'},
)
