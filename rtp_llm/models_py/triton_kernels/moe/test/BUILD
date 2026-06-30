load("@arch_config//:arch_select.bzl", "requirement")

requirement(["triton", "torch", "numpy"])

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

py_test(
    name = "test_ep_kernels",
    srcs = ["test_ep_kernels.py"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:moe",
    ] + [":triton", ":torch", ":numpy"],
    visibility = ["//visibility:public"],
    exec_properties = {"gpu": "H20"},
)

py_test(
    name = "test_ep_scatter",
    srcs = ["test_ep_scatter.py"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:moe",
    ] + [":triton", ":torch"],
    visibility = ["//visibility:public"],
    exec_properties = {"gpu": "H20"},
)
