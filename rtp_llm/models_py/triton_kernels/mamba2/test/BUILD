load("@arch_config//:arch_select.bzl", "requirement")

requirement([
    "triton",
    "torch",
])

py_test(
    name = "mamba2_chunk_scan_test",
    srcs = [
        "mamba2_chunk_scan_test.py",
    ],
    deps = [
        "//rtp_llm/models_py/triton_kernels:mamba2",
    ] + [
        ":torch",
        ":triton",
    ],
    visibility = ["//visibility:public"],
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
)

py_test(
    name = "mamba2_selective_state_update_test",
    srcs = [
        "mamba2_selective_state_update_test.py",
    ],
    deps = [
        "//rtp_llm/models_py/triton_kernels:mamba2",
    ] + [
        ":torch",
        ":triton",
    ],
    visibility = ["//visibility:public"],
    exec_properties = {"gpu": "L20_CU13"},
    tags = ["L20_CU13"],
)
