load("//bazel:arch_select.bzl", "requirement")

requirement(["triton", "torch", "numpy", "einops", "packaging"])

py_test(
    name = "test_gdn_block_prefill",
    srcs = [
        "test_gdn_block_prefill.py"
    ],
    deps = [
        "//rtp_llm/models_py/triton_kernels:fla",
    ] + [":triton", ":torch", ":numpy", ":einops", ":packaging"],
    visibility = ["//visibility:public"],
    exec_properties = {'gpu':'H20'},
)

py_test(
    name = "test_gdn_decode",
    srcs = [
        "test_gdn_decode.py"
    ],
    deps = [
        "//rtp_llm/models_py/triton_kernels:fla",
    ] + [":triton", ":torch", ":numpy", ":einops", ":packaging"],
    visibility = ["//visibility:public"],
    exec_properties = {'gpu':'H20'},
)

py_test(
    name = "test_chunk_prefill",
    srcs = [
        "test_chunk_prefill.py"
    ],
    deps = [
        "//rtp_llm/models_py/triton_kernels:fla",
    ] + [":triton", ":torch", ":numpy", ":einops", ":packaging"],
    visibility = ["//visibility:public"],
    exec_properties = {'gpu':'H20'},
)

py_test(
    name = "test_l2norm",
    srcs = [
        "test_l2norm.py"
    ],
    deps = [
        "//rtp_llm/models_py/triton_kernels:fla",
    ] + [":triton", ":torch"],
    visibility = ["//visibility:public"],
    exec_properties = {'gpu':'H20'},
)

py_test(
    name = "test_chunk_prefill_cp_zigzag",
    srcs = [
        "test_chunk_prefill_cp_zigzag.py"
    ],
    deps = [
        "//rtp_llm/models_py/triton_kernels:fla",
        "//rtp_llm:testlib",
    ] + [":triton", ":torch"],
    visibility = ["//visibility:public"],
    env = {"GPU_COUNT": "2"},
    exec_properties = {'gpu': 'H20', 'gpu_count': '2'},
)