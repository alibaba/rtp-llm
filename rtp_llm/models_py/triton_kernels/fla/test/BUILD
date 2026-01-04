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