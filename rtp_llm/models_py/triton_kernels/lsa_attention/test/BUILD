load("@arch_config//:arch_select.bzl", "requirement")

requirement([
    "triton",
    "torch",
])

py_test(
    name = "lsa_reference_test",
    srcs = [
        "test_lsa_reference_test.py",
    ],
    deps = [
        "//rtp_llm/models_py/triton_kernels:lsa_attention",
    ] + [
        ":torch",
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "sparse_mla_triton_test",
    srcs = [
        "test_sparse_mla_triton_test.py",
    ],
    deps = [
        "//rtp_llm/models_py/triton_kernels:lsa_attention",
    ] + [
        ":torch",
        ":triton",
    ],
    visibility = ["//visibility:public"],
    exec_properties = {
        "gpu": "A10_CU13",
        "gpu_count": "1",
    },
    tags = [
        "A10_CU13",
    ],
)

py_test(
    name = "swa_dense_triton_test",
    srcs = [
        "test_swa_dense_triton_test.py",
    ],
    deps = [
        "//rtp_llm/models_py/triton_kernels:lsa_attention",
    ] + [
        ":torch",
        ":triton",
    ],
    visibility = ["//visibility:public"],
    exec_properties = {
        "gpu": "A10_CU13",
        "gpu_count": "1",
    },
    tags = [
        "A10_CU13",
    ],
)

py_test(
    name = "merge_triton_test",
    srcs = [
        "test_merge_triton_test.py",
    ],
    deps = [
        "//rtp_llm/models_py/triton_kernels:lsa_attention",
    ] + [
        ":torch",
        ":triton",
    ],
    visibility = ["//visibility:public"],
    exec_properties = {
        "gpu": "A10_CU13",
        "gpu_count": "1",
    },
    tags = [
        "A10_CU13",
    ],
)

py_test(
    name = "split_merge_consistency_test",
    srcs = [
        "test_split_merge_consistency_test.py",
    ],
    deps = [
        "//rtp_llm/models_py/triton_kernels:lsa_attention",
    ] + [
        ":torch",
        ":triton",
    ],
    visibility = ["//visibility:public"],
    exec_properties = {
        "gpu": "A10_CU13",
        "gpu_count": "1",
    },
    tags = [
        "A10_CU13",
    ],
)