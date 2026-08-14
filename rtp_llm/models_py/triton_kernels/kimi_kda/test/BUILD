load("//bazel:arch_select.bzl", "requirement")

requirement(["torch", "triton"])

py_test(
    name = "attn_res_test",
    srcs = ["attn_res_test.py"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:kimi_kda",
        ":torch",
        ":triton",
    ],
    exec_properties = {"gpu": "B300"},
    tags = ["B300"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "rms_norm_gate_test",
    srcs = ["rms_norm_gate_test.py"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:kimi_kda",
        ":torch",
        ":triton",
    ],
    exec_properties = {"gpu": "H20"},
    tags = ["H20"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "short_conv_paged_decode_test",
    srcs = ["short_conv_paged_decode_test.py"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:kimi_kda",
        ":torch",
        ":triton",
    ],
    exec_properties = {"gpu": "H20"},
    tags = ["H20"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "paged_recurrent_layout_test",
    srcs = ["paged_recurrent_layout_test.py"],
    deps = [
        "//rtp_llm:models",
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm/models_py/triton_kernels:kimi_kda",
        ":torch",
    ],
    exec_properties = {"gpu": "H20"},
    tags = ["H20"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "cache_store_test",
    srcs = ["cache_store_test.py"],
    deps = [
        "//rtp_llm/models_py/triton_kernels:kimi_kda",
        ":torch",
        ":triton",
    ],
    exec_properties = {"gpu": "B300"},
    tags = ["B300"],
    visibility = ["//visibility:public"],
)
