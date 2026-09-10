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
    name = "fused_paged_prefill_test",
    srcs = ["fused_paged_prefill_test.py"],
    deps = [
        "//rtp_llm:nvidia-cutlass-dsl",
        "//rtp_llm/models_py:modules",
        "//rtp_llm/models_py/triton_kernels:kimi_kda",
        ":torch",
        ":triton",
    ],
    exec_properties = {"gpu": "B300"},
    tags = ["B300"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "fp8_quant_test",
    srcs = ["fp8_quant_test.py"],
    deps = [
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
        "//rtp_llm/models_py/triton_kernels:kimi_kda",
        ":torch",
        ":triton",
    ],
    exec_properties = {"gpu": "B300"},
    tags = ["B300"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "producer_fp8_test",
    srcs = ["producer_fp8_test.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    tags = ["manual", "B300"],
    exec_properties = {"gpu": "B300"},
)

py_test(
    name = "collective_fp8_test",
    srcs = ["collective_fp8_test.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib", "//rtp_llm/models_py/modules/hybrid/test:collective_gemm_reference"],
    tags = ["manual", "B300"],
    exec_properties = {"gpu": "B300"},
)

py_binary(
    name = "producer_fp8_benchmark",
    srcs = ["producer_fp8_benchmark.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    tags = ["manual", "B300"],
)

py_test(
    name = "dynamic_length_fp8_test",
    srcs = ["dynamic_length_fp8_test.py"],
    deps = ["//rtp_llm/models_py/standalone:py_standalone_testlib"],
    tags = ["manual", "B300"],
    exec_properties = {"gpu": "B300"},
)
