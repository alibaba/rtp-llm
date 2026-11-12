load("//rtp_llm/models_py/modules/dsv4/test:defs.bzl", "dsv4_py_test", "dsv4_sm100_py_test", "dsv4_sm120_py_test")

_CUDA13_ONLY = select({
    "@//:using_cuda13_x86": [],
    "@//:using_cuda13_arm": [],
    "//conditions:default": ["@platforms//:incompatible"],
})

dsv4_sm120_py_test(
    name = "test_sm120_decode_combine",
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    dsv4_profile_only = True,  # DeepGEMM 2.8.2 m_grouped_fp8_fp4 grouped GEMM
    visibility = ["//visibility:public"],
)

dsv4_sm120_py_test(
    name = "test_sm120_moe_numerics",
    deps = ["//rtp_llm/models_py:modules", "//rtp_llm:testlib"],
    dsv4_profile_only = True,  # fork grouped FP4 MoE + DeepGEMM 2.8.2
    visibility = ["//rtp_llm/models_py/modules/dsv4/test:__pkg__"],
)

dsv4_sm120_py_test(
    name = "test_sm120_decode_transport",
    gpu_count = 2,
    args = ["-k", "test_bounded_dispatch_with_real_nccl_2rank"],
    deps = ["//rtp_llm/models_py:modules", "//rtp_llm:testlib"],
    dsv4_profile_only = True,  # fork-pinned flashinfer mxfp8 quantize dispatch
    visibility = ["//rtp_llm/models_py/modules/dsv4/test:__pkg__"],
)

dsv4_sm120_py_test(
    name = "test_sm120_decode_transport_4rank",
    src = "test_sm120_decode_transport.py",
    gpu_count = 4,
    args = ["-k", "test_bounded_dispatch_with_real_nccl_4rank"],
    # Required DP4/EP4 DECODE baseline. The dedicated four-GPU CI job selects
    # it explicitly; multi_device keeps it out of broad UT lanes.
    tags = ["multi_device"],
    deps = ["//rtp_llm/models_py:modules", "//rtp_llm:testlib"],
    dsv4_profile_only = True,  # fork-pinned flashinfer mxfp8 quantize dispatch
    visibility = ["//rtp_llm/models_py/modules/dsv4/test:__pkg__"],
)

py_test(
    name = "timeline_test",
    srcs = ["timeline_test.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "quantized_linear_import_test",
    srcs = ["quantized_linear_import_test.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "shared_expert_test",
    srcs = ["shared_expert_test.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "fp8_fp4_layer_test",
    srcs = ["fp8_fp4_layer_test.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    visibility = ["//rtp_llm/models_py/modules/dsv4/test:__pkg__"],
)

py_test(
    name = "test_moe_weight_adapter",
    srcs = ["test_moe_weight_adapter.py"],
    deps = ["//rtp_llm/models_py:modules", "//rtp_llm:testlib"],
    target_compatible_with = _CUDA13_ONLY,
)

py_test(
    name = "test_mega_moe_jit_warmup",
    srcs = ["test_mega_moe_jit_warmup.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    target_compatible_with = _CUDA13_ONLY,
)

py_test(
    name = "mega_moe_static_buf_test",
    srcs = ["mega_moe_static_buf_test.py"],
    deps = ["//rtp_llm:testlib"],
    target_compatible_with = _CUDA13_ONLY,
)

dsv4_py_test(
    name = "test_chunked_moe",
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    visibility = ["//visibility:public"],
)

dsv4_sm100_py_test(
    name = "test_shared_expert_executor",
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    visibility = ["//visibility:public"],
)

dsv4_sm100_py_test(
    name = "test_mega_moe_input_packer",
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    visibility = ["//visibility:public"],
)

# SM120 layer/strategy cases split out of fp8_fp4_layer_test.py.
py_test(
    name = "fp8_fp4_layer_sm120_test",
    srcs = ["fp8_fp4_layer_sm120_test.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    visibility = ["//rtp_llm/models_py/modules/dsv4/test:__pkg__"],
)
