load("//rtp_llm/models_py/modules/dsv4/test:defs.bzl", "dsv4_py_test", "dsv4_sm100_py_test")

_CUDA13_ONLY = select({
    "@//:using_cuda13_x86": [],
    "@//:using_cuda13_arm": [],
    "//conditions:default": ["@platforms//:incompatible"],
})

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
