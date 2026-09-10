load("//rtp_llm/test/utils:cuda13_sm100_test.bzl", "cuda13_sm100_py_test")

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

py_test(
    name = "test_chunked_moe",
    srcs = ["test_chunked_moe.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    target_compatible_with = _CUDA13_ONLY,
)

cuda13_sm100_py_test(
    name = "test_shared_expert_executor",
    srcs = ["test_shared_expert_executor.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
)

cuda13_sm100_py_test(
    name = "test_mega_moe_input_packer",
    srcs = ["test_mega_moe_input_packer.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
)
