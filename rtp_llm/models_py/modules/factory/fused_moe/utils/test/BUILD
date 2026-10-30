load("//rtp_llm/models_py/modules/dsv4/test:defs.bzl", "dsv4_py_test", "dsv4_sm100_py_test")

_CUDA13_ONLY = select({
    "@//:using_cuda13_x86": [],
    "@//:using_cuda13_arm": [],
    "//conditions:default": ["@platforms//:incompatible"],
})

py_test(
    name = "test_mega_moe_activation_contract",
    srcs = ["test_mega_moe_activation_contract.py"],
    data = ["//rtp_llm/models_py:modules"],
)

py_test(
    name = "test_mega_moe_backend",
    srcs = ["test_mega_moe_backend.py"],
    deps = ["//rtp_llm:testlib"],
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

# Importing the K3 model loads CUDA native libraries even with its launches mocked.
# Keep separate targets because tags cannot use select().
py_test(
    name = "test_k3_fused_shared_jit_warmup",
    srcs = ["test_k3_fused_shared_jit_warmup.py"],
    main = "test_k3_fused_shared_jit_warmup.py",
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "L20_CU13", "gpu_count": "1"},
    tags = ["L20_CU13"],
    target_compatible_with = select({
        "@//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "test_k3_fused_shared_jit_warmup_arm",
    srcs = ["test_k3_fused_shared_jit_warmup.py"],
    main = "test_k3_fused_shared_jit_warmup.py",
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "SM100_ARM_CU13", "gpu_count": "1"},
    tags = ["SM100_ARM_CU13"],
    target_compatible_with = select({
        "@//:using_cuda13_arm": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "mega_moe_static_buf_test",
    srcs = ["mega_moe_static_buf_test.py"],
    deps = ["//rtp_llm:testlib"],
    target_compatible_with = _CUDA13_ONLY,
)

py_test(
    name = "mega_moe_k3_buffer_contract_test",
    srcs = ["mega_moe_k3_buffer_contract_test.py"],
    data = ["//rtp_llm/models_py:modules/factory/fused_moe/utils/mega_moe/buffer.py"],
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
