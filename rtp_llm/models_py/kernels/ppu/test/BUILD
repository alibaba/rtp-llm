load("@arch_config//:arch_select.bzl", "requirement")

requirement(["deep_gemm", "flashinfer-python", "numpy", "torch"])

py_test(
    name = "test_group_topk",
    srcs = ["test_group_topk.py"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "PPU-ZW810E", "gpu_count": "1"},
    tags = ["PPU-ZW810E", "manual"],
    target_compatible_with = select({
        "//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        "//rtp_llm/models_py:internal_kernels",
        "//rtp_llm/models_py:ppu_modules",
        ":torch",
    ],
)

py_test(
    name = "test_ragged_fp8_quant",
    srcs = ["test_ragged_fp8_quant.py"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "PPU-ZW810E", "gpu_count": "1"},
    tags = ["PPU-ZW810E", "manual"],
    target_compatible_with = select({
        "//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        "//rtp_llm/models_py:internal_kernels",
        "//rtp_llm/models_py:ppu_modules",
        ":torch",
    ],
)

py_test(
    name = "test_sparse_mla_decode",
    srcs = ["test_sparse_mla_decode.py"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "PPU-ZW810E", "gpu_count": "1"},
    tags = ["PPU-ZW810E", "manual"],
    target_compatible_with = select({
        "//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        "//rtp_llm/models_py:internal_kernels",
        "//rtp_llm/models_py:ppu_modules",
        ":torch",
    ],
)

py_test(
    name = "test_fp8_normal_layout",
    srcs = ["test_fp8_normal_layout.py"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "PPU-ZW810E", "gpu_count": "1"},
    tags = ["PPU-ZW810E", "manual"],
    target_compatible_with = select({
        "//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        "//rtp_llm/models_py:internal_kernels",
        "//rtp_llm/models_py:ppu_modules",
        ":torch",
    ],
)

py_test(
    name = "test_fp8_pd_strategy",
    srcs = ["test_fp8_pd_strategy.py"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "PPU-ZW810E", "gpu_count": "1"},
    tags = ["PPU-ZW810E", "manual"],
    target_compatible_with = select({
        "//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = ["//rtp_llm/models_py:ppu_modules", ":torch"],
)

py_test(
    name = "test_block_scaled_fp8_gemm",
    srcs = ["test_block_scaled_fp8_gemm.py"],
    data = ["test_block_scaled_fp8_sgl_golden.json"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "PPU-ZW810E", "gpu_count": "1"},
    tags = ["PPU-ZW810E", "manual"],
    target_compatible_with = select({
        "//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        "//rtp_llm/models_py:internal_kernels",
        "//rtp_llm/models_py:ppu_modules",
        ":numpy",
        ":torch",
    ],
)

py_test(
    name = "test_fused_quant",
    srcs = ["test_fused_quant.py"],
    data = ["test_fp8_quant_sgl_golden.json"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "PPU-ZW810E", "gpu_count": "1"},
    tags = ["PPU-ZW810E", "manual"],
    target_compatible_with = select({
        "//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:internal_kernels",
        "//rtp_llm/models_py:ppu_modules",
        "//rtp_llm/models_py/triton_kernels:triton_kernels",
        ":flashinfer-python",
        ":torch",
    ],
)

py_test(
    name = "test_indexer_k_quant",
    srcs = ["test_indexer_k_quant.py"],
    data = ["test_indexer_k_sgl_golden.json"],
    exec_properties = {"gpu": "PPU-ZW810E"},
    tags = ["PPU-ZW810E", "manual"],
    target_compatible_with = select({
        "//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = ["//rtp_llm/models_py:ppu_modules", ":torch"],
)

py_test(
    name = "test_indexer_k_rope",
    srcs = ["test_indexer_k_rope.py"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "PPU-ZW810E", "gpu_count": "1"},
    tags = ["PPU-ZW810E", "manual"],
    target_compatible_with = select({
        "//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        "//rtp_llm/models_py:ppu_modules",
        "//rtp_llm/models_py/triton_kernels:triton_kernels",
        ":torch",
    ],
)

py_test(
    name = "test_glm5_indexer_query_scale",
    srcs = ["test_glm5_indexer_query_scale.py"],
    data = ["test_indexer_query_ue8_sgl_golden.json"],
    exec_properties = {"gpu": "PPU-ZW810E"},
    tags = ["PPU-ZW810E", "manual"],
    target_compatible_with = select({
        "//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = ["//rtp_llm/models_py:ppu_modules", ":torch"],
)

py_test(
    name = "test_mla_kv_quant",
    srcs = ["test_mla_kv_quant.py"],
    data = ["test_mla_kv_sgl_golden.json"],
    exec_properties = {"gpu": "PPU-ZW810E"},
    tags = ["PPU-ZW810E", "manual"],
    target_compatible_with = select({
        "//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = ["//rtp_llm/models_py:ppu_modules", ":torch"],
)

py_test(
    name = "test_fp8_quant",
    srcs = ["test_fp8_quant.py"],
    data = ["test_fp8_quant_sgl_golden.json"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "PPU-ZW810E", "gpu_count": "1"},
    tags = ["PPU-ZW810E", "manual"],
    target_compatible_with = select({
        "//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py:internal_kernels",
        ":deep_gemm",
        ":torch",
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "test_deepgemm_warmup",
    srcs = ["test_deepgemm_warmup.py"],
    exec_properties = {"gpu": "PPU-ZW810E"},
    tags = [
        "PPU-ZW810E",
        "manual",
    ],
    target_compatible_with = select({
        "//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        "//rtp_llm/models_py:internal_kernels",
    ] + [":torch"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "test_deepgemm_wrapper",
    srcs = ["test_deepgemm_wrapper.py"],
    exec_properties = {"gpu": "PPU-ZW810E"},
    tags = [
        "PPU-ZW810E",
        "manual",
    ],
    target_compatible_with = select({
        "//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = ["//rtp_llm/models_py:internal_kernels"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "test_indexer_topk_order",
    srcs = ["test_indexer_topk_order.py"],
    exec_properties = {"gpu": "PPU-ZW810E"},
    tags = ["PPU-ZW810E", "manual"],
    target_compatible_with = select({
        "//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = ["//rtp_llm/models_py:ppu_modules", ":torch"],
)

py_test(
    name = "test_glm5_gate_fp32",
    srcs = ["test_glm5_gate_fp32.py"],
    exec_properties = {"gpu": "PPU-ZW810E"},
    tags = ["PPU-ZW810E", "manual"],
    target_compatible_with = select({
        "//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = ["//rtp_llm/models_py:ppu_modules", ":torch"],
)

py_test(
    name = "test_deepgemm_masked_graph",
    srcs = ["test_deepgemm_masked_graph.py"],
    env = {"GPU_COUNT": "1"},
    exec_properties = {
        "gpu": "PPU-ZW810E",
        "gpu_count": "1",
    },
    tags = [
        "PPU-ZW810E",
        "manual",
    ],
    target_compatible_with = select({
        "//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        "//rtp_llm/models_py:internal_kernels",
        ":deep_gemm",
        ":numpy",
        ":torch",
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "test_deepgemm_fp8_grouped_graph",
    srcs = ["test_deepgemm_fp8_grouped_graph.py"],
    exec_properties = {"gpu": "PPU-ZW810E"},
    tags = ["PPU-ZW810E", "manual"],
    target_compatible_with = select({
        "//:using_ppu": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = ["//rtp_llm/models_py:internal_kernels", ":deep_gemm", ":torch", ":numpy"],
)
