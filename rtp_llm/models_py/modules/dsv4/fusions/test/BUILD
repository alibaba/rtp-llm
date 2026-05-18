py_library(
    name = "graphfx_fusion_test_utils",
    srcs = ["graphfx_fusion_test_utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/models_py/modules/dsv4/test:dsv4_kernel_perf_utils",
        "//rtp_llm:torch",
    ],
)

py_test(
    name = "test_graphfx_injector",
    srcs = ["test_graphfx_injector.py"],
    deps = [
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "test_fused_rmsnorm_fp8_quant",
    srcs = ["test_fused_rmsnorm_fp8_quant.py"],
    deps = [
        ":graphfx_fusion_test_utils",
        "//rtp_llm/models_py:kernels",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "test_fused_rmsnorm_bf16_fp8_quant",
    srcs = ["test_fused_rmsnorm_bf16_fp8_quant.py"],
    deps = [
        ":graphfx_fusion_test_utils",
        "//rtp_llm/models_py:kernels",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "test_dsv4_indexed_rmsnorm_rope",
    srcs = ["test_dsv4_indexed_rmsnorm_rope.py"],
    data = [
        "//:rtp_compute_ops",
        "//:th_transformer_config",
    ],
    deps = [
        ":graphfx_fusion_test_utils",
        "//rtp_llm/models_py:dsv4_indexed_rope_test_sources",
        "//rtp_llm/models_py:kernels",
        "//rtp_llm:config",
        "//rtp_llm:device",
        "//rtp_llm:torch",
    ],
)

py_test(
    name = "test_dsv4_indexed_rmsnorm_rope_direct_op",
    srcs = ["test_dsv4_indexed_rmsnorm_rope_direct_op.py"],
    data = [
        "//:rtp_compute_ops",
        "//:th_transformer_config",
    ],
    deps = [
        "//rtp_llm:ops",
        "//rtp_llm:torch",
    ],
    tags = ["manual", "H20", "SM100_ARM"],
    exec_properties = {"gpu": "H20"},
)

py_test(
    name = "test_dsv4_indexed_rmsnorm_rope_runtime_gate",
    srcs = ["test_dsv4_indexed_rmsnorm_rope_runtime_gate.py"],
    data = [
        "//:rtp_compute_ops",
        "//:th_transformer_config",
    ],
    deps = [
        "//rtp_llm/models_py:kernels",
        "//rtp_llm:torch",
    ],
    tags = ["H20", "SM100_ARM"],
    exec_properties = {"gpu": "H20"},
)

py_test(
    name = "test_dsv4_indexed_inv_rope_fp8_quant",
    srcs = ["test_dsv4_indexed_inv_rope_fp8_quant.py"],
    deps = [
        ":graphfx_fusion_test_utils",
        "//rtp_llm/models_py:kernels",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "test_kv_rope_fp8_quant_graphfx",
    srcs = ["test_kv_rope_fp8_quant_graphfx.py"],
    deps = [
        ":graphfx_fusion_test_utils",
        "//rtp_llm/models_py:kernels",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
)

py_test(
    name = "test_fused_rmsnorm_fp8_quant_perf",
    srcs = [
        "test_fused_rmsnorm_fp8_quant.py",
        "test_fused_rmsnorm_fp8_quant_perf.py",
    ],
    deps = [
        ":graphfx_fusion_test_utils",
        "//rtp_llm/models_py/modules/dsv4/test:dsv4_kernel_perf_utils",
        "//rtp_llm/models_py:kernels",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    tags = ["manual", "H20", "SM100_ARM"],
)

py_test(
    name = "test_fused_rmsnorm_bf16_fp8_quant_perf",
    srcs = [
        "test_fused_rmsnorm_bf16_fp8_quant.py",
        "test_fused_rmsnorm_bf16_fp8_quant_perf.py",
    ],
    deps = [
        ":graphfx_fusion_test_utils",
        "//rtp_llm/models_py/modules/dsv4/test:dsv4_kernel_perf_utils",
        "//rtp_llm/models_py:kernels",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    tags = ["manual", "H20", "SM100_ARM"],
)

py_test(
    name = "test_dsv4_indexed_rmsnorm_rope_perf",
    srcs = [
        "test_dsv4_indexed_rmsnorm_rope.py",
        "test_dsv4_indexed_rmsnorm_rope_perf.py",
    ],
    deps = [
        ":graphfx_fusion_test_utils",
        "//rtp_llm/models_py/modules/dsv4/test:dsv4_kernel_perf_utils",
        "//rtp_llm/models_py:kernels",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    tags = ["manual", "H20", "SM100_ARM"],
)

py_test(
    name = "test_dsv4_indexed_inv_rope_fp8_quant_perf",
    srcs = [
        "test_dsv4_indexed_inv_rope_fp8_quant.py",
        "test_dsv4_indexed_inv_rope_fp8_quant_perf.py",
    ],
    deps = [
        ":graphfx_fusion_test_utils",
        "//rtp_llm/models_py/modules/dsv4/test:dsv4_kernel_perf_utils",
        "//rtp_llm/models_py:kernels",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    tags = ["manual", "H20", "SM100_ARM"],
)

py_test(
    name = "test_kv_rope_fp8_quant_graphfx_perf",
    srcs = [
        "test_kv_rope_fp8_quant_graphfx.py",
        "test_kv_rope_fp8_quant_graphfx_perf.py",
    ],
    deps = [
        ":graphfx_fusion_test_utils",
        "//rtp_llm/models_py/modules/dsv4/test:dsv4_kernel_perf_utils",
        "//rtp_llm/models_py:kernels",
        "//rtp_llm/models_py:modules",
        "//rtp_llm:testlib",
    ],
    tags = ["manual", "H20", "SM100_ARM"],
)
