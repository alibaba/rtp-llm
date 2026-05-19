py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_library(
    name = "graphfx_fusion_test_utils",
    srcs = ["graphfx_fusion_test_utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm:torch",
    ],
)

py_test(
    name = "test_graphfx_injector",
    srcs = ["test_graphfx_injector.py"],
    deps = py_test_deps,
)

py_test(
    name = "test_add_rmsnorm_fp8_quant_pass",
    srcs = ["test_add_rmsnorm_fp8_quant_pass.py"],
    deps = py_test_deps + [
        ":graphfx_fusion_test_utils",
    ],
)

py_test(
    name = "test_add_rmsnorm_cross_graph",
    srcs = ["test_add_rmsnorm_cross_graph.py"],
    deps = py_test_deps + [
        ":graphfx_fusion_test_utils",
    ],
)

py_test(
    name = "test_rmsnorm_gated_fp8_quant_pass",
    srcs = ["test_rmsnorm_gated_fp8_quant_pass.py"],
    deps = py_test_deps + [
        ":graphfx_fusion_test_utils",
    ],
)

py_test(
    name = "test_silu_and_mul_fp8_quant_pass",
    srcs = ["test_silu_and_mul_fp8_quant_pass.py"],
    deps = py_test_deps + [
        ":graphfx_fusion_test_utils",
    ],
)

py_test(
    name = "test_sigmoid_mul_fp8_quant_pass",
    srcs = ["test_sigmoid_mul_fp8_quant_pass.py"],
    deps = py_test_deps + [
        ":graphfx_fusion_test_utils",
    ],
)
