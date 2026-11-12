load("//rtp_llm/models_py/test:defs.bzl", "models_py_test")

package(default_visibility = ["//visibility:public"])

models_py_test(
    name = "ple_gather_triton_test",
    src = "ple_gather_triton_test.py",
    gpu = "L20_CU13",
)

models_py_test(
    name = "ple_outer_residual_test",
    src = "ple_outer_residual_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = ["//rtp_llm/models_py/modules/qwen4_exp/test:test_support"],
)

models_py_test(
    name = "ple_hash_triton_test",
    src = "ple_hash_triton_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = ["//rtp_llm/models_py/modules/qwen4_exp/test:math_references"],
)

models_py_test(
    name = "ple_read_blocks_test",
    src = "ple_read_blocks_test.py",
    gpu = "RTX_5000_PRO_CU13",
)

models_py_test(
    name = "gate_test",
    src = "gate_test.py",
    gpu = "RTX_5000_PRO_CU13",
    deps = ["//rtp_llm/models_py/modules/qwen4_exp/test:test_support"],
)
