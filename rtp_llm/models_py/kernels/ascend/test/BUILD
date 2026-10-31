load("@arch_config//:arch_select.bzl", "requirement")

requirement(["torch"])

_DEPS = [
    ":torch",
    "//rtp_llm/models_py/kernels/ascend:qwen3_next_ops",
]

py_test(
    name = "block_test",
    srcs = ["test_block.py"],
    deps = _DEPS,
)

py_test(
    name = "backend_selection_test",
    srcs = ["test_backend_selection.py"],
    data = ["//rtp_llm/models_py:qwen3_next_source"],
    deps = _DEPS,
)

py_test(
    name = "causal_conv1d_test",
    srcs = ["test_causal_conv1d.py"],
    deps = _DEPS,
)

py_test(
    name = "common_test",
    srcs = ["test_common.py"],
    deps = _DEPS,
)

py_test(
    name = "linear_attention_test",
    srcs = ["test_linear_attention.py"],
    deps = _DEPS,
)

py_test(
    name = "plan_c_device_path_test",
    srcs = ["test_plan_c_device_path.py"],
    deps = _DEPS,
)

py_test(
    name = "recurrent_test",
    srcs = ["test_recurrent.py"],
    deps = _DEPS,
)

py_test(
    name = "state_migration_test",
    srcs = ["test_state_migration.py"],
    deps = _DEPS,
)
