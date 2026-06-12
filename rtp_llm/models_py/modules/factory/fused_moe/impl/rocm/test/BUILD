
py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test (
    name = "rocm_fused_moe_test",
    srcs = ["rocm_fused_moe_test.py"],
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
        "//rtp_llm/test/model_test/test_util:test_util"
    ],
    timeout = "eternal",
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X-ROCM7'},
)

# TODO: fix this test
# py_test (
#     name = "deepep_normal_router_test",
#     srcs = ["deepep_normal_router_test.py"],
#     deps = py_test_deps,
#     timeout = "eternal",
#     env = test_envs,
#     tags = ["rocm"],
#     exec_properties = {'gpu':'MI308X-ROCM7'},
# )

py_test (
    name = "test_pure_tp_router",
    srcs = ["test_pure_tp_router.py"],
    deps = py_test_deps,
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X-ROCM7'},
)

py_test (
    name = "rocm_fp8_fused_moe_test",
    srcs = ["rocm_fp8_fused_moe_test.py"],
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
    ],
    timeout = "eternal",
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X-ROCM7'},
)

# Quark MXFP4 fused_moe numerical test (MI355 / gfx950 only).
# Not part of bulk MI308 ROCm CI: tag "rocm" is omitted; "manual" skips default
# bazel test //...; "rocm_gfx950" only matches jobs that opt into that filter.
# On MI308 this target hung when tagged "rocm" only — see PR #1005.
# Run on MI355: bazelisk test ...:rocm_mxfp4_fused_moe_test --test_tag_filters=rocm_gfx950
py_test (
    name = "rocm_mxfp4_fused_moe_test",
    srcs = ["rocm_mxfp4_fused_moe_test.py"],
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
    ],
    timeout = "eternal",
    tags = ["rocm_gfx950", "manual"],
    exec_properties = {'gpu':'MI355X-ROCM7'},
)

py_test (
    name = "rocm_moe_flydsl_gate_test",
    srcs = ["rocm_moe_flydsl_gate_test.py"],
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
    ],
    tags = ["rocm"],
)

# CPU-only regression test for torch_moe_ref (PR #882 review #4):
# the apply_router_weight_on_input=True path used to raise UnboundLocalError.
# Pure PyTorch CPU - intentionally no rocm tag / gpu exec_properties so it
# runs on the default CI runners.
py_test (
    name = "torch_moe_ref_test",
    srcs = ["torch_moe_ref_test.py"],
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
    ],
)

py_test (
    name = "moriep_intranode_router_test_2gpu",
    srcs = ["moriep_intranode_router_test.py"],
    main = "moriep_intranode_router_test.py",
    deps = py_test_deps + [
        "//rtp_llm/models_py:models",
    ],
    timeout = "eternal",
    env = {"MORIEP_TEST_MAX_GPUS": "2"},
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X-ROCM7', 'gpu_count':'2'},
)

py_test (
    name = "moriep_intranode_router_test_4gpu",
    srcs = ["moriep_intranode_router_test.py"],
    main = "moriep_intranode_router_test.py",
    deps = py_test_deps + [
        "//rtp_llm/models_py:models",
    ],
    timeout = "eternal",
    env = {"MORIEP_TEST_MAX_GPUS": "4"},
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X-ROCM7', 'gpu_count':'4'},
)
