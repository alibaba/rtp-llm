
test_envs = {
    "DEVICE_RESERVE_MEMORY_BYTES": "512000000",  # 512MB
}

py_test_deps = [
    "//rtp_llm/models_py/standalone:py_standalone_testlib",
]

py_test (
    name = "norm_test",
    srcs = ["norm_test.py"],
    deps = py_test_deps,
    env = test_envs,
    exec_properties = {'gpu':'A10'},
)

py_test (
    name = "rocm_norm_test",
    srcs = ["rocm_norm_test.py"],
    deps = py_test_deps,
    timeout = "eternal",
    env = test_envs,
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X'},
)

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
    env = test_envs,
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X'},
)

py_test (
    name = "rocm_layer_norm_test",
    srcs = ["rocm_layer_norm_test.py"],
    deps = py_test_deps,
    timeout = "eternal",
    env = test_envs,
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X'},
)

py_test (
    name = "rocm_fused_add_layernorm_test",
    srcs = ["rocm_fused_add_layernorm_test.py"],
    deps = py_test_deps,
    timeout = "eternal",
    env = test_envs,
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X'},
)

py_test (
    name = "rocm_general_layer_norm_test",
    srcs = ["rocm_general_layer_norm_test.py"],
    deps = py_test_deps,
    timeout = "eternal",
    env = test_envs,
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X'},
)

py_test (
    name = "rocm_embedding_test",
    srcs = ["rocm_embedding_test.py"],
    deps = py_test_deps,
    timeout = "eternal",
    env = test_envs,
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X'},
)

py_test (
    name = "rocm_fusedqkrmsnorm_test",
    srcs = ["rocm_fusedqkrmsnorm_test.py"],
    deps = py_test_deps,
    timeout = "eternal",
    env = test_envs,
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X'},
)

py_test (
    name = "rocm_fmha_test",
    srcs = ["rocm_fmha_test.py"],
    deps = py_test_deps,
    env = test_envs,
    timeout = "eternal",
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X'},
)

py_test (
    name = "rmsresnorm_test",
    srcs = ["rmsresnorm_test.py"],
    deps = py_test_deps,
    env = test_envs,
    exec_properties = {'gpu':'A10'},
)

#py_test (
#    name = "mlp_test",
#    srcs = ["mlp_test.py"],
#    deps = [
#        "//rtp_llm/models_py:models",
#        "//rtp_llm:config",
#        "//rtp_llm:utils",
#       "//rtp_llm:testlib",
#        "//rtp_llm/test/model_test/test_util:test_util"
#    ],
#    env = test_envs,
#    exec_properties = {'gpu':'A10'},
#)

py_test (
    name = "fusedqkrmsnorm_test",
    srcs = ["fusedqkrmsnorm_test.py"],
    deps = py_test_deps,
    env = test_envs,
    exec_properties = {'gpu':'A10'},
)

py_test (
    name = "layernorm_test",
    srcs = ["layernorm_test.py"],
    deps = py_test_deps,
    env = test_envs,
    exec_properties = {'gpu':'A10'},
)

py_test (
    name = "fused_add_layernorm_test",
    srcs = ["fused_add_layernorm_test.py"],
    deps = py_test_deps,
    env = test_envs,
    exec_properties = {'gpu':'A10'},
)

py_test (
    name = "embeding_test",
    srcs = ["embeding_test.py"],
    deps = py_test_deps,
    env = test_envs,
    exec_properties = {'gpu':'A10'},
)

py_test (
    name = "select_topk_op_test",
    srcs = ["select_topk_op_test.py"],
    deps = py_test_deps,
    env = test_envs,
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)

py_test (
    name = "fused_moe_op_test",
    srcs = ["fused_moe_op_test.py"],
    deps = py_test_deps,
    env = test_envs,
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)

py_test (
    name = "per_token_group_quant_8bit_test",
    srcs = ["per_token_group_quant_8bit_test.py"],
    deps = py_test_deps,
    env = test_envs,
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)

py_test (
    name = "fused_moe_test",
    srcs = ["fused_moe_test.py"],
    deps = py_test_deps,
    env = test_envs,
    exec_properties = {'gpu':'H20'},
)

py_test (
    name = "cutlass_fp8_grouped_gemm_test",
    srcs = ["cutlass_fp8_grouped_gemm_test.py"],
    deps = py_test_deps,
    env = test_envs,
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)

py_test (
    name = "per_tensor_scaled_fp8_quant_test",
    srcs = ["per_tensor_scaled_fp8_quant_test.py"],
    deps = py_test_deps,
    env = test_envs,
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)

py_test (
    name = "per_token_scaled_fp8_quant_test",
    srcs = ["per_token_scaled_fp8_quant_test.py"],
    deps = py_test_deps,
    env = test_envs,
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)

py_test (
    name = "moe_ep_reorder_test",
    srcs = ["moe_ep_reorder_test.py"],
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm:config",
        "//rtp_llm:utils",
        "//rtp_llm:testlib",
        "//rtp_llm/test/model_test/test_util:test_util"
    ],
    env = test_envs,
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)


py_test (
    name = "mla_attention_test",
    srcs = ["mla_attention_test.py"],
    deps = py_test_deps + [
        "//rtp_llm:testlib",
    ],
    env = test_envs,
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)

py_test (
    name = "mla_reuse_cache_test",
    srcs = ["mla_reuse_cache_test.py"],
    deps = py_test_deps + [
        "//rtp_llm:testlib",
    ],
    env = test_envs,
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)

py_test (
    name = "group_topk_test",
    srcs = ["group_topk_test.py"],
    deps = py_test_deps,
    env = test_envs,
    tags = ["open_skip", "H20"],
    exec_properties = {'gpu':'H20'},
)