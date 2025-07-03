
test_envs = {
    "DEVICE_RESERVE_MEMORY_BYTES": "512000000",  # 512MB
}

py_test (
    name = "norm_test",
    srcs = ["norm_test.py"],
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
        "//rtp_llm/test/model_test/test_util:test_util"
    ],
    env = test_envs,
    exec_properties = {'gpu':'A10'},
)

py_test (
    name = "rocm_norm_test",
    srcs = ["rocm_norm_test.py"],
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
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
        "//rtp_llm/test/model_test/test_util:test_util"
    ],
    env = test_envs,
    tags = ["rocm"],
    exec_properties = {'gpu':'MI308X'},
)

py_test (
    name = "rmsresnorm_test",
    srcs = ["rmsresnorm_test.py"],
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
        "//rtp_llm/test/model_test/test_util:test_util"
    ],
    env = test_envs,
    exec_properties = {'gpu':'A10'},
)

py_test (
    name = "mlp_test",
    srcs = ["mlp_test.py"],
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm:config",
        "//rtp_llm:utils",
        "//rtp_llm:testlib",
        "//rtp_llm/test/model_test/test_util:test_util"
    ],
    env = test_envs,
    exec_properties = {'gpu':'A10'},
)

py_test (
    name = "fusedqkrmsnorm_test",
    srcs = ["fusedqkrmsnorm_test.py"],
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm:config",
        "//rtp_llm:utils",
        "//rtp_llm:testlib",
        "//rtp_llm/test/model_test/test_util:test_util"
    ],
    env = test_envs,
    exec_properties = {'gpu':'A10'},
)

py_test (
    name = "layernorm_test",
    srcs = ["layernorm_test.py"],
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm:config",
        "//rtp_llm:utils",
        "//rtp_llm:testlib",
        "//rtp_llm/test/model_test/test_util:test_util"
    ],
    env = test_envs,
    exec_properties = {'gpu':'A10'},
)

py_test (
    name = "fused_add_layernorm_test",
    srcs = ["fused_add_layernorm_test.py"],
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm:config",
        "//rtp_llm:utils",
        "//rtp_llm:testlib",
        "//rtp_llm/test/model_test/test_util:test_util"
    ],
    env = test_envs,
    exec_properties = {'gpu':'A10'},
)

py_test (
    name = "embeding_test",
    srcs = ["embeding_test.py"],
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm:config",
        "//rtp_llm:utils",
        "//rtp_llm:testlib",
        "//rtp_llm/test/model_test/test_util:test_util"
    ],
    env = test_envs,
    exec_properties = {'gpu':'A10'},
)

py_test (
    name = "select_topk_op_test",
    srcs = ["select_topk_op_test.py"],
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm:config",
        "//rtp_llm:utils",
        "//rtp_llm:testlib",
        "//rtp_llm/test/model_test/test_util:test_util"
    ],
    env = test_envs,
    exec_properties = {'gpu':'H20'},
)

py_test (
    name = "fused_moe_op_test",
    srcs = ["fused_moe_op_test.py"],
    deps = [
        "//rtp_llm/models_py:models",
        "//rtp_llm:config",
        "//rtp_llm:utils",
        "//rtp_llm:testlib",
        "//rtp_llm/test/model_test/test_util:test_util"
    ],
    env = test_envs,
    exec_properties = {'gpu':'H20'},
)