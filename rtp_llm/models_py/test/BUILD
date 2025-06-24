py_test (
    name = "norm_test",
    srcs = ["norm_test.py"],
    deps = [
        "//rtp_llm:py_models",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
        "//rtp_llm/test/model_test/test_util:test_util"
    ],
    env = {
        "DEVICE_RESERVE_MEMORY_BYTES": "0"
    },
    exec_properties = {'gpu':'A10'},
)

py_test (
    name = "mlp_test",
    srcs = ["mlp_test.py"],
    deps = [
        "//rtp_llm:py_models",
        "//rtp_llm:config",
        "//rtp_llm:utils",
        "//rtp_llm:testlib",
        "//rtp_llm/test/model_test/test_util:test_util"
    ],
    env = {
        "DEVICE_RESERVE_MEMORY_BYTES": "0"
    },
    exec_properties = {'gpu':'A10'},
)

py_test (
    name = "fusedqkrmsnorm_test",
    srcs = ["fusedqkrmsnorm_test.py"],
    deps = [
        "//rtp_llm:py_models",
        "//rtp_llm:config",
        "//rtp_llm:utils",
        "//rtp_llm:testlib",
        "//rtp_llm/test/model_test/test_util:test_util"
    ],
    env = {
        "DEVICE_RESERVE_MEMORY_BYTES": "0"
    },
    exec_properties = {'gpu':'A10'},
)