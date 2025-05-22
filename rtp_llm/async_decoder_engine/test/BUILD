py_test (
    name = "rpc_model_test",
    srcs = ["rpc_model_test.py"],
    data = [
        "//rtp_llm/test/model_test/fake_test/testdata:testdata",
    ],
    deps = [
        "//rtp_llm:models",
        "//rtp_llm:_ft_pickler",
        "//rtp_llm:config",
        "//rtp_llm:testlib",
        "//rtp_llm/test/model_test/test_util:test_util"
    ],
    env = {
        "DEVICE_RESERVE_MEMORY_BYTES": "0"
    },
    exec_properties = {'gpu':'A10'},
)