py_test (
    name = "rpc_model_test",
    srcs = ["rpc_model_test.py"],
    data = [
        "//maga_transformer/test/model_test/fake_test/testdata:testdata",
    ],
    deps = [
        "//maga_transformer:models",
        "//maga_transformer:_ft_pickler",
        "//maga_transformer:config",
        "//maga_transformer:testlib",
        "//maga_transformer/test/model_test/test_util:test_util"
    ],
    env = {
        "DEVICE_RESERVE_MEMORY_BYTES": "0"
    },
    exec_properties = {'gpu':'A10'},
)