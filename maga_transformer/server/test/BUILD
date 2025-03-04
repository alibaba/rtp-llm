py_test(
    name = "frontend_server_test",
    srcs = [
        "frontend_server_test.py",
    ],
    deps = [
        "//maga_transformer:testlib",
    ],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "frontend_worker_test",
    srcs = ["frontend_worker_test.py"],
    data = [
        "//maga_transformer/test/model_test/fake_test/testdata:testdata",
    ],
    deps = [
        "//maga_transformer/server:server",
        "//maga_transformer:_ft_pickler",
        "//maga_transformer:config",
        "//maga_transformer:testlib",
        "//maga_transformer/test/model_test/test_util:test_util",
        "//maga_transformer/test/utils:port_util",
    ],
    exec_properties = {'gpu':'A10'},
)
