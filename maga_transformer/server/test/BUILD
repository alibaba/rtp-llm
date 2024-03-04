py_test(
    name = "inference_server_test",
    srcs = [
        "inference_server_test.py",
    ],
    deps = [
        "//maga_transformer:testlib",
    ]    
)
py_test (
    name = "inference_worker_test",
    srcs = ["inference_worker_test.py"],
    data = [
        "//maga_transformer/test/model_test/fake_test/testdata:testdata",
    ],
    deps = [
        "//maga_transformer/server:server",
        "//maga_transformer:_ft_pickler",
        "//maga_transformer:config",
        "//maga_transformer:testlib",
        "//maga_transformer/test/model_test/test_util:test_util"
    ],
)
