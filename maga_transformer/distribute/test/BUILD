filegroup(
    name = "testdata",
    srcs = glob([
        "testdata/*",
        "testdata/**",
        "testdata/tokenizer",
        "testdata/cpt",
    ]),
)

py_test(
    name = "gang_test",
    srcs = [
        "gang_test.py",
        "fake_model.py"
    ],
    data = [
        ":testdata",
        "//maga_transformer:sdk"
    ],
    deps = [
        "//maga_transformer:testlib",
    ],
    timeout = 'short',
    env = {"ASYNC_MODE":"0"},
    imports = ["aios/kmonitor/python_client"],
)
