
filegroup(
    name = "testdata",
    srcs = glob([
        "testdata/*",
        "testdata/**",
    ]),
)

py_test(
    name = "incremental_decode_test",
    srcs = [
        "incremental_decode_test.py",
    ],
    data = [
        "//maga_transformer/utils/test:testdata",
        "//maga_transformer/test/model_test/fake_test/testdata:testdata",
    ],
    deps = [
        "//maga_transformer:utils",
        "//maga_transformer:config",
        "//maga_transformer:models",
        "//maga_transformer:testlib",
        "//maga_transformer:_ft_pickler",
    ],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "ckpt_database_test",
    srcs = [
        "ckpt_database_test.py",
    ],
    deps = [
        "//maga_transformer:utils",
        "//maga_transformer:config",
        "//maga_transformer:_ft_pickler",
        "//maga_transformer:lora"
    ],
    data = [
        "//maga_transformer/utils/test:testdata",
    ]
)

py_test(
    name = "fuser_test",
    srcs = [
        "fuser_test.py",
    ],
    deps = [
        "//maga_transformer:utils",
    ],
)


py_test(
    name = "util_test",
    srcs = [
        "util_test.py",
    ],
    deps = [
        "//maga_transformer:utils",
        "//maga_transformer:_ft_pickler",
        "//maga_transformer:pynvml",
    ],
)
