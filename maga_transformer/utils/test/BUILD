
filegroup(
    name = "testdata",
    srcs = glob([
        "testdata/*",
        "testdata/**",
    ]),
)

py_test(
    name = "model_weights_test",
    srcs = [
        "model_weights_test.py",
    ],
    data = [
        "//maga_transformer/utils/test:testdata",
    ],
    deps = [
        "//maga_transformer:testlib",
        "//maga_transformer:config",
        "//maga_transformer:_ft_pickler",
    ],
)

py_test(
    name = "model_weights_loader_test",
    srcs = [
        "model_weights_loader_test.py",
    ],
    data = [
        "//maga_transformer/utils/test:testdata",
    ],
    deps = [
        "//maga_transformer:utils",
        "//maga_transformer:config",
        "//maga_transformer:models",
        "//maga_transformer:testlib",
        "//maga_transformer:_ft_pickler",
    ],
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