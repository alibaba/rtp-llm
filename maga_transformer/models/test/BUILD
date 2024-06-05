filegroup(
    name = "testdata",
    srcs = glob([
        "testdata/*",
        "testdata/**",
    ]),
)

py_test(
    name = "qwen_test",
    srcs = [
        "qwen_test.py",
    ],
    data = [
        "//maga_transformer/models/test:testdata",
    ],
    deps = [
        "//maga_transformer:testlib",
    ],
)
py_test(
    name = "llama_test",
    srcs = [
        "llama_test.py",
    ],
    data = [
        "//maga_transformer/models/test:testdata",
    ],
    deps = [
        "//maga_transformer:testlib",
    ],
)

py_test(
    name = "starcode_test",
    srcs = [
        "starcode_test.py",
    ],
    data = [
        "//maga_transformer/models/test:testdata",
    ],
    deps = [
        "//maga_transformer:testlib",
    ],
)
py_test(
    name = "starcode2_test",
    srcs = [
        "starcode2_test.py",
    ],
    data = [
        "//maga_transformer/models/test:testdata",
    ],
    deps = [
        "//maga_transformer:testlib",
    ],
)