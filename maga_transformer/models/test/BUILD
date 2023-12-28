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
    name = "chatglm_test",
    srcs = [
        "chatglm_test.py",
    ],
    data = [
        "//maga_transformer/models/test:testdata",
        "//maga_transformer/test/model_test/fake_test/testdata:testdata"
    ],
    deps = [
        "//maga_transformer:testlib",
    ],
)