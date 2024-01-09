filegroup(
    name = "testdata",
    srcs = glob([
        "testdata/**/*",
    ]),
)

py_test(
    name = "chatglm3_tokenizer_test",
    srcs = [
        "chatglm3_tokenizer_test.py"
    ],
    data = [
        ":testdata"
    ],
    deps = [
        "//maga_transformer:testlib",
    ]
)
