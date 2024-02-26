filegroup(
    name = "testdata",
    srcs = glob([
        "testdata/**/*",
    ]),
    visibility = ["//visibility:public"],
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

py_test(
    name = "qwen2_tokenizer_test",
    srcs = [
        "qwen2_tokenizer_test.py"
    ],
    data = [
        ":testdata"
    ],
    deps = [
        "//maga_transformer:testlib",
    ]
)
