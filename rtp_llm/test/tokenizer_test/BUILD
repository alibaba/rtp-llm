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
        "//rtp_llm:testlib",
    ],
    exec_properties = {'gpu':'A10'},
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
        "//rtp_llm:testlib",
    ]
)
