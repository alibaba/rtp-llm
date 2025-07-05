
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
        "//rtp_llm/utils/test:testdata",
        "//rtp_llm/test/model_test/fake_test/testdata:testdata",
    ],
    deps = [
        "//rtp_llm:utils",
        "//rtp_llm:config",
        "//rtp_llm:models",
        "//rtp_llm:testlib",
        "//rtp_llm:_ft_pickler",
    ],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "ckpt_database_test",
    srcs = [
        "ckpt_database_test.py",
    ],
    deps = [
        "//rtp_llm:utils",
        "//rtp_llm:config",
        "//rtp_llm:_ft_pickler",
        "//rtp_llm:lora"
    ],
    data = [
        "//rtp_llm/utils/test:testdata",
    ]
)

py_test(
    name = "fuser_test",
    srcs = [
        "fuser_test.py",
    ],
    deps = [
        "//rtp_llm:utils",
    ],
)


py_test(
    name = "util_test",
    srcs = [
        "util_test.py",
    ],
    deps = [
        "//rtp_llm:utils",
        "//rtp_llm:_ft_pickler",
        "//rtp_llm:pynvml",
    ],
)

py_test(
    name = "multimodal_util_test",
    srcs = [
        "multimodal_util_test.py",
    ],
    deps = [
        "//rtp_llm:utils",
        "//rtp_llm:_ft_pickler",
        "//rtp_llm:pynvml",
        "//rtp_llm:Pillow",
        "//rtp_llm:oss2",
        "//rtp_llm:pyOpenSSL",
        "//rtp_llm:config_ops"
    ],
    data = [
        "//:th_transformer"
    ]
)
