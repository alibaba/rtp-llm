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
    name = "frontend_server_test",
    srcs = [
        "frontend_server_test.py",
    ],
    deps = [
        "//rtp_llm:testlib",
    ],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "token_processor_test",
    srcs = ["token_processor_test.py"],
    data = [
        "//rtp_llm/test/model_test/fake_test/testdata:testdata",
    ],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm:frontend",
    ],
    exec_properties = {'gpu':'A10'},
)

py_test(
    name = "request_id_generator_test",
    srcs = ["request_id_generator_test.py"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm:frontend",
    ],
)