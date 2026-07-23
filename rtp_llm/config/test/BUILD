
py_test(
    name = "server_config_setup_test",
    srcs = [
        "server_config_setup_test.py",
    ],
    data = [
        "//rtp_llm/test/model_test/fake_test/testdata:testdata",
        "//rtp_llm:sdk"
    ],
    deps = [
        "//rtp_llm:testlib",
    ],
    exec_properties = {'gpu':'H20'},
)

py_test(
    name = "deepseek_config_dtype_test",
    srcs = ["deepseek_config_dtype_test.py"],
    deps = [
        "//rtp_llm:testlib",
    ],
)
