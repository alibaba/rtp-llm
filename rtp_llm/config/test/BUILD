
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


# The expert_dtype allow-list entry and its parser. Host-only: no GPU, and the
# import chain is ModelConfig plus one pure function.
py_test(
    name = "expert_dtype_test",
    srcs = [
        "expert_dtype_test.py",
    ],
    deps = [
        "//rtp_llm:testlib",
    ],
)
