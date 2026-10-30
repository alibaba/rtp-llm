
py_test(
    name = "xgrammar_loader_test",
    size = "small",
    srcs = ["xgrammar_loader_test.py"],
    deps = ["//rtp_llm:xgrammar"],
    target_compatible_with = select({
        "//:using_cuda12_9_x86": [],
        "//:using_cuda13_x86": [],
        "//:using_cuda13_arm": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

py_test(
    name = "grammar_compile_threads_test",
    srcs = [
        "grammar_compile_threads_test.py",
    ],
    data = [
        "//rtp_llm:sdk"
    ],
    deps = [
        "//rtp_llm:testlib",
    ],
)

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

