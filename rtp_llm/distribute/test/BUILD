filegroup(
    name = "testdata",
    srcs = glob([
        "testdata/*",
        "testdata/**",
        "testdata/tokenizer",
        "testdata/cpt",
    ]),
)

py_test(
    name = "gang_test",
    srcs = [
        "gang_test.py",
        "fake_model.py"
    ],
    data = [
        ":testdata",
        "//rtp_llm:sdk"
    ],
    deps = [
        "//rtp_llm:testlib",
    ],
    timeout = 'short',
    flaky = True,
    env = {"ASYNC_MODE":"0"},
    imports = ["aios/kmonitor/python_client"],
    exec_properties = {'gpu':'A10'},
)
