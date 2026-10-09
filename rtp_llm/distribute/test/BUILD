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
    name = "distributed_server_test",
    srcs = [
        "distributed_server_test.py",
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
    imports = ["aios/kmonitor/python_client"],
    exec_properties = {'gpu':'H20','gpu_count':'2'},
    env = {"GPU_COUNT": "2"},
)

py_test(
    name = "frontend_routing_test",
    srcs = ["frontend_routing_test.py"],
    data = ["//:th_transformer_config"],
    deps = [
        "//rtp_llm:sdk",
        "//rtp_llm:config_ops",
        "//rtp_llm:utils",
        "//rtp_llm/distribute:distribute",
    ],
)
