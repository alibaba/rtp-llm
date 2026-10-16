py_test(
    name = "int8_allreduce_fp16_scale_config_test",
    srcs = ["server_args_test.py"],
    main = "server_args_test.py",
    args = ["Int8AllReduceFp16ScaleConfigTest"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/server:server",
        "//rtp_llm:config_ops",
        "//rtp_llm:utils",
    ],
    tags = ["RTX_5000_PRO", "RTX_5000_PRO_CU13"],
    exec_properties = {"gpu": "RTX_5000_PRO_CU13"},
)

py_test(
    name = "server_args_test",
    srcs = ["server_args_test.py"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/server:server",
        "//rtp_llm:config_ops",
        "//rtp_llm:utils",
    ],
    exec_properties = {'gpu':'H20'},
)
