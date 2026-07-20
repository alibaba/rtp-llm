py_test(
    name = "codec_test",
    srcs = ["codec_test.py"],
    deps = ["//rtp_llm:testlib"],
    imports = [],
    exec_properties = {"gpu": "A10"},
)

py_test(
    name = "think_test",
    srcs = ["think_test.py"],
    main = "think_test.py",
    deps = ["//rtp_llm:testlib"],
    imports = [],
    exec_properties = {"gpu": "A10"},
)

py_test(
    name = "inference_servicer_test",
    srcs = ["inference/servicer_test.py"],
    main = "inference/servicer_test.py",
    deps = ["//rtp_llm:testlib"],
    imports = [],
    exec_properties = {"gpu": "A10"},
)

py_test(
    name = "proxy_servicer_test",
    srcs = ["proxy/servicer_test.py"],
    main = "proxy/servicer_test.py",
    deps = ["//rtp_llm:testlib"],
    imports = [],
    exec_properties = {"gpu": "A10"},
)

py_test(
    name = "access_log_test",
    srcs = ["access_log_test.py"],
    data = ["//rtp_llm/cpp/repetition:online_repetition_tracker"],
    deps = ["//rtp_llm:testlib"],
    imports = [],
    exec_properties = {"gpu": "A10"},
)

py_test(
    name = "repetition_monitor_test",
    srcs = ["repetition_monitor_test.py"],
    data = ["//rtp_llm/cpp/repetition:online_repetition_tracker"],
    deps = ["//rtp_llm:testlib"],
    imports = [],
    exec_properties = {"gpu": "A10"},
)

py_test(
    name = "app_test",
    srcs = ["app_test.py"],
    deps = ["//rtp_llm:testlib"],
    imports = [],
    exec_properties = {"gpu": "A10"},
)

py_test(
    name = "server_test",
    srcs = ["server_test.py"],
    deps = ["//rtp_llm:testlib"],
    imports = [],
    exec_properties = {"gpu": "A10"},
)
