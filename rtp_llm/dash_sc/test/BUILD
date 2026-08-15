# These tests import rtp_llm.dash_sc.{codec,think,client,...}; those modules belong
# to //rtp_llm/dash_sc:dash_sc_grpc_server and are not in //rtp_llm:testlib's
# closure. Depending on testlib alone leaves these files out of runfiles
# ("No module named rtp_llm.dash_sc.<mod>"). All eight targets share one deps list.
dash_sc_test_deps = [
    "//rtp_llm:testlib",
    "//rtp_llm/dash_sc:dash_sc_grpc_server",
]

py_test(
    name = "codec_test",
    srcs = ["codec_test.py"],
    deps = dash_sc_test_deps,
    imports = [],
    exec_properties = {"gpu": "A10"},
)

py_test(
    name = "grammar_validator_test",
    srcs = ["inference/grammar_validator_test.py"],
    main = "inference/grammar_validator_test.py",
    deps = ["//rtp_llm:testlib"],
    imports = [],
    exec_properties = {"gpu": "A10"},
)

py_test(
    name = "think_test",
    srcs = ["think_test.py"],
    main = "think_test.py",
    deps = dash_sc_test_deps,
    imports = [],
    exec_properties = {"gpu": "A10"},
)

py_test(
    name = "inference_servicer_test",
    srcs = ["inference/servicer_test.py"],
    main = "inference/servicer_test.py",
    deps = dash_sc_test_deps,
    imports = [],
    exec_properties = {"gpu": "A10"},
)

py_test(
    name = "mrcr_smoke_test",
    srcs = ["inference/mrcr_smoke_test.py"],
    data = ["data/mrcr_deepseek_v4_think_leak_cases.json"],
    main = "inference/mrcr_smoke_test.py",
    deps = dash_sc_test_deps,
    imports = [],
    exec_properties = {"gpu": "A10"},
)

py_test(
    name = "proxy_servicer_test",
    srcs = ["proxy/servicer_test.py"],
    main = "proxy/servicer_test.py",
    deps = dash_sc_test_deps,
    imports = [],
    exec_properties = {"gpu": "A10"},
)

py_test(
    name = "access_log_test",
    srcs = ["access_log_test.py"],
    data = ["//rtp_llm/cpp/repetition:online_repetition_tracker"],
    deps = dash_sc_test_deps,
    imports = [],
    exec_properties = {"gpu": "A10"},
)

py_test(
    name = "repetition_monitor_test",
    srcs = ["repetition_monitor_test.py"],
    data = ["//rtp_llm/cpp/repetition:online_repetition_tracker"],
    deps = dash_sc_test_deps,
    imports = [],
    exec_properties = {"gpu": "A10"},
)

py_test(
    name = "app_test",
    srcs = ["app_test.py"],
    deps = dash_sc_test_deps,
    imports = [],
    exec_properties = {"gpu": "A10"},
)
