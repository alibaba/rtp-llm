package(default_visibility = ["//visibility:public"])

py_test(
    name = "advertise_ip_test",
    srcs = ["advertise_ip_test.py"],
    deps = ["//rtp_llm:utils"],
)


py_test(
    name = "endpoint_config_test",
    srcs = ["endpoint_config_test.py"],
    data = ["//:th_transformer_config"],
    deps = [
        "//rtp_llm:config_ops",
        "//rtp_llm:sdk",
        "//rtp_llm:structure",
        "//rtp_llm:utils",
        "//rtp_llm/access_logger:access_logger",
        "//rtp_llm/metrics:metrics",
        "//rtp_llm/models_py/distributed:collective_torch",
        "//rtp_llm/distribute:distribute",
        "//rtp_llm/server:server",
        "//rtp_llm:device",
    ],
)


py_test(
    name = "endpoint_provider_test",
    srcs = ["endpoint_provider_test.py"],
    data = ["//:th_transformer_config"],
    deps = [
        "//rtp_llm:config_ops",
        "//rtp_llm:utils",
        "//rtp_llm/distribute:distribute",
    ],
)


py_test(
    name = "kmonitor_lifecycle_test",
    srcs = ["kmonitor_lifecycle_test.py"],
    deps = ["//rtp_llm:utils", "//rtp_llm/aios/kmonitor:kmonitor_py"],
)


py_test(
    name = "local_comm_test",
    srcs = ["local_comm_test.py"],
    deps = ["//rtp_llm:utils"],
)


py_test(
    name = "native_logger_test",
    srcs = ["native_logger_test.py"],
    data = ["//rtp_llm/cpp/utils:scr_logger_test_sources"],
)


py_test(
    name = "pd_advertisement_test",
    srcs = ["pd_advertisement_test.py"],
    data = ["//:th_transformer_config"],
    deps = [
        "//rtp_llm:config",
        "//rtp_llm:config_ops",
        "//rtp_llm:utils",
        "//rtp_llm/distribute:distribute",
    ],
)


py_test(
    name = "restore_context_test",
    srcs = ["restore_context_test.py"],
    data = ["//:th_transformer_config"],
    deps = [
        "//rtp_llm:config_ops",
        "//rtp_llm:utils",
        "//rtp_llm/distribute:distribute",
    ],
)


py_test(
    name = "restore_env_file_test",
    srcs = ["restore_env_file_test.py"],
    deps = ["//rtp_llm:utils"],
)


py_test(
    name = "runtime_fixup_test",
    srcs = ["runtime_fixup_test.py"],
    deps = ["//rtp_llm:utils", "//rtp_llm/aios/kmonitor:kmonitor_py"],
)


py_test(
    name = "service_startup_test",
    srcs = ["service_startup_test.py"],
    data = ["//:th_transformer_config"],
    deps = [
        "//rtp_llm:config_ops",
        "//rtp_llm:sdk",
        "//rtp_llm:structure",
        "//rtp_llm:utils",
        "//rtp_llm/access_logger:access_logger",
        "//rtp_llm/metrics:metrics",
        "//rtp_llm/models_py/distributed:collective_torch",
        "//rtp_llm/distribute:distribute",
        "//rtp_llm/server:server",
    ],
)


py_test(
    name = "lifecycle_test",
    srcs = ["lifecycle_test.py"],
    deps = ["//rtp_llm:utils"],
)


py_test(
    name = "template_utils_test",
    srcs = [
        "template_utils_test.py",
    ],
    deps = [
        "//rtp_llm:utils",
    ],
)


py_test(
    name = "transport_boundary_test",
    srcs = ["transport_boundary_test.py"],
    data = ["//:th_transformer_config"],
    deps = ["//rtp_llm:config", "//rtp_llm:config_ops", "//rtp_llm/distribute:distribute"],
)


py_test(
    name = "vip_integration_test",
    srcs = ["vip_integration_test.py"],
    data = ["//:th_transformer_config"],
    deps = [
        "//rtp_llm:sdk",
        "//rtp_llm:config_ops",
        "//rtp_llm:utils",
        "//rtp_llm/distribute:distribute",
        "//rtp_llm/models_py/distributed:collective_torch",
    ],
)


py_test(
    name = "vip_test",
    srcs = ["vip_test.py"],
    data = ["//:th_transformer_config"],
    deps = [
        "//rtp_llm:sdk",
        "//rtp_llm:config_ops",
        "//rtp_llm:structure",
        "//rtp_llm:utils",
        "//rtp_llm:device",
        "//rtp_llm/distribute:distribute",
        "//rtp_llm/access_logger:access_logger",
        "//rtp_llm/metrics:metrics",
        "//rtp_llm/models_py/distributed:collective_torch",
    ],
)


py_test(
    name = "arrival_test",
    srcs = ["arrival_test.py"],
    deps = ["//rtp_llm:sdk"],
)


py_test(
    name = "scheduler_e2e_test",
    srcs = ["scheduler_e2e_test.py"],
    deps = ["//rtp_llm:sdk"],
)


py_test(
    name = "backend_integration_test",
    srcs = ["backend_integration_test.py"],
    data = [
        "//:th_transformer_config",
    ],
    deps = ["//rtp_llm:sdk"],
)

