load("@arch_config//:arch_select.bzl", get_triton_deps = "triton_deps", "internal_deps")

py_test(
    name = "schedule_meta_test",
    srcs = ["schedule_meta_test.py"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/server:server",
        "//rtp_llm:config_ops",
        "//rtp_llm:pydantic",
    ],
    tags = ["manual"],
)

py_test(
    name = "host_service_test",
    srcs = ["host_service_test.py"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/server:server",
        "//rtp_llm:config_ops",
        "//rtp_llm:pydantic",
    ],
    tags = ["manual"],
)

py_test(
    name = "vit_proxy_server_test",
    srcs = ["vit_proxy_server_test.py"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/server:server",
    ],
)

py_test(
    name = "sleep_mode_compatibility_test",
    srcs = ["sleep_mode_compatibility_test.py"],
    deps = [
        "//rtp_llm:config_ops",
        "//rtp_llm:pydantic",
        "//rtp_llm:testlib",
        "//rtp_llm/server:server",
    ],
    tags = ["manual"],
)
