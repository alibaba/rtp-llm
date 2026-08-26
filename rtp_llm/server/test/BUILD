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
    name = "kvcm_fallback_test",
    srcs = ["kvcm_fallback_test.py"],
    deps = [
        "//rtp_llm/server:server",
        "//rtp_llm:grpcio",
        "//rtp_llm:protobuf",
    ],
    tags = ["manual"],
)

py_test(
    name = "master_client_fallback_test",
    srcs = ["master_client_fallback_test.py"],
    deps = [
        "//rtp_llm/server:server",
        "//rtp_llm:aiohttp",
    ],
    tags = ["manual"],
)

py_test(
    name = "backend_rpc_server_visitor_fallback_test",
    srcs = ["backend_rpc_server_visitor_fallback_test.py"],
    deps = [
        "//rtp_llm/server:server",
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
