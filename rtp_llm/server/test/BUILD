load("@arch_config//:arch_select.bzl", get_triton_deps = "triton_deps", "internal_deps")

py_test(
    name = "backend_rpc_server_visitor_test",
    srcs = ["backend_rpc_server_visitor_test.py"],
    deps = [
        "//rtp_llm:config_ops",
        "//rtp_llm:pydantic",
        "//rtp_llm:testlib",
        "//rtp_llm:torch",
        "//rtp_llm/cpp/model_rpc:model_rpc_client",
        "//rtp_llm/server:server",
    ],
)

py_test(
    name = "backend_manager_test",
    srcs = ["backend_manager_test.py"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/server:server",
    ],
    tags = ["manual"],
)

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
    name = "recent_cache_key_window_test",
    srcs = ["recent_cache_key_window_test.py"],
    deps = [
        "//rtp_llm/server:server",
    ],
    tags = ["manual"],
)

py_test(
    name = "backend_manager_shutdown_test",
    exec_properties = {"gpu": "A10"},
    srcs = ["backend_manager_shutdown_test.py"],
    deps = [
        "//rtp_llm:testlib",
        "//rtp_llm/models_py/distributed:deepep_wrapper",
        "//rtp_llm/server:server",
    ],
)

py_test(
    name = "start_backend_server_shutdown_test",
    srcs = ["start_backend_server_shutdown_test.py"],
    deps = [
        "//rtp_llm:sdk",
        "//rtp_llm:testlib",
    ],
)
