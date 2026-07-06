# BUILD for rtp_llm/dash_sc: Python frontend gRPC (same pattern as model_rpc).
load("//bazel:arch_select.bzl", "requirement")

requirement(["grpcio"])

# Scripts and data to be packaged with the wheel (dash_sc gRPC client).
filegroup(
    name = "grpc_client_scripts",
    srcs = [
        "grpc_client_run.sh",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "dash_sc_grpc_server",
    srcs = [
        "__init__.py",
        "access_log.py",
        "access_record.py",
        "app.py",
        "client.py",
        "codec.py",
        "grpc_metrics.py",
        "json_format_e2e_smoke.py",
        "inference/__init__.py",
        "inference/__main__.py",
        "inference/servicer.py",
        "proxy/__init__.py",
        "proxy/__main__.py",
        "proxy/service_route.py",
        "proxy/service_route_config.py",
        "proxy/servicer.py",
        "repetition_monitor.py",
        "server.py",
        "status.py",
        "structural_tag.py",
        "think.py",
    ],
    data = [
        ":grpc_client_scripts",
        "//rtp_llm/cpp/repetition:online_repetition_tracker",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/server:server",
        ":grpcio",
        "//rtp_llm/access_logger:access_logger",
        "//rtp_llm/dash_sc/proto:predict_v2_py_proto",
        "//rtp_llm/metrics:metrics",
        "//rtp_llm:utils",
    ],
)

py_binary(
    name = "json_format_e2e_smoke",
    srcs = ["json_format_e2e_smoke.py"],
    main = "json_format_e2e_smoke.py",
    visibility = ["//visibility:public"],
)
