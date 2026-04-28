# BUILD for rtp_llm/dash_sc: Python frontend gRPC (same pattern as model_rpc).
load("@arch_config//:arch_select.bzl", "requirement")

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
        "dash_sc_app.py",
        "dash_sc_grpc_access_log.py",
        "dash_sc_grpc_enqueue_loop.py",
        "dash_sc_grpc_request.py",
        "dash_sc_grpc_response_fake.py",
        "dash_sc_grpc_response_real.py",
        "dash_sc_grpc_real_infer.py",
        "dash_sc_grpc_service.py",
        "dash_sc_grpc_pure_forward_service.py",
        "dash_sc_grpc_server.py",
        "dash_sc_grpc_client.py",
    ],
    data = [":grpc_client_scripts"],
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/server:server",
        ":grpcio",
        "//rtp_llm/access_logger:access_logger",
        "//rtp_llm/dash_sc/proto:predict_v2_py_proto",
    ],
)

py_test(
    name = "dash_sc_grpc_request_test",
    srcs = ["test/dash_sc_grpc_request_test.py"],
    deps = ["//rtp_llm:testlib"],
    imports = [],
    exec_properties = {"gpu": "A10"},
)

py_test(
    name = "dash_sc_grpc_response_server_test",
    srcs = ["test/dash_sc_grpc_response_server_test.py"],
    deps = ["//rtp_llm:testlib"],
    imports = [],
    exec_properties = {"gpu": "A10"},
)

py_test(
    name = "dash_sc_grpc_pure_forward_service_test",
    srcs = ["test/dash_sc_grpc_pure_forward_service_test.py"],
    deps = ["//rtp_llm:testlib"],
    imports = [],
    exec_properties = {"gpu": "A10"},
)

py_test(
    name = "dash_sc_grpc_access_log_test",
    srcs = ["test/dash_sc_grpc_access_log_test.py"],
    deps = ["//rtp_llm:testlib"],
    imports = [],
    exec_properties = {"gpu": "A10"},
)
