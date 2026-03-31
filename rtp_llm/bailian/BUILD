# BUILD for rtp_llm/bailian: Python frontend gRPC (same pattern as model_rpc).
load("//bazel:arch_select.bzl", "requirement")

requirement(["grpcio"])

# Scripts and data to be packaged with the wheel (bailian gRPC client).
filegroup(
    name = "grpc_client_scripts",
    srcs = [
        "grpc_client_run.sh",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "bailian_grpc_server",
    srcs = [
        "__init__.py",
        "bailian_grpc_enqueue_loop.py",
        "bailian_grpc_request.py",
        "bailian_grpc_response_fake.py",
        "bailian_grpc_response_real.py",
        "bailian_grpc_real_infer.py",
        "bailian_grpc_service.py",
        "bailian_grpc_server.py",
        "bailian_grpc_client.py",
    ],
    data = [":grpc_client_scripts"],
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/server:server",
        ":grpcio",
        "//rtp_llm/bailian/proto:predict_v2_py_proto",
    ],
)

py_test(
    name = "bailian_grpc_request_test",
    srcs = ["test/bailian_grpc_request_test.py"],
    deps = ["//rtp_llm:testlib"],
    imports = [],
    exec_properties = {"gpu": "A10"},
)

py_test(
    name = "bailian_grpc_response_server_test",
    srcs = ["test/bailian_grpc_response_server_test.py"],
    deps = ["//rtp_llm:testlib"],
    imports = [],
    exec_properties = {"gpu": "A10"},
)
