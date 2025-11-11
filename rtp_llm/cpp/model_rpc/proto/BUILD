load("//bazel:tf_proto.bzl", "tf_proto_library_cc")
load("//bazel:py_proto.bzl", "generate_grpc_proto")

tf_proto_library_cc(
    name = "model_rpc_service",
    srcs = ["model_rpc_service.proto"],
    has_services = True,
    cc_grpc_version = True,
    default_header = True,
    # static = False,
)

py_binary(
    name = "create_grpc_proto",
    srcs = ["create_grpc_proto.py"],
    data = ["model_rpc_service.proto"],
    deps = ["//rtp_llm:grpcio-tools", "//rtp_llm:setuptools"],
    visibility = ["//visibility:public"],
)

generate_grpc_proto(
    name = "model_rpc_service_py",
    proto = "model_rpc_service.proto",
    create_grpc_proto = ":create_grpc_proto"
)


cc_library(
    name = "model_rpc_service_cc_proto",
    hdrs = ["model_rpc_service.pb.h",
            "model_rpc_service.grpc.pb.h"],
    srcs = ["model_rpc_service.pb.cc",
            "model_rpc_service.grpc.pb.cc"],
    deps = [":model_rpc_service_cc_impl"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "model_rpc_service_py_proto",
    srcs = [":model_rpc_service_py"],
    visibility = ["//visibility:public"],
)
