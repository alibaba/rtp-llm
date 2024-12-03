load("//bazel:tf_proto.bzl", "tf_proto_library")

tf_proto_library(
    name = "model_rpc_service",
    srcs = ["model_rpc_service.proto"],
    has_services = True,
    cc_grpc_version = True,
    default_header = True,
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
    srcs = ["model_rpc_service_pb2.py",
            "model_rpc_service_pb2_grpc.py"],
    deps = [":model_rpc_service_py"],
    visibility = ["//visibility:public"],
)
