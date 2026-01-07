
# Proto 文件定义
# 包含 model_rpc_service 和 all_embedding_rpc_service

load("//bazel:tf_proto.bzl", "tf_proto_library_cc")

# Model RPC Service Proto
tf_proto_library_cc(
    name = "model_rpc_service",
    srcs = ["proto/model_rpc_service.proto"],
    has_services = True,
    cc_grpc_version = True,
    default_header = True,
)

cc_library(
    name = "model_rpc_service_cc_proto",
    hdrs = ["proto/model_rpc_service.pb.h",
            "proto/model_rpc_service.grpc.pb.h"],
    srcs = ["proto/model_rpc_service.pb.cc",
            "proto/model_rpc_service.grpc.pb.cc"],
    deps = [":model_rpc_service_cc_impl"],
    visibility = ["//visibility:public"],
)
