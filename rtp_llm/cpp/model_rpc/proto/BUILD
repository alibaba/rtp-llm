load("@arch_config//:arch_select.bzl", "requirement")
load("//bazel:tf_proto.bzl", "tf_proto_library_cc")
load("//bazel:py_proto.bzl", "generate_grpc_proto")

exports_files(
    ["flexlb_schedule_service.proto", "model_rpc_service.proto"],
    visibility = ["//rtp_llm/flexlb:__pkg__"],
)

requirement(["grpcio"])

# The generator is an exec tool; its wheels are not present in client runfiles.
# Both generated *_pb2.py and *_pb2_grpc.py need their runtime imports declared.
_PY_GRPC_RUNTIME_DEPS = [":grpcio", "//rtp_llm:protobuf"]

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
    data = [
        "flexlb_schedule_service.proto",
        "model_rpc_service.proto",
    ],
    deps = ["//rtp_llm:grpcio-tools", "//rtp_llm:setuptools"],
    visibility = ["//visibility:public"],
)

generate_grpc_proto(
    name = "model_rpc_service_py",
    proto = "model_rpc_service.proto",
    create_grpc_proto = ":create_grpc_proto"
)

cc_library(
    name = "grpc_buffer_list_link_fix",
    srcs = ["@grpc//:src/core/lib/iomgr/buffer_list.cc"],
    alwayslink = True,
    copts = ["-DGRPC_LINUX_ERRQUEUE=1"],
    deps = ["@grpc//:grpc_base_c"],
)

cc_library(
    name = "model_rpc_service_cc_proto",
    hdrs = ["model_rpc_service.pb.h",
            "model_rpc_service.grpc.pb.h"],
    srcs = ["model_rpc_service.pb.cc",
            "model_rpc_service.grpc.pb.cc"],
    deps = [
        ":grpc_buffer_list_link_fix",
        ":model_rpc_service_cc_impl",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "model_rpc_service_py_proto",
    srcs = [":model_rpc_service_py"],
    visibility = ["//visibility:public"],
    deps = _PY_GRPC_RUNTIME_DEPS,
)

generate_grpc_proto(
    name = "flexlb_schedule_service_py",
    proto = "flexlb_schedule_service.proto",
    create_grpc_proto = ":create_grpc_proto",
)

py_library(
    name = "flexlb_schedule_service_py_proto",
    srcs = [":flexlb_schedule_service_py"],
    visibility = ["//visibility:public"],
    deps = _PY_GRPC_RUNTIME_DEPS,
)
