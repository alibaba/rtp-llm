load("//bazel:py_proto.bzl", "generate_grpc_proto")
load("//bazel:tf_proto.bzl", "tf_proto_library_cc")

package(default_visibility = ["//visibility:public"])

tf_proto_library_cc(
    name = "tensor_rdma",
    srcs = ["tensor_rdma.proto"],
    default_header = True,
    visibility = ["//visibility:public"],
)

# tf_proto_library_cc exposes the C++ generator as *_cc_genproto, while model_rpc's
# protodeps convention refers to the proto name without the _cc suffix.
alias(
    name = "tensor_rdma_genproto",
    actual = ":tensor_rdma_cc_genproto",
)

generate_grpc_proto(
    name = "tensor_rdma_py",
    proto = "tensor_rdma.proto",
    create_grpc_proto = "//rtp_llm/cpp/model_rpc/proto:create_grpc_proto",
)

py_library(
    name = "tensor_rdma_py_proto",
    srcs = [":tensor_rdma_py"],
)
