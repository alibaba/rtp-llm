load("//:def.bzl", "copts")
load("@arch_config//:arch_select.bzl", "rdma_transport_deps")

package(default_visibility = ["//visibility:public"])

rdma_transport_deps()

cc_library(
    name = "rdma_transport",
    srcs = ["RdmaTransport.cc"],
    hdrs = ["RdmaTransport.h"],
    deps = [
        "//:rtp_compute_ops",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/rdma_transport/proto:tensor_rdma_cc",
        "//rtp_llm/cpp/utils:error_code",
    ],
    copts = copts(),
)

cc_library(
    name = "rdma_transport_no_impl",
    srcs = ["RdmaTransportNoImpl.cc"],
    deps = [":rdma_transport", "//rtp_llm/cpp/utils:core_utils"],
    copts = copts(),
)
