load("//:def.bzl", "copts")

cc_library(
    name = "cp_transfer_projection",
    srcs = ["CpTransferProjection.cc"],
    hdrs = ["CpTransferProjection.h"],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = ["//rtp_llm/cpp/cache:cache_group_type"],
)
