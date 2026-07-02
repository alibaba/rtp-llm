load("//:def.bzl", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

cc_library(
    name = "torch_beam_search_op_impl",
    hdrs = ["BeamSearchOp.h"],
    deps = torch_deps(),
    visibility = ["//visibility:public"],
    copts = copts(),
)
