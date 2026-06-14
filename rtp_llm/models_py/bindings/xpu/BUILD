load("//:def.bzl", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

package(default_visibility=["//rtp_llm:__subpackages__"])

cc_library(
    name = "xpu_bindings_register",
    srcs = ["RegisterXpuOps.cc"],
    hdrs = [
        "RegisterXpuBaseBindings.hpp",
        "XpuTorchExt.h",
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/models_py/bindings:register_ops_hdr",
        "//rtp_llm/models_py/bindings/common:common",
    ] + torch_deps(),
    visibility = ["//visibility:public"],
    alwayslink = 1,
)
