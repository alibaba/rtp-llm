load("//:def.bzl", "copts")

cc_library(
    name = "handler_args",
    hdrs = [
        "HandlerArgs.h",
    ],
    srcs = [
        "HandlerArgs.cc",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "post_layers_processor",
    hdrs = [
        "PostLayersProcessor.h",
    ],
    srcs = [
        "PostLayersProcessor.cc",
    ],
    copts = copts(),
    deps = [
        ":handler_args",
        "//:rtp_compute_ops",
    ],
    visibility = ["//visibility:public"],
)
