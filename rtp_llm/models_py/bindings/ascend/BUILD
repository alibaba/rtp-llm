load("//:def.bzl", "copts", "ascend_copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

cc_library(
    name = "ascend_types_hdr",
    hdrs = [
        "ascend_types_hdr.h",
    ],
    deps = select({
        "@//:using_ascend": [
            "@local_config_ascend//ascend:ascend_headers",
            "//rtp_llm/cpp/utils:core_utils",
        ],
        "//conditions:default": [],
    }),
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ascend_host_utils",
    srcs = [
        "ascend_host_utils.cc",
    ],
    hdrs = [
        "ascend_host_utils.h",
        "ascend_types_hdr.h",
    ],
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
    ] + select({
        "@//:using_ascend": [
            "@local_config_ascend//ascend:ascend_headers",
            "@local_config_ascend//ascend:ascend",
        ],
        "//conditions:default": [],
    }),
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ascend",
    deps = [
        ":ascend_host_utils",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ascend_bindings_register",
    srcs = [
        "AscendRegister.cc",
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/models_py/bindings:register_ops_hdr",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)
