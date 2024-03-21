load("//:def.bzl", "copts", "cuda_copts", "torch_deps")

cc_library(
    name = "common",
    hdrs = glob([
        "common/*.h",
    ]),
)

cc_library(
    name = "dataclass",
    hdrs = glob([
        "dataclass/*.h",
    ]),
    srcs = glob([
        "dataclass/*.cc",
    ]),
    deps = torch_deps() + [
        ":common",
        "//:gpt_init_params"
    ],
)

cc_library(
    name = "utils",
    hdrs = glob([
        "utils/*.h",
    ]),
    srcs = glob([
        "utils/*.cc",
    ]),
    deps = torch_deps() + [
        ":dataclass",
        "//:gpt_init_params",
        "//src/fastertransformer/devices:devices_base",
        "//src/fastertransformer/devices:device_utils",
        "//src/fastertransformer/models:weights_define",
    ],
)

cc_library(
    name = "models",
    hdrs = glob([
        "models/*.h",
    ]),
    srcs = glob([
        "models/*.cc",
    ]),
    deps = torch_deps() + [
        ":dataclass",
        ":utils",
    ],
)

cc_library(
    name = "components",
    hdrs = glob([
        "components/*.h",
    ]),
    srcs = glob([
        "components/*.cc",
    ]),
    deps = torch_deps() + [
        ":dataclass",
        ":models",
    ],
)

cc_library(
    name = "maga_op_lib",
    hdrs = [
        "MagaOp.h",
    ],
    srcs = [
        "MagaOp.cc",
    ],
    deps = torch_deps() + [
        ":components",
        "//src/fastertransformer/devices:devices_base",
        "//src/fastertransformer/devices/cpu_impl:cpu_impl",
    ] + select({
        "//:using_cuda": ["//src/fastertransformer/devices/cuda_impl:cuda_impl"],
        "//conditions:default": [],
    }),
    copts = copts(),
    alwayslink = 1,
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "maga_op",
    deps = [
        ":maga_op_lib",
    ],
    linkshared = 1,
    visibility = ["//visibility:public"],
)
