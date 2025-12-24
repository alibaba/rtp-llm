package(default_visibility = ["//visibility:public"])
load("//bazel:arch_select.bzl", "requirement")

requirement([
    "torch",
])

py_library(
    name = "bench_util",
    deps = [],
    srcs = ["bench_util.py"],
)

py_library(
    name = "numeric_util",
    deps = [],
    srcs = ["numeric_util.py"],
)

py_binary(
    name = "gpu_lock",
    deps = [":torch"],
    main = "device_resource.py",
    srcs = ["device_resource.py"],
)

py_library(
    name = "device_resource",
    deps = [],
    srcs = ["device_resource.py"],
)

py_library(
    name = "port_util",
    deps = [],
    srcs = ["port_util.py"],
)

py_library(
    name = "stream_util",
    srcs = ["stream_util.py"],
    deps = [
        "//rtp_llm:openai_api",
    ],
)

py_library(
    name = "maga_server_manager",
    srcs = ["maga_server_manager.py"],
    deps = [":base_server_manager"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "base_server_manager",
    srcs = ["base_server_manager.py"],
    deps = [":port_util"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "flexlb_server_manager",
    srcs = ["flexlb_server_manager.py"],
    deps = [
        ":base_server_manager",
        "//rtp_llm:requests",
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "port_util_test",
    srcs = ["port_util_test.py"],
    deps = [
        ":port_util",
    ],
)
