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

py_library(
    name = "cuda_graph_util",
    deps = [],
    srcs = ["cuda_graph_util.py"],
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
    visibility = ["//visibility:public"],
)

py_test(
    name = "port_util_test",
    srcs = ["port_util_test.py"],
    deps = [
        ":port_util",
    ],
)
