package(default_visibility = ["//visibility:public"])
load("@arch_config//:arch_select.bzl", "requirement")

requirement([
    "filelock",
    "torch",
])

py_library(
    name = "test_util",
    srcs = [
        "bench_util.py",
        "numeric_util.py",
        "cuda_graph_util.py",
        "port_util.py",
        "stream_util.py",
        "diff_util.py",
        "coredump_util.py",
    ],
    deps = [],
)

py_library(
    name = "coredump_util",
    srcs = ["coredump_util.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "jit_sys_path_setup",
    srcs = ["jit_sys_path_setup.py"],
    deps = [":filelock"],
)

py_binary(
    name = "gpu_lock",
    deps = [":jit_sys_path_setup", ":filelock", ":torch"],
    main = "device_resource.py",
    srcs = ["device_resource.py"],
)

py_library(
    name = "device_resource",
    deps = [":filelock"],
    srcs = ["device_resource.py"],
)

py_library(
    name = "maga_server_manager",
    srcs = ["maga_server_manager.py"],
    deps = [
        "//rtp_llm:port_config",
        "//rtp_llm:psutil",
        "//rtp_llm:requests",
        ":test_util",
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "port_util_test",
    srcs = ["port_util_test.py"],
    deps = [
        ":test_util",
    ],
)

py_test(
    name = "numeric_util_test",
    srcs = ["numeric_util_test.py"],
    deps = [
        ":test_util",
        ":torch",
    ],
)

py_test(
    name = "maga_server_manager_test",
    srcs = ["maga_server_manager_test.py"],
    deps = [":maga_server_manager"],
)
