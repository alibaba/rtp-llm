package(default_visibility = ["//visibility:public"])
load("//bazel:arch_select.bzl", "requirement")

requirement([
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
    ],
    deps = [],
)

py_binary(
    name = "gpu_lock",
    deps = [":torch"],
    main = "device_resource.py",
    srcs = [
        "device_resource.py",
        "jit_sys_path_setup.py",
    ],
)

py_library(
    name = "device_resource",
    deps = [],
    srcs = ["device_resource.py"],
)

py_test(
    name = "device_resource_test",
    srcs = ["device_resource_test.py"],
    deps = [":device_resource"],
    tags = ["SM103_CU13"],
    exec_properties = {"gpu": "SM103_CU13", "gpu_count": "1"},
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
        ":test_util",
    ],
)
