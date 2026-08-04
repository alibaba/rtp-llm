package(default_visibility = ["//visibility:public"])
load("@arch_config//:arch_select.bzl", "requirement")

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
        "coredump_util.py",
    ],
    deps = [],
)

py_library(
    name = "coredump_util",
    srcs = ["coredump_util.py"],
    visibility = ["//visibility:public"],
)

py_binary(
    name = "gpu_lock",
    deps = [
        ":torch",
        # Ensure AOT cache + cubin are present in runfiles for Package Copy /
        # flashinfer import on SM100 (otherwise JIT fmha_gen needs ninja/nvcc).
    ] + select({
        "@//:using_cuda12_9_x86": [
            "//rtp_llm:flashinfer-python",
            "//rtp_llm:flashinfer-jit-cache",
            "//rtp_llm:flashinfer-cubin",
        ],
        "@//:using_cuda12_arm": [
            "//rtp_llm:flashinfer-python",
            "//rtp_llm:flashinfer-jit-cache",
            "//rtp_llm:flashinfer-cubin",
        ],
        "//conditions:default": [],
    }),
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
