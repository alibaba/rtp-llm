load("//:def.bzl", "copts", "cuda_copts", "gen_cpp_code")
load("//bazel:arch_select.bzl", "torch_deps")
load("@rules_python//python:defs.bzl", "py_library") 
load("//bazel:arch_select.bzl", "torch_deps", "select_py_bindings")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts_without_arch")

cc_library(
    name = "tipc_core_lib",
    hdrs = [
        "IpcOp.h",
    ],
    srcs = [
        "IpcOp.cc",
    ],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps(),
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "_tipc_lib.so",
    srcs = [
        "binding.cc",
    ],
    deps = [
        ":tipc_core_lib",
    ] + torch_deps() + select_py_bindings(),
    copts = copts(),
    linkshared = 1,
    visibility = ["//visibility:public"],
)

py_library(
    name = "tipc_lib",
    srcs = ["__init__.py"], 
    
    data = [
        ":_tipc_lib.so", 
    ],
    imports = [
        "tipc",
    ],
    visibility = ["//visibility:public"],
)
