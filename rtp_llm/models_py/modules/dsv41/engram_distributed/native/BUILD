load("@arch_config//:arch_select.bzl", "torch_deps")
load("//:def.bzl", "copts")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "fabric_memory_lib",
    srcs = ["FabricMemory.cc"],
    copts = copts(),
    local_defines = ["TORCH_EXTENSION_NAME=libfabric_memory"],
    alwayslink = True,
    deps = [
        "@local_config_cuda//cuda:cuda_driver",
        "@local_config_cuda//cuda:cuda_headers",
    ] + torch_deps(),
    target_compatible_with = select({
        "//:using_cuda13_arm": [],
        "//:using_cuda13_x86": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)
