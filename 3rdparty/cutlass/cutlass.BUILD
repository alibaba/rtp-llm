
cc_library(
    name = "cutlass",
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.cuh",
        "include/**/*.hpp",
        "include/**/*.inl",
    ]),
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cutlass_utils",
    hdrs = glob([
        "tools/util/include/**/*.h",
        "tools/util/include/**/*.hpp",
        "tools/util/include/**/*.cuh"
    ]),
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    strip_include_prefix = "tools/util/include/",
    visibility = ["//visibility:public"],
)

filegroup(
    name = "cutlass_origin",
    srcs = glob([
        "include/**/*.h",
        "include/**/*.cuh",
        "include/**/*.hpp",
        "include/**/*.inl"
    ]),
    visibility = ["//visibility:public"],
)