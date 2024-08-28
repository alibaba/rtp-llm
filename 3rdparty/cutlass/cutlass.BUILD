
cc_library(
    name = "cutlass",
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.cuh",
        "include/**/*.hpp",
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
        "tools/util/include/cutlass/util/*.h",
        "tools/util/include/cutlass/util/*.hpp",
        "tools/util/include/cutlass/util/*.cuh"
    ]),
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    strip_include_prefix = "tools/util/include/",
    visibility = ["//visibility:public"],
)


