
cc_library(
    name = "cutlass",
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.cuh",
        "include/**/*.hpp",
    ]),
    deps = [
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",
    ],
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)
