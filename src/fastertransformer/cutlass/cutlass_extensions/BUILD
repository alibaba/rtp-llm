load("//:def.bzl", "copts")

cc_library(
    name = "cutlass_extensions",
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.cuh",
        "include/**/*.hpp",
    ]),
    deps = [
        "//src/fastertransformer/cutlass:cutlass_interface",
        "@cutlass//:cutlass",
        "@cutlass//:cutlass_utils",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = copts(),
    strip_include_prefix = "include",
    visibility = ["//src:__subpackages__"],
)
