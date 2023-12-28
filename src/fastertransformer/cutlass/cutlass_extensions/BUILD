load("//:def.bzl", "cuda_copts")

cc_library(
    name = "cutlass_extensions",
    hdrs = glob([
        "include/**/*.h",
    ]),
    deps = [
        "@cutlass//:cutlass",
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = cuda_copts(),
    strip_include_prefix = "include",
    visibility = ["//src:__subpackages__"],
)
