load("//:def.bzl", "copts")

cc_library(
    name = "cutlass_extensions",
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.cuh",
        "include/**/*.hpp",
    ]),
    deps = [
        "//rtp_llm/cpp/cutlass:cutlass_interface",
	    "//rtp_llm/cpp/cuda:cuda_utils",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = copts(),
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)
