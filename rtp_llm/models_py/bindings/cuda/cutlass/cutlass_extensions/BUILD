load("//:def.bzl", "copts")

cc_library(
    name = "cutlass_extensions",
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.cuh",
        "include/**/*.hpp",
    ]),
    deps = [
        "//rtp_llm/models_py/bindings/cuda/cutlass:cutlass_interface",
	    "//rtp_llm/models_py/bindings/cuda:cuda_host_utils",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = copts(),
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)
