load("//:def.bzl", "rpm_library", "copts", "cuda_copts",)
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "trt_beam_search_impl",
    srcs = glob([
        "*.cu",
    ]),
    hdrs = glob([
        "*.h",
    ]),
    deps = [
        ":trt_beam_search_header",
        "//rtp_llm/cpp/cuda:trt_utils",
        "//rtp_llm/cpp/cuda:cuda_host_utils",
        "//rtp_llm/cpp/cuda:cuda_utils_cu",
        "//rtp_llm/cpp/utils:math_utils",
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = cuda_copts(),
    visibility = ["//:__subpackages__"],
    alwayslink = True,
)


cc_library(
    name = "trt_beam_search_header",
    hdrs = glob([
        "*.h",
        "**/*.h",
        "**/*.cuh",
    ]),
    deps = [
        "@local_config_cuda//cuda:cuda",
        "@local_config_cuda//cuda:cudart",
    ],
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)