load("//:def.bzl", "rpm_library", "copts", "cuda_copts", "rocm_copts", "any_cuda_copts", "if_cuda")
load("@arch_config//:arch_select.bzl", "torch_deps")
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "trt_beam_search_impl",
    srcs = glob([
        "*.cu",
    ]),
    hdrs = glob([
        "*.h",
        "**/*.hpp",
    ]),
    deps = [
        "//rtp_llm/models_py/bindings/cuda:cuda_utils_cu",
        "//rtp_llm/cpp/utils:math_utils",
        "//rtp_llm/cpp/utils:core_utils",
    ] + select({
        "@//:using_rocm": [
            "//rtp_llm/models_py/bindings/rocm:rocm_types_hdr",
            "//rtp_llm/models_py/bindings/rocm:rocm_host_utils",
            "//rtp_llm/models_py/bindings/rocm:hipcub_shims",
            "@local_config_rocm//rocm:rocm_headers",
            "@local_config_rocm//rocm:hip",
        ],
        "//conditions:default": [
            ":trt_beam_search_header",
            "//rtp_llm/models_py/bindings/cuda:trt_utils",
            "//rtp_llm/models_py/bindings/cuda:cuda_host_utils",
            "@local_config_cuda//cuda:cuda",
            "@local_config_cuda//cuda:cudart",
        ],
    }),
    # addCumLogProbs is instantiated in beamSearchKernels.cu and launched from
    # the beam-width specializations. CUDA 12.1+ otherwise gives its host stub
    # internal linkage, leaving the cross-translation-unit launches unresolved.
    copts = any_cuda_copts() + if_cuda(["-nvcc_options=static-global-template-stub=false"]),
    visibility = ["//:__subpackages__"],
    alwayslink = True,
)

cc_library(
    name = "trt_beam_search_header",
    hdrs = glob([
        "*.h",
        "**/*.h",
        "**/*.hpp",
        "**/*.cuh",
    ]),
    deps = select({
        "@//:using_rocm": [
            "@local_config_rocm//rocm:rocm_headers",
            "@local_config_rocm//rocm:hip",
        ],
        "//conditions:default": [
            "@local_config_cuda//cuda:cuda",
            "@local_config_cuda//cuda:cudart",
        ],
    }),
    copts = any_cuda_copts(),
    visibility = ["//visibility:public"],
)
