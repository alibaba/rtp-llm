load("//:def.bzl", "copts")

cc_library(
    name = "arm_cpu_impl",
    hdrs = glob([
        "*.h",
        "gemm_opt/*.h",
        "gemm_opt/*.hpp",
    ]),
    srcs = glob([
        "*.cc",
        "gemm_opt/*.cc",
    ]),
    deps = [
        "//rtp_llm/cpp/devices:devices_base",
        "//rtp_llm/cpp/devices:devices_base_impl",
        "//rtp_llm/cpp/core:cpu_allocator",
        "//rtp_llm/cpp/devices/arm_impl/type_bf16:hie_bfloat16",
        "//rtp_llm/cpp/pybind:th_utils",
        "@arm_compute//arm_compute:core_headers",
        "@arm_compute//arm_compute:runtime_headers",
        "@arm_compute//support",
        "@KleidiAI//kai/ukernels/matmul",
    ],
    visibility = ["//visibility:public"],
    copts = copts() + [
        "-march=armv8.6-a+dotprod+fp16+bf16+i8mm+sve+sve2",
        "-fopenmp",
        "-fPIC",
    ],
    alwayslink = 1,
    linkopts = ["-fopenmp"],
)
