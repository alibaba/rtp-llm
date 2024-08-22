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
        "//src/fastertransformer/devices:devices_base",
        "//src/fastertransformer/devices:devices_base_impl",
        "//src/fastertransformer/core:cpu_allocator",
        "//src/fastertransformer/devices/arm_impl/type_bf16:hie_bfloat16",
        "@arm_compute//arm_compute:core_headers",
        "@arm_compute//arm_compute:runtime_headers",
        "@arm_compute//support",
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
