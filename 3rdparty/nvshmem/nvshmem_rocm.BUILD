load("@//:def.bzl", "rocm_copts")

cc_library(
    name = "nvshmem_rocm_hdrs",
    hdrs = glob([
        "opt/nvshmem/include/*.h",
        "opt/nvshmem/include/**/*.h",
    ]),
    strip_include_prefix = "opt/nvshmem/include",
    copts = rocm_copts(),
    visibility = ["//visibility:public"],
)
