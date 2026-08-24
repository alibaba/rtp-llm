load("@//:def.bzl", "cuda_copts")

cc_library(
    name = "nvshmem_host",
    srcs = [
        "usr/lib64/libnvshmem.a",
        "usr/lib64/nvshmem_bootstrap_uid.so",
    ],
    hdrs = glob(["usr/include/**"]),
    strip_include_prefix = "usr/include",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "nvshmem_device",
    hdrs = glob(["usr/include/**"]),
    strip_include_prefix = "usr/include",
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)
