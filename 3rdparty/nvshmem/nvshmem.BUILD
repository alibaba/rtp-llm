load("@//:def.bzl", "cuda_copts")

_NVSHMEM_HEADERS = glob([
    "usr/include/nvshmem*.h",
    "usr/include/host/**",
    "usr/include/device/**",
    "usr/include/device_host/**",
    "usr/include/device_host_transport/**",
    "usr/include/bootstrap_device_host/**",
    "usr/include/non_abi/**",
])

cc_library(
    name = "nvshmem_host",
    srcs = [
        "usr/lib64/libnvshmem.a",
        "usr/lib64/nvshmem_bootstrap_uid.so",
    ],
    hdrs = _NVSHMEM_HEADERS,
    strip_include_prefix = "usr/include",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "nvshmem_device",
    srcs = ["usr/lib64/libnvshmem_device.a"],
    hdrs = _NVSHMEM_HEADERS,
    strip_include_prefix = "usr/include",
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "nvshmem_device_archive",
    srcs = ["usr/lib64/libnvshmem_device.a"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "nvshmem_host_archive",
    srcs = ["usr/lib64/libnvshmem.a"],
    visibility = ["//visibility:public"],
)
