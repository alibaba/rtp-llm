load("@//:def.bzl", "cuda_copts")

# The local repository can be rooted at /. Export only NVSHMEM's headers:
# aliasing all of /usr/include into -I virtual includes also shadows libc
# headers and breaks CUDA compilation with GCC 13 (_FloatN typedef conflicts).
_NVSHMEM_HEADERS = glob([
    "usr/include/nvshmem*.h",
    "usr/include/nvshmem*.cuh",
    "usr/include/nvshmem*.hpp",
    "usr/include/bootstrap_device_host/**",
    "usr/include/device/**",
    "usr/include/device_host/**",
    "usr/include/device_host_transport/**",
    "usr/include/host/**",
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
    hdrs = _NVSHMEM_HEADERS,
    strip_include_prefix = "usr/include",
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)
