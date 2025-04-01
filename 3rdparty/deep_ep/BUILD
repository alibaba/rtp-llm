load("//bazel:defs.bzl", "rpm_library")

rpm_library(
    name = "deep_ep" ,
    rpms = ["@deep_ep_rpm//file:file"],
    lib_path = "usr/lib64",
    include_path = "usr/include",
    #static_lib = "libnvshmem.a",
    shared_libs = [
        "deep_ep_cpp.cpython-310-x86_64-linux-gnu.so",
    ],
    deps = [
        "@deep_ep//:deep_ep_hdrs",
    ],
    hdrs = [
        "nvshmem.h",
        "non_abi/nvshmem_build_options.h",
        "host/nvshmem_api.h",
        "device_host/nvshmem_common.cuh",
        "device_host_transport/nvshmem_common_transport.h",
        "device_host/nvshmem_types.h",
        "bootstrap_device_host/nvshmem_uniqueid.h",
        "device_host_transport/nvshmem_constants.h",
        "deep_ep.hpp",
        "config.hpp",
        "kernels/exception.cuh",
        "kernels/configs.cuh",
        "kernels/api.cuh",
        "event.hpp",
    ],
)
