load("//bazel:defs.bzl", "rpm_library")
load("@//:def.bzl", "cuda_copts")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@//bazel:arch_select.bzl", "torch_deps")

rpm_library(
    name = "accl_ep" ,
    rpms = ["@accl_ep_rpm//file:file"],
    lib_path = "usr/lib64",
    include_path = "usr/include",
    shared_libs = [
        "accl_ep_cpp.cpython-310-x86_64-linux-gnu.so",
    ],
    deps = [
        ":accl_ep_hdrs",
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

cc_library(
    name = "accl_ep_hdrs",
    hdrs = glob([
        "csrc/*.hpp",
        "csrc/kernels/*.cuh",
    ]),
    includes = ["csrc"],
    deps = torch_deps() + [
        "@local_config_cuda//cuda:cuda_headers",
    ],
    copts = cuda_copts(),
    visibility = ["//visibility:public"],
)
