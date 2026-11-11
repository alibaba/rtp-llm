load("//:def.bzl", "copts")
load("//bazel:arch_select.bzl", "no_block_copy_link_deps")

exports_files(["TransferExecutorTest.cc"])

cc_library(
    name = "copy_options",
    hdrs = ["DeviceHostCopyOptions.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "legacy",
    srcs = [
        "TransferExecutor.cc",
        "DeviceHostTransferExecutor.cc",
        "HostDiskTransferExecutor.cc",
        "DeviceDiskTransferExecutor.cc",
        "DeviceHostCopyStrategy.cc",
        "Cuda3DBatchDeviceHostCopyStrategy.cc",
        "LegacyTransferEngine.cc",
        "CrcCopyExecutorSync.cc",
    ],
    hdrs = [
        "TransferExecutor.h",
        "DeviceHostTransferExecutor.h",
        "HostDiskTransferExecutor.h",
        "DeviceDiskTransferExecutor.h",
        "DeviceHostCopyStrategy.h",
        "LegacyTransferEngine.h",
    ],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":copy_options",
        "//rtp_llm/cpp/cache/block_tree_cache:transfer_common",
        "//rtp_llm/models_py/bindings:no_block_copy_hdr",
    ] + no_block_copy_link_deps(),
)
