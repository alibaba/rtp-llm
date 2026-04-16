package(default_visibility = ["//visibility:public"])

load("@release_version//:defs.bzl", "RELEASE_VERSION")

load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "rocm_default_copts",
    _if_rocm = "if_rocm",
)

if_rocm = _if_rocm

filegroup(
    name = "cutlass_config",
    srcs = glob(["utils/gemm_utils/luts/*"]),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "alog_config",
    srcs = ["config/alog.conf"],
    visibility = ["//visibility:public"],
)
