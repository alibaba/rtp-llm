load("//bazel:defs.bzl", "rpm_library")

# u2mm rpm — backing file for @u2mm referenced from
# `internal_source/rdma/aios/network/accl-barex/src/BUILD` via if_eic([...]).
# Required by the EIC variant of ROCm + CUDA RDMA builds (image-amd-eic /
# image-sm9x-eic etc. with `--copt=-DACCL_USE_EIC=1`). Do NOT delete unless
# the matching `if_eic([...])` in accl-barex/src/BUILD is also being removed.
# Originally added by ae2e4f68d, dropped by 5bc9f0cf3 during deps cleanup
# while EIC builds were silently bypassed (BAZEL_ARGS dead field), restored
# once BAZEL_ARGS→RTP_BAZEL_APPEND_CONFIG migration started actually applying
# the EIC copt.
rpm_library(
    name = "u2mm" ,
    rpms = ["@u2mm//file:file"],
    lib_path = "usr/lib64",
    include_path = "usr/include",
    shared_lib = "libu2mm.so",
    hdrs = ["u2mm/u2mm.h", "u2mm/pool.h", "u2mm/memzone.h", "u2mm/common.h"],
)
