load("//bazel:defs.bzl", "rpm_library")

# unicm rpm — backing file for @unicm referenced from
# `internal_source/rdma/aios/network/accl-barex/src/BUILD` via if_eic([...]).
# See sibling 3rdparty/u2mm/BUILD for the same rationale and history; both
# were dropped together by 5bc9f0cf3 and restored together.
rpm_library(
    name = "unicm" ,
    rpms = ["@unicm//file:file"],
    lib_path = "usr/lib64",
    include_path = "usr/include",
    shared_lib = "libunicm.so",
    hdrs = ["unicm.h"],
)
