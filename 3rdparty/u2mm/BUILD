load("//bazel:defs.bzl", "rpm_library")

rpm_library(
    name = "u2mm" ,
    rpms = ["@u2mm//file:file"],
    lib_path = "usr/lib64",
    include_path = "usr/include",
    shared_lib = "libu2mm.so",
    hdrs = ["u2mm/u2mm.h", "u2mm/pool.h", "u2mm/memzone.h", "u2mm/common.h"],
)