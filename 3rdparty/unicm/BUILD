load("//bazel:defs.bzl", "rpm_library")

rpm_library(
    name = "unicm" ,
    rpms = ["@unicm//file:file"],
    lib_path = "usr/lib64",
    include_path = "usr/include",
    shared_lib = "libunicm.so",
    hdrs = ["unicm.h"],
)
