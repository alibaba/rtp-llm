load("//bazel:defs.bzl", "rpm_library")

rpm_library(
    name = "tnet" ,
    rpms = ["@tnet//file:file"],
    lib_path = "usr/lib64",
    include_path = "usr/include",
    static_lib = "liblibtnet.a",
    shared_lib = "liblibtnet.so",
    hdrs = ["libtnet.h",],
)