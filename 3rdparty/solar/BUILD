load("//bazel:defs.bzl", "rpm_library")

rpm_library(
    name = "solar" ,
    rpms = ["@solar//file:file"],
    lib_path = "usr/lib64",
    include_path = "usr/include",
    static_lib = "libsolar.a",
    shared_lib = "libsolar.so",
    hdrs = ["solar_verbs.h",],
)