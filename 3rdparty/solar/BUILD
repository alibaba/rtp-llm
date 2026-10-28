# Wraps the @solar rpm (declared as http_file in `internal_source/deps/http.bzl`)
# into a cc_library `//3rdparty/solar:solar`, consumed by
# `internal_source/rdma/aios/network/accl-barex/**` when ACCL_USE_SOLAR is on.
#
# Same brittleness as `//3rdparty/tnet:tnet` — required for CUDA + ROCm Bazel
# analysis of `//:th_transformer`. Do NOT delete during deps cleanup unless the
# accl-barex consumers and the `@solar` http_file are also being removed.
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
