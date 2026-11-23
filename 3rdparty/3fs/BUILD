load("//bazel:defs.bzl", "rpm_library")

# The default release is x86_64. Distributions providing a native 3FS
# package can select it for the corresponding target toolchain.
label_flag(
    name = "client_rpm",
    build_setting_default = "@hf3fs_rpm//file:file",
    visibility = ["//visibility:public"],
)

rpm_library(
    name = "hf3fs",
    rpms = [
        ":client_rpm",
    ],
    lib_path = "usr/lib64",
    include_path = "usr/include",
    shared_libs = [
        "libhf3fs_api_shared.so",
        "libboost_context.so.1.71.0",
        "libboost_filesystem.so.1.71.0",
        "libboost_program_options.so.1.71.0",
        "libboost_regex.so.1.71.0",
        "libboost_system.so.1.71.0",
        "libboost_thread.so.1.71.0",
        "libboost_atomic.so.1.71.0",
        "libdouble-conversion.so.3",
        "libgflags.so.2.2",
        "libglog.so.0",
        "libevent-2.1.so.7",
        "libdwarf.so.1",
        "libicui18n.so.66",
        "libicuuc.so.66",
        "libicudata.so.66",
        "libunwind.so.8",
        "libssl.so.1.1",
        "libcrypto.so.1.1",
    ],
    hdrs = [
        "hf3fs_usrbio.h",
    ],
    origin_rpath = True,
)
