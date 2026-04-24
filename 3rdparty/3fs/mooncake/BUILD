load("//bazel:defs.bzl", "rpm_library")

_HF3FS_SHARED_LIBS_WITHOUT_GLOG = [
    "libhf3fs_api_shared.so",
    "libboost_context.so.1.71.0",
    "libboost_filesystem.so.1.71.0",
    "libboost_program_options.so.1.71.0",
    "libboost_regex.so.1.71.0",
    "libboost_system.so.1.71.0",
    "libboost_thread.so.1.71.0",
    "libboost_atomic.so.1.71.0",
    "libdouble-conversion.so.3",
    "libevent-2.1.so.7",
    "libdwarf.so.1",
    "libicui18n.so.66",
    "libicuuc.so.66",
    "libicudata.so.66",
    "libunwind.so.8",
    "libssl.so.1.1",
    "libcrypto.so.1.1",
]

rpm_library(
    name = "hf3fs_mooncake_impl",
    rpms = [
        "@hf3fs_rpm//file:file",
    ],
    lib_path = "usr/lib64",
    include_path = "usr/include",
    shared_libs = _HF3FS_SHARED_LIBS_WITHOUT_GLOG,
    hdrs = [
        "hf3fs_usrbio.h",
    ],
    deps = [
        "@mooncake_transfer_engine_local//:glog",
        "@mooncake_transfer_engine_local//:gflags",
    ],
)
