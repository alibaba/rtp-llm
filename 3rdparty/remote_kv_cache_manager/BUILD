load("//bazel:defs.bzl", "rpm_library", "runfiles_alias")

rpm_library(
    name = "remote_kv_cache_manager_client",
    hdrs = [
        "common.h",
        "manager_client.h",
        "meta_client.h",
        "rtp_llm_client.h",
        "transfer_client.h",
    ],
    include_path = "include/kv_cache_manager",
    include_prefix = "kvcm_client",
    lib_path = "lib",
    rpms = ["@remote_kv_cache_manager_client_rpm//file:file"],
    shared_libs = [
        "kv_cache_manager_client.so",
    ],
    origin_rpath = True,
)

runfiles_alias(
    name = "remote_kv_cache_manager_server_bin",
    src = "@remote_kv_cache_manager_server//:bin/kv_cache_manager_bin",
    runfile_path = "3rdparty/remote_kv_cache_manager/bin/kv_cache_manager_bin",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "remote_kv_cache_manager_client_headers",
    hdrs = [":remote_kv_cache_manager_client_hdrs_fg"],
    include_prefix = "kvcm_client",
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)
