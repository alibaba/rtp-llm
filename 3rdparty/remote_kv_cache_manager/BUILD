load("//bazel:defs.bzl", "rpm_library")

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
    origin_rpath = True,
    rpms = select({
        "@//:using_cuda13_x86": ["@remote_kv_cache_manager_client_rpm_cuda130_x86//file:file"],
        "@//:using_cuda13_arm": ["@remote_kv_cache_manager_client_rpm_cuda130_arm//file:file"],
        "//conditions:default": ["@remote_kv_cache_manager_client_rpm//file:file"],
    }),
    shared_libs = [
        "kv_cache_manager_client.so",
    ],
)

alias(
    name = "remote_kv_cache_manager_server_bin",
    actual = select({
        "@//:using_cuda13_x86": "@remote_kv_cache_manager_server_cuda130//:bin/kv_cache_manager_bin",
        "@//:using_cuda13_arm": "@remote_kv_cache_manager_server_cuda130//:bin/kv_cache_manager_bin",
        "//conditions:default": "@remote_kv_cache_manager_server//:bin/kv_cache_manager_bin",
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "remote_kv_cache_manager_client_headers",
    hdrs = [":remote_kv_cache_manager_client_hdrs_fg"],
    include_prefix = "kvcm_client",
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)
