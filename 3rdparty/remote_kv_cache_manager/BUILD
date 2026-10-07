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
    rpms = select({
        "@//:using_cuda12_9_x86": ["@remote_kv_cache_manager_client_rpm_cuda129_x86//file:file"],
        "@//:using_cuda13_x86": ["@remote_kv_cache_manager_client_rpm_cuda130_x86//file:file"],
        "@//:using_cuda13_arm": ["@remote_kv_cache_manager_client_rpm_cuda130_arm//file:file"],
        "//conditions:default": ["@remote_kv_cache_manager_client_rpm//file:file"],
    }),
    shared_libs = [
        "kv_cache_manager_client.so",
    ],
    origin_rpath = True,
)

runfiles_alias(
    name = "remote_kv_cache_manager_server_bin",
    src = "@remote_kv_cache_manager_server//:bin/kv_cache_manager_bin",
    runfile_path = "3rdparty/remote_kv_cache_manager/bin/kv_cache_manager_bin",
    # The standalone Manager package is x86-only and shared by CUDA toolchains.
    target_compatible_with = ["@platforms//cpu:x86_64"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "remote_kv_cache_manager_client_headers",
    hdrs = [":remote_kv_cache_manager_client_hdrs_fg"],
    include_prefix = "kvcm_client",
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)

_CLIENT_METADATA = ["KVCM_SOURCE_ID", "KVCM_CLIENT_VARIANT", "KVCM_ARTIFACT_SHA256"]
_SERVER_METADATA = ["KVCM_SOURCE_ID", "KVCM_ARTIFACT_SHA256", "etc/default_startup_config.json"]

filegroup(
    name = "remote_kv_cache_manager_client_metadata",
    srcs = select({
        "@//:using_cuda12_9_x86": ["@remote_kv_cache_manager_client_rpm_cuda129_x86//:" + name for name in _CLIENT_METADATA],
        "@//:using_cuda13_x86": ["@remote_kv_cache_manager_client_rpm_cuda130_x86//:" + name for name in _CLIENT_METADATA],
        "@//:using_cuda13_arm": ["@remote_kv_cache_manager_client_rpm_cuda130_arm//:" + name for name in _CLIENT_METADATA],
        "//conditions:default": ["@remote_kv_cache_manager_client_rpm//:" + name for name in _CLIENT_METADATA],
    }),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "remote_kv_cache_manager_server_metadata",
    srcs = ["@remote_kv_cache_manager_server//:" + name for name in _SERVER_METADATA],
    target_compatible_with = ["@platforms//cpu:x86_64"],
    visibility = ["//visibility:public"],
)
