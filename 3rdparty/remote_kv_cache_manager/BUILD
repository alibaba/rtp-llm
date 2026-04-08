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
    rpms = select({
        "@platforms//cpu:aarch64": ["@remote_kv_cache_manager_client_rpm_aarch64//file:file"],
        "@platforms//cpu:x86_64": ["@remote_kv_cache_manager_client_rpm_x86_64//file:file"],
        "//conditions:default": ["@remote_kv_cache_manager_client_rpm_x86_64//file:file"],
    }),
    shared_libs = [
        "kv_cache_manager_client.so",
    ],
)
