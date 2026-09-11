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
        "//:using_cuda13_x86": ["@remote_kv_cache_manager_client_cuda13_rpm//file:file"],
        "//conditions:default": ["@remote_kv_cache_manager_client_rpm//file:file"],
    }),
    shared_libs = [
        "kv_cache_manager_client.so",
    ],
)

# Optional server artifact for CUDA 13 tools; existing smoke data declarations
# keep their original server repository until those cases opt into CUDA 13.
alias(
    name = "remote_kv_cache_manager_server_bin",
    actual = select({
        "//:using_cuda13_x86": "@remote_kv_cache_manager_server_cuda13//:bin/kv_cache_manager_bin",
        "//conditions:default": "@remote_kv_cache_manager_server//:bin/kv_cache_manager_bin",
    }),
    visibility = ["//visibility:public"],
)
