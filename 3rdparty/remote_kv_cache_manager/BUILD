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
    rpms = [
        "@remote_kv_cache_manager_client_rpm//file:file",
    ],
    shared_libs = [
        "kv_cache_manager_client.so",
    ],
)

# Kept separate from remote_kv_cache_manager_client so the currently pinned
# RPM remains usable by the existing KV-cache path. Only the opt-in EMB build
# requests these headers, after a KVCM RPM containing the object API is pinned.
rpm_library(
    name = "remote_kv_cache_manager_object_headers",
    hdrs = [
        "kv_meta_client.h",
        "kv_meta_object_client.h",
        "kv_meta_transfer_client.h",
    ],
    include_path = "include/kv_cache_manager",
    include_prefix = "kvcm_client",
    rpms = [
        "@remote_kv_cache_manager_client_rpm//file:file",
    ],
)
