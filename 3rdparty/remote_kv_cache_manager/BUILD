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
        "@//:using_cuda13_x86": ["@remote_kv_cache_manager_client_rpm_cuda130_x86//file:file"],
        "@//:using_cuda13_arm": ["@remote_kv_cache_manager_client_rpm_cuda130_arm//file:file"],
        "@//:using_cuda12_9_x86": ["@remote_kv_cache_manager_client_rpm_cuda129_x86//file:file"],
        "@//:cuda_pre_12_9": ["@remote_kv_cache_manager_client_rpm_cuda12_x86//file:file"],
        "//conditions:default": ["@remote_kv_cache_manager_client_rpm//file:file"],
    }),
    shared_libs = [
        "kv_cache_manager_client.so",
    ],
)
