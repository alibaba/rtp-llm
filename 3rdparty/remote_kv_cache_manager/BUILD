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
        "//:using_cuda13_x86": ["@remote_kv_cache_manager_client_rpm_cuda13_x86_64//file:file"],
        "//conditions:default": ["@remote_kv_cache_manager_client_rpm//file:file"],
    }),
    shared_libs = [
        "kv_cache_manager_client.so",
    ],
    # Remote KV cache is disabled for CUDA ARM configs. Keep every generated
    # RPM target incompatible so an accidental enable fails during analysis
    # instead of silently extracting the default x86 package.
    target_compatible_with = select({
        "//:using_cuda_arm": ["@platforms//:incompatible"],
        "//conditions:default": [],
    }),
)
