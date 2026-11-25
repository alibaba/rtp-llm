load("//bazel:defs.bzl", "rpm_library", "runfiles_alias")

config_setting(
    name = "pace_smoke_prebuilt",
    define_values = {"pace_smoke_prebuilt": "true"},
    visibility = ["//rtp_llm/test/smoke:__pkg__"],
)

alias(
    name = "default_kvcm_sdk_rpm",
    actual = select({
        "@//:using_cuda13_x86": "@remote_kv_cache_manager_client_rpm_cuda130_x86//file:file",
        "@//:using_cuda13_arm": "@remote_kv_cache_manager_client_rpm_cuda130_arm//file:file",
        "@//:using_cuda12_9_x86": "@remote_kv_cache_manager_client_rpm_cuda129_x86//file:file",
        "@//:cuda_pre_12_9": "@remote_kv_cache_manager_client_rpm_cuda12_x86//file:file",
        "//conditions:default": "@remote_kv_cache_manager_client_rpm//file:file",
    }),
)

alias(
    name = "pace_smoke_sdk_rpm",
    actual = select({
        "@//:using_cuda13_x86": "@pace_test_kvcm_sdk_rpm_cuda130_x86//file:file",
        "@//:using_cuda13_arm": "@pace_test_kvcm_sdk_rpm_cuda130_arm//file:file",
    }, no_match_error = "The PACE prebuilt smoke SDK requires a CUDA13 configuration."),
)

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
        ":pace_smoke_prebuilt": [":pace_smoke_sdk_rpm"],
        "//conditions:default": [":default_kvcm_sdk_rpm"],
    }),
    shared_libs = [
        "kv_cache_manager_client.so",
    ],
    origin_rpath = True,
)

alias(
    name = "default_kvcm_manager_bin",
    actual = select({
        "@//:using_cuda13_x86": "@remote_kv_cache_manager_server_cuda130//:bin/kv_cache_manager_bin",
        "@//:using_cuda13_arm": "@remote_kv_cache_manager_server_cuda130//:bin/kv_cache_manager_bin",
        "//conditions:default": "@remote_kv_cache_manager_server//:bin/kv_cache_manager_bin",
    }),
)

alias(
    name = "pace_smoke_manager_bin",
    actual = select({
        "@//:using_cuda13_x86": "@pace_test_kvcm_manager_x86//:bin/kv_cache_manager_bin",
        "@//:using_cuda13_arm": "@pace_test_kvcm_manager_arm//:bin/kv_cache_manager_bin",
    }, no_match_error = "The pinned prebuilt KVCM Manager images require a CUDA13 configuration."),
)

runfiles_alias(
    name = "pace_smoke_manager_startup_config",
    src = select({
        "@//:using_cuda13_x86": "@pace_test_kvcm_manager_x86//:etc/default_startup_config.json",
        "@//:using_cuda13_arm": "@pace_test_kvcm_manager_arm//:etc/default_startup_config.json",
    }),
    runfile_path = "3rdparty/remote_kv_cache_manager/etc/default_startup_config.json",
    testonly = True,
    visibility = ["//rtp_llm/test/smoke:__pkg__"],
)

runfiles_alias(
    name = "remote_kv_cache_manager_server_bin",
    src = select({
        ":pace_smoke_prebuilt": ":pace_smoke_manager_bin",
        "//conditions:default": ":default_kvcm_manager_bin",
    }),
    runfile_path = "3rdparty/remote_kv_cache_manager/bin/kv_cache_manager_bin",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "remote_kv_cache_manager_client_headers",
    hdrs = [
        ":remote_kv_cache_manager_client_hdrs_fg",
        "include/HostCacheStateCompat.h",
        "include/TransferClientCompat.h",
    ],
    include_prefix = "kvcm_client",
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)
