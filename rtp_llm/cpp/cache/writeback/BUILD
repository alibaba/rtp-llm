load("//:def.bzl", "copts")

cc_library(
    name = "pd_kv_writeback_transfer",
    srcs = ["PdKvWritebackTransfer.cc"],
    hdrs = ["PdKvWritebackTransfer.h"],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/cpp/cache:batch_kv_cache_resource",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "pd_kv_writeback_manifest",
    srcs = ["PdKvWritebackManifest.cc"],
    hdrs = ["PdKvWritebackManifest.h"],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/cpp/cache:batch_kv_cache_resource",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "pd_kv_writeback_manager",
    srcs = ["PdKvWritebackManager.cc"],
    hdrs = ["PdKvWritebackManager.h"],
    copts = copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":pd_kv_writeback_manifest",
        ":pd_kv_writeback_transfer",
        "//rtp_llm/cpp/cache:batch_kv_cache_resource",
        "//rtp_llm/cpp/config:config_modules",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "test_pd_kv_writeback_manifest",
    srcs = ["test/PdKvWritebackManifestTest.cc"],
    copts = copts(),
    deps = [
        ":pd_kv_writeback_manifest",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "test_pd_kv_writeback_manager",
    srcs = ["test/PdKvWritebackManagerTest.cc"],
    copts = copts(),
    deps = [
        ":pd_kv_writeback_manager",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
