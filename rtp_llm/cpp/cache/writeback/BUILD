load("//:def.bzl", "copts")

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
