cc_library(
    name = "prefill_progress",
    srcs = ["PrefillProgress.cc"],
    hdrs = ["PrefillProgress.h"],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "prefill_progress_test",
    srcs = ["PrefillProgressTest.cc"],
    deps = [
        ":prefill_progress",
        "@com_google_googletest//:gtest_main",
    ],
)
