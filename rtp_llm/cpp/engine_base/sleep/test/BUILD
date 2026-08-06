load("//:def.bzl", "cc_test_wrapper", "copts")

cc_test_wrapper(
    name = "sleep_lifecycle_controller_test",
    srcs = [
        "SleepLifecycleControllerTest.cc",
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/engine_base/sleep",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test_wrapper(
    name = "drain_manager_test",
    srcs = [
        "DrainManagerTest.cc",
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/engine_base/sleep:drain_manager",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test_wrapper(
    name = "admission_gate_test",
    srcs = [
        "AdmissionGateTest.cc",
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/engine_base/sleep:admission_gate",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
