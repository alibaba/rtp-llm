load("//:def.bzl", "cc_test_wrapper", "copts")

package(default_visibility = ["//visibility:public"])

cc_test_wrapper(
    name = "token_tool_call_loop_guard_test",
    srcs = ["TokenToolCallLoopGuardTest.cc"],
    deps = [
        "//rtp_llm/cpp/repetition:token_tool_call_loop_guard_core",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
    copts = copts(),
)
