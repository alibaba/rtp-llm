load("//:def.bzl", "cc_test_wrapper", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "online_repetition_tracker_core",
    srcs = ["OnlineRepetitionTracker.cc"],
    hdrs = ["OnlineRepetitionTracker.h"],
    copts = copts(),
)

cc_library(
    name = "token_tool_call_loop_guard_core",
    srcs = ["TokenToolCallLoopGuard.cc"],
    hdrs = ["TokenToolCallLoopGuard.h"],
    copts = copts(),
)

cc_library(
    name = "online_repetition_tracker_pybind_lib",
    srcs = ["OnlineRepetitionPybind.cc"],
    copts = copts(),
    deps = [
        ":online_repetition_tracker_core",
        ":token_tool_call_loop_guard_core",
    ] + torch_deps(),
    alwayslink = True,
)

cc_binary(
    name = "online_repetition_tracker",
    deps = [":online_repetition_tracker_pybind_lib"],
    linkshared = 1,
)

cc_test_wrapper(
    name = "online_repetition_tracker_test",
    srcs = ["OnlineRepetitionTrackerTest.cc"],
    deps = [":online_repetition_tracker_core"],
    copts = copts(),
)

cc_test_wrapper(
    name = "token_tool_call_loop_guard_test",
    srcs = ["TokenToolCallLoopGuardTest.cc"],
    deps = [":token_tool_call_loop_guard_core"],
    copts = copts(),
)
