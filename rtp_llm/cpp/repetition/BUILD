load("//:def.bzl", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

package(default_visibility = ["//visibility:public"])

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
        ":token_tool_call_loop_guard_core",
    ] + torch_deps(),
    alwayslink = True,
)

cc_binary(
    name = "online_repetition_tracker",
    deps = [":online_repetition_tracker_pybind_lib"],
    linkshared = 1,
)
