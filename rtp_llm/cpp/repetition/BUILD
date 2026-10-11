load("//:def.bzl", "copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

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
    # Same rationale as th_transformer_config: the pybind lib pulls the full
    # torch_deps() link line but references only torch CPU ABI symbols, so
    # --as-needed keeps the dead CUDA DT_NEEDED entries out of the .so.
    linkopts = ["-Wl,--as-needed"],
    deps = [":online_repetition_tracker_pybind_lib"],
    linkshared = 1,
)
