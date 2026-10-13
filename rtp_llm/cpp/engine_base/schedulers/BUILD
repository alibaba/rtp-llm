load("//:def.bzl", "copts")

cc_library(
    name = "scheduler_admission",
    hdrs = ["SchedulerAdmission.h"],
    srcs = ["SchedulerAdmission.cc"],
    copts = copts(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "schedulers",
    hdrs = glob([
        "*.h",
    ], exclude = ["SchedulerAdmission.h"]),
    srcs = glob([
        "*.cc",
    ], exclude = ["SchedulerAdmission.cc"]),
    deps = [
        ":scheduler_admission",
        "//rtp_llm/cpp/engine_base/sleep:drain_manager",
        "//rtp_llm/cpp/engine_base/stream:stream",
        "//rtp_llm/cpp/engine_base/system_prompt:system_prompt",
        "//rtp_llm/cpp/cache:cache",
        "//rtp_llm/cpp/models:context_parallel_layout",
        "//rtp_llm/cpp/utils:profiling_scope",
    ],
    visibility = ["//visibility:public"],
)
