load("//:def.bzl", "copts")

cc_library(
    name = "scheduler_synchronization_state",
    hdrs = ["SchedulerSynchronizationState.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "schedulers",
    hdrs = glob(["*.h"], exclude = ["SchedulerSynchronizationState.h"]),
    srcs = glob([
        "*.cc",
    ]),
    deps = [
        ":scheduler_synchronization_state",
        "//rtp_llm/cpp/engine_base/stream:stream",
        "//rtp_llm/cpp/engine_base/system_prompt:system_prompt",
        "//rtp_llm/cpp/cache:cache",
        "//rtp_llm/cpp/models:context_parallel_layout",
        "//rtp_llm/cpp/utils:profiling_scope",
    ],
    visibility = ["//visibility:public"],
)
