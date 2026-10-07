load("//:def.bzl", "copts")

cc_library(
    name = "pdfusion_schedule_coordinator",
    hdrs = ["PDFusionScheduleCoordinator.h"],
    srcs = ["PDFusionScheduleCoordinator.cc"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "schedulers",
    hdrs = glob([
        "*.h",
    ], exclude = ["PDFusionScheduleCoordinator.h"]),
    srcs = glob([
        "*.cc",
    ], exclude = ["PDFusionScheduleCoordinator.cc"]),
    deps = [
        ":pdfusion_schedule_coordinator",
        "//rtp_llm/cpp/engine_base/stream:stream",
        "//rtp_llm/cpp/engine_base/system_prompt:system_prompt",
        "//rtp_llm/cpp/cache:cache",
        "//rtp_llm/cpp/models:context_parallel_layout",
        "//rtp_llm/cpp/utils:profiling_scope",
    ],
    visibility = ["//visibility:public"],
)
