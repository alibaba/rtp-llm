load("//:def.bzl", "copts")

cc_library(
    name = "sleep",
    hdrs = [
        "SleepLifecycleController.h",
    ],
    srcs = [
        "SleepLifecycleController.cc",
    ],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/utils:core_utils",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "drain_manager",
    hdrs = [
        "DrainManager.h",
    ],
    srcs = [
        "DrainManager.cc",
    ],
    copts = copts(),
    deps = [
        ":sleep",
        "//rtp_llm/cpp/utils:core_utils",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "admission_gate",
    hdrs = [
        "AdmissionGate.h",
    ],
    srcs = [
        "AdmissionGate.cc",
    ],
    copts = copts(),
    deps = [
        ":sleep",
        "//rtp_llm/cpp/model_rpc/proto:model_rpc_service_cc_proto",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/cpp/utils:error_code",
    ],
    visibility = ["//visibility:public"],
)
