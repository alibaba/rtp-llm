cc_library(
    name = "execution_recorder",
    srcs = ["ExecutionRecorder.cc"],
    hdrs = ["ExecutionRecorder.h"],
    deps = [
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/utils:core_utils",
        "@havenask//aios/autil:env_util",
        "@havenask//aios/alog:alog",
        "@havenask//aios/autil:json",
    ],
    linkopts = ["-pthread"],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "execution_recorder_test",
    data = ["//rtp_llm:alog_config"],
    copts = ["-UNDEBUG"],
    srcs = ["ExecutionRecorderTest.cc"],
    deps = [
        ":execution_recorder",
        "//rtp_llm/cpp/config:config_modules",
        "//rtp_llm/cpp/utils:core_utils",
    ],
)
