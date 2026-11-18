py_library(
    name = "metrics",
    srcs = glob([
        "*.py"
    ], exclude = ["*_test.py"]),
    visibility = ["//visibility:public"],
    deps = ["//rtp_llm/aios/kmonitor:kmonitor_py",
            "//rtp_llm/distribute:distribute"],
)

py_test(
    name = "kmonitor_metric_reporter_test",
    srcs = ["kmonitor_metric_reporter_test.py"],
    deps = [":metrics"],
)
