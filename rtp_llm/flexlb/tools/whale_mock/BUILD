# Only the opt-in ingress helper belongs in the frontend wheel. The bundle
# supervisor, configs and observer are copied into the test image separately.
py_library(
    name = "schedule_only",
    srcs = ["schedule_only.py"],
    deps = ["//rtp_llm/dash_sc/proto:predict_v2_py_proto"],
    visibility = ["//visibility:public"],
)
