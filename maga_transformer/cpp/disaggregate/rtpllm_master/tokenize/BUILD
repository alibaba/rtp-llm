cc_library(
    name = "tokenize_service",
    srcs = ["RemoteTokenizeModule.cpp"],
    hdrs = ["RemoteTokenizeModule.h"],
    deps = [
        "//maga_transformer/cpp/disaggregate/rtpllm_master/common",
        "//maga_transformer/cpp/disaggregate/rtpllm_master/cluster:prefill_load_balancer"
    ],
    visibility = ["//visibility:public"],
)
