cc_library(
    name = "tokenize_service",
    srcs = ["RemoteTokenizeModule.cpp"],
    hdrs = ["RemoteTokenizeModule.h"],
    deps = [
        "//rtp_llm/cpp/disaggregate/rtpllm_master/common",
        "//rtp_llm/cpp/disaggregate/rtpllm_master/cluster:prefill_load_balancer"
    ],
    visibility = ["//visibility:public"],
)
