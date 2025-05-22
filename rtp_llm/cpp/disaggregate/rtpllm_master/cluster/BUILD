cc_library(
    name = "prefill_load_balancer",
    srcs = [
        "PrefillLoadBalancer.cpp",
    ],
    hdrs = [
        "PrefillLoadBalancer.h",
        "PrefillWorkerInfo.h",
    ],
    deps = [
        "//rtp_llm/cpp/disaggregate/load_balancer:load_balancer",
        "//rtp_llm/cpp/disaggregate/rtpllm_master/estimator",
        "//rtp_llm/cpp/disaggregate/rtpllm_master/common",
    ],
    visibility=['//visibility:public'],
)