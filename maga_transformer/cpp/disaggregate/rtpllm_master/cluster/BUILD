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
        "//maga_transformer/cpp/disaggregate/load_balancer:load_balancer",
        "//maga_transformer/cpp/disaggregate/rtpllm_master/estimator",
        "//maga_transformer/cpp/disaggregate/rtpllm_master/common",
    ],
    visibility=['//visibility:public'],
)