cc_library(
    name = "load_balancer",
    srcs = [
        "RRLoadBalancer.cpp",
    ],
    hdrs = [
        "RRLoadBalancer.h",
    ],
    deps = [
        "@havenask//aios/autil:log",
        "@havenask//aios/autil:thread",
        "//maga_transformer/cpp/disaggregate/load_balancer/subscribe:subscribe",
    ],
    visibility=['//visibility:public'],
)