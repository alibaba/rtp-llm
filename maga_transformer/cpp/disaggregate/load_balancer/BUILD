load("//bazel:arch_select.bzl", "subscribe_deps")
subscribe_deps()

cc_library(
    name = "load_balancer",
    srcs = [
        "BaseLoadBalancer.cpp",
        "RRLoadBalancer.cpp",
        "WRRLoadBalancer.cpp",
    ],
    hdrs = [
        "BaseLoadBalancer.h",
        "RRLoadBalancer.h",
        "WRRLoadBalancer.h",
    ],
    deps = [
        "@havenask//aios/autil:log",
        "@havenask//aios/autil:thread",
        ":subscribe_deps",
        "//maga_transformer/cpp/http_server:http_client",
        "//maga_transformer/cpp:dataclass",
    ],
    visibility=['//visibility:public'],
)