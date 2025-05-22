load("//bazel:arch_select.bzl", "subscribe_deps")
subscribe_deps()

cc_library(
    name = "load_balancer",
    srcs = [
        "BaseLoadBalancer.cpp",
        "RRLoadBalancer.cpp",
        "WRRLoadBalancer.cpp",
        "HeartbeatSynchronizer.cpp",
        "WorkerAwaredLoadBalancer.cpp"
    ],
    hdrs = [
        "BaseLoadBalancer.h",
        "RRLoadBalancer.h",
        "WRRLoadBalancer.h",
        "WorkerAwaredLoadBalancer.h",
        "HeartbeatSynchronizer.h",
    ],
    deps = [
        "@havenask//aios/autil:log",
        "@havenask//aios/autil:thread",
        ":subscribe_deps",
        "//rtp_llm/cpp/http_server:http_client",
        "//rtp_llm/cpp:dataclass",
    ],
    visibility=['//visibility:public'],
)