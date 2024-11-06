load("//bazel:arch_select.bzl", "subscribe_deps")
subscribe_deps()

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
        ":subscribe_deps"
    ],
    visibility=['//visibility:public'],
)