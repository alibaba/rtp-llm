cc_binary(
    name = "rr_load_balancer_main",
    srcs = [
        "RRLoadBalancerMain.cpp",
    ],
    deps = [
        "//maga_transformer/cpp/disaggregate/load_balancer:load_balancer",
    ],
    linkstatic = True,
)

cc_test(
    name = "wrr_load_balancer_test",
    srcs = [
        "main.cpp",
        "WRRLoadBalancerTest.cpp",
    ],
    deps = [
        "//maga_transformer/cpp/disaggregate/load_balancer:load_balancer",
        "//maga_transformer/cpp/http_server:http_server",
        "@havenask//aios/autil:net",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
    copts = ["-fno-access-control",],
    linkstatic = True,
)