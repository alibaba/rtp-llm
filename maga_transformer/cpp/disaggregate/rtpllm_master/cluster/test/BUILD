cc_library(
    name = "fake_server",
    deps = [
        "//maga_transformer/cpp/disaggregate/rtpllm_master/cluster:prefill_load_balancer",
        "//maga_transformer/cpp/disaggregate/load_balancer:load_balancer",
        "//maga_transformer/cpp/http_server:http_server",
        "@havenask//aios/autil:net",
    ],
    hdrs = ["FakeServer.h"],
    visibility = ["//visibility:public"]
)

cc_test(
    name = "prefill_load_balancer_test",
    srcs = [
        "PrefillLoadBalancerTest.cpp",
    ],
    deps = [
        "//maga_transformer/cpp/disaggregate/rtpllm_master/cluster:prefill_load_balancer",
        "//maga_transformer/cpp/disaggregate/load_balancer:load_balancer",
        "//maga_transformer/cpp/http_server:http_server",
        "@havenask//aios/autil:net",
        ":fake_server",
        "//src/fastertransformer/devices/testing:device_test_utils",
    ],
    copts = ["-fno-access-control",],
    exec_properties = {'gpu':'A10'},
    linkstatic = True,
)