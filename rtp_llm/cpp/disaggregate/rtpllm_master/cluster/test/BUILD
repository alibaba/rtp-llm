cc_library(
    name = "fake_server",
    deps = [
        "//rtp_llm/cpp/disaggregate/rtpllm_master/cluster:prefill_load_balancer",
        "//rtp_llm/cpp/disaggregate/load_balancer:load_balancer",
        "//rtp_llm/cpp/http_server:http_server",
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
        "//rtp_llm/cpp/disaggregate/rtpllm_master/cluster:prefill_load_balancer",
        "//rtp_llm/cpp/disaggregate/load_balancer:load_balancer",
        "//rtp_llm/cpp/http_server:http_server",
        "@havenask//aios/autil:net",
        ":fake_server",
        "//rtp_llm/cpp/devices/testing:device_test_utils",
    ],
    copts = ["-fno-access-control",],
    exec_properties = {'gpu':'A10'},
    linkstatic = True,
)