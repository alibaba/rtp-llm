cc_test(
    name = "remote_tokenize_service_test",
    srcs = [
        "RemoteTokenizeModuleTest.cpp",
    ],
    deps = [
        "//rtp_llm/cpp/disaggregate/rtpllm_master/estimator",
        "//rtp_llm/cpp/disaggregate/rtpllm_master/tokenize:tokenize_service",
        "//rtp_llm/cpp/disaggregate/rtpllm_master/cluster/test:fake_server",
        "//rtp_llm/cpp/devices/testing:device_test_utils",
    ],
    copts = ["-fno-access-control",],
    exec_properties = {'gpu':'A10'},
    linkstatic = True,
)