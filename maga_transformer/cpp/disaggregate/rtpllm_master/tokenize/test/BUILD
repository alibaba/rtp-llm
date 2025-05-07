cc_test(
    name = "remote_tokenize_service_test",
    srcs = [
        "RemoteTokenizeModuleTest.cpp",
    ],
    deps = [
        "//maga_transformer/cpp/disaggregate/rtpllm_master/estimator",
        "//maga_transformer/cpp/disaggregate/rtpllm_master/tokenize:tokenize_service",
        "//maga_transformer/cpp/disaggregate/rtpllm_master/cluster/test:fake_server",
        "//maga_transformer/cpp/devices/testing:device_test_utils",
    ],
    copts = ["-fno-access-control",],
    exec_properties = {'gpu':'A10'},
    linkstatic = True,
)