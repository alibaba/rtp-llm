cc_test(
    name = "lookup_base_estimator_test",
    srcs = [
        "LookupMapImplTest.cpp",
    ],
    deps = [
        "//rtp_llm/cpp/disaggregate/rtpllm_master/estimator",
        "//rtp_llm/cpp/devices/testing:device_test_utils",
    ],
    data = [
        "testdata/config.json"
    ],
    copts = ["-fno-access-control",],
    exec_properties = {'gpu':'A10'},
    linkstatic = True,
)