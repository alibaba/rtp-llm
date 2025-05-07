cc_test(
    name = "lookup_base_estimator_test",
    srcs = [
        "LookupMapImplTest.cpp",
    ],
    deps = [
        "//maga_transformer/cpp/disaggregate/rtpllm_master/estimator",
        "//maga_transformer/cpp/devices/testing:device_test_utils",
    ],
    data = [
        "testdata/config.json"
    ],
    copts = ["-fno-access-control",],
    exec_properties = {'gpu':'A10'},
    linkstatic = True,
)