cc_library(
    name = "http_server",
    srcs = glob(["http_server/*.cpp"]),
    hdrs = glob(["http_server/*.h"]),
    include_prefix = "http_server",
    strip_include_prefix = "http_server",
    visibility = ["//visibility:public"],
    deps = [
        "@havenask//aios/autil:log",
        "@havenask//aios/autil:string_helper",
        "@havenask//aios/autil:lock_free",
        "@havenask//aios/autil:json",
        "@havenask//aios/network/anet",
    ],
)
cc_library(
    name = "http_client",
    srcs = glob(["http_client/*.cpp"]),
    hdrs = glob(["http_client/*.h"]),
    include_prefix = "http_client",
    strip_include_prefix = "http_client",
    visibility = ["//visibility:public"],
    deps = [
        "@havenask//aios/autil:log",
        "@havenask//aios/network/anet",
        "//rtp_llm/cpp/utils:utils",
    ],
)

cc_test(
    name = "http_client_gtest",
    srcs = [
        "test/main.cpp",
        "test/HttpClientTest.cpp",
        "test/ConnectionPoolTest.cpp"
    ],
    deps = [
        ":http_client",
        ":http_server",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@havenask//aios/autil:net",
        "//rtp_llm/cpp:dataclass",
    ],
    copts = ["-fno-access-control",],
    linkstatic = 1,
)
