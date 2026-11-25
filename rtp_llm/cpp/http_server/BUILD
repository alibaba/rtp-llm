load("//:def.bzl", "cc_test_wrapper")

# Generic HTTP transport. Request handlers are supplied by the caller.
cc_library(
    name = "http_server",
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    visibility = ["//visibility:public"],
    deps = [
        "@havenask//aios/autil:lock_free",
        "@havenask//aios/autil:log",
        "@havenask//aios/autil:string_helper",
        "@havenask//aios/autil:time",
        "@havenask//aios/network/anet:anet",
    ],
)

cc_test_wrapper(
    name = "http_server_test",
    srcs = ["test/HttpServerTest.cpp"],
    deps = [
        ":http_server",
        "@havenask//aios/autil:net",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
