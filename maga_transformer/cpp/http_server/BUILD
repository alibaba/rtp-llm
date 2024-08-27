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
    name = "http_server_example_lib",
    srcs = glob(["http_server/example/*.cpp"]),
    hdrs = glob(["http_server/example/*.h"]),
    include_prefix = "http_server",
    deps = [":http_server"],
)

cc_binary(
    name = "http_server_example",
    deps = [":http_server_example_lib"],
)

cc_test(
    name = "http_server_unittest",
    srcs = glob([
        "http_server/unittest/*.cpp",
        "http_server/unittest/*.h",
        "http_server/unittest/mock/*.cpp",
        "http_server/unittest/mock/*.h",
    ]),
    copts = ["-fno-access-control"],
    deps = [
        ":http_server",
        "@havenask//aios/autil:net",
        "@havenask//aios/unittest_framework",
    ],
)
