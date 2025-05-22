cc_library(
    name = "vipserver",
    hdrs = glob([
        "include/*",
    ]),
    srcs = glob([
        "src/**/*.h",
        "src/**/*.hpp",
        "src/**/*.cpp",
    ]),
    implementation_deps = [
        "@rtp_llm//3rdparty/easy:easy",
        "@havenask//aios/alog:alog",
        "@boost//:headers-base",
        "@boost//:interprocess",
        "@boost//:date_time",
        "@boost//:property_tree",
        "@jsoncpp_git//:jsoncpp",
        "@curl//:curl",
    ],
    includes = ["src"],
    copts = [
        "-Iexternal/vipserver/deps/include",
        "-DEASY_MULTIPLICITY",
    ],
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "vipserver_headers",
    hdrs = glob([
        "include/*",
    ]),
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)