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
        "@havenask//aios/alog:alog",
        "@boost//:headers-base",
        "@boost//:interprocess",
        "@boost//:date_time",
        "@boost//:property_tree",
        "@jsoncpp_git//:jsoncpp",
        "@curl//:curl",
    ] + select({
        "@platforms//cpu:aarch64": [
            "@easy_for_vipserver_aarch64//:libeasy",
            "@easy_for_vipserver_aarch64//:libeasy_headers",
        ],
        "@platforms//cpu:x86_64": [
            "@easy_for_vipserver_x86_64//:libeasy",
            "@easy_for_vipserver_x86_64//:libeasy_headers",
        ],
        "//conditions:default": [
            "@easy_for_vipserver_x86_64//:libeasy",
            "@easy_for_vipserver_x86_64//:libeasy_headers",
        ],
    }),
    includes = ["src"],
    copts = select({
        "@platforms//cpu:aarch64": [
            "-DEASY_MULTIPLICITY",
        ],
        "@platforms//cpu:x86_64": [],
        "//conditions:default": [],
    }),
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