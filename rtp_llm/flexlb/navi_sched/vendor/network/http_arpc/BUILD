
cc_library(
    name = "http_arpc",
    srcs = glob([
        "HTTPRPCServer.cpp",
        "HTTPRPCServerAdapter.cpp",
        "HTTPRPCServerWorkItem.cpp",
        "HTTPRPCController.cpp",
        "HTTPRPCServerClosure.cpp",
        "ProtoJsonizer.cpp",
        "Log.cpp",
        "HTTPANetApp.cpp",
        "SimpleProtoJsonizer.cpp",
        "DefaultUriParser.cpp",
        "*.h",
    ]),
    hdrs = glob([
        "http_arpc/*.h",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/autil:hash_func",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:json",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:lock",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:string_helper",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:url_util",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:zlib",
        "//rtp_llm/flexlb/navi_sched/vendor/network/anet/metric:anet_metric",
        "//rtp_llm/flexlb/navi_sched/vendor/network/arpc",
        "//rtp_llm/flexlb/navi_sched/vendor/network/http_arpc/metric:http_arpc_metric",
    ],
)

## unittest


