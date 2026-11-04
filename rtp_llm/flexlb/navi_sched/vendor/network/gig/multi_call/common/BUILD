package(default_visibility = ["//rtp_llm/flexlb/navi_sched/vendor/network/gig:__subpackages__"])

cc_library(
    name = "common_headers",
    hdrs = [
        "common.h",
        "ErrorCode.h",
        "VersionInfo.h",
        "ControllerParam.h",
        "HbInfo.h",
        "MetaEnv.h",
        "QueryResultStatistic.h",
        "Spec.h",
        "TopoInfoBuilder.h",
        "WorkerNodeInfo.h",
    ],
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig:heartbeat_proto_cc_proto_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:json",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:log",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:common_macros",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:time",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:lock",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:string_helper",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:base64",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:synchronized_queue",
        "@com_google_protobuf//:protobuf_headers",
    ],
    visibility = ["//visibility:public"]
)

cc_library(
    name = "common",
    srcs = [
        "common.cpp",
        "ErrorCode.cpp",
        "ControllerParam.cpp",
        "HbInfo.cpp",
        "MetaEnv.cpp",
        "QueryResultStatistic.cpp",
        "Spec.cpp",
        "TopoInfoBuilder.cpp",
        "WorkerNodeInfo.cpp",
    ],
    deps = [
        ":common_headers",
    # ],
    # implementation_deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util:util_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:time",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:string_helper",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:log",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:common_macros",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:json",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig:heartbeat_proto_cc_proto",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig:agent_proto_cc_proto",
        "@com_google_protobuf//:protobuf",
    ],
)

