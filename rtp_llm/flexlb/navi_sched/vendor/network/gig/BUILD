package(default_visibility = ["//rtp_llm/flexlb/navi_sched/vendor/network/gig:__subpackages__"])
load("//rtp_llm/flexlb/navi_sched/vendor/bazel:defs.bzl", "cc_proto")

cc_library (
    name = "multi_call",
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface:interface_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service:service_public_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/rpc:rpc_public_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/arpc:arpc_public_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/agent:agent_public_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/config:config_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common:common_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream:stream_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/subscribe:subscribe_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/metric:metric_headers",
        ":common_proto_cc_proto_headers",
        ":jni_proto_cc_proto_headers",
        "@grpc//:grpc++",
    # ],
    # implementation_deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface:interface",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service:service",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream:stream",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/subscribe:subscribe",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/metric:metric",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/agent:agent",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/rpc:rpc",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/new_heartbeat:new_heartbeat",
    ],
    copts = [
	"-Werror",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,  # for opensearch plugin
)



cc_proto(
    name = "heartbeat_proto",
    srcs = [
        "multi_call/proto/Heartbeat.proto",
    ],
    deps = ["//rtp_llm/flexlb/navi_sched/vendor/network/arpc:proto"],
)

cc_proto(
    name = "new_heartbeat_proto",
    srcs = [
        "multi_call/proto/NewHeartbeat.proto",
    ],
    deps = [
        ":agent_proto",
    ],
)

cc_proto(
    name = "agent_proto",
    srcs = [
        "multi_call/proto/GigAgent.proto",
    ],
    deps = [
        ":common_proto",
    ],
)

cc_proto(
    name = "jni_proto",
    srcs = [
        "multi_call/proto/GigSelectProto.proto",
        "multi_call/proto/GigCallProto.proto",
    ],
    deps = [
        ":common_proto",
    ],
)

cc_proto(
    name = "stream_proto",
    srcs = [
        "multi_call/proto/GigStreamHeader.proto",
    ],
    deps = [],
)

cc_proto(
    name = "common_proto",
    srcs = [
        "multi_call/proto/GigErrorCodeProto.proto",
        "multi_call/proto/GigCommonProto.proto",
    ],
    deps = [],
)

