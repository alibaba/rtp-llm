package(default_visibility = ["//rtp_llm/flexlb/navi_sched/vendor/network/gig:__subpackages__"])

public_headers = [
    "GigStreamBase.h",
    "GigClientStream.h",
    "GigServerStream.h",
    "GigServerStreamCreator.h",
    "GigStreamMessage.h",
    "GigStreamRpcInfo.h",
]

cc_library(
    name = "stream_headers",
    hdrs = public_headers,
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/autil:fmt_string_util",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig:stream_proto_cc_proto_headers",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "stream_inner_headers",
    hdrs = glob(
        [
            "*.h",
        ],
        exclude = public_headers,
    ),
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/grpc:grpc_headers",
    ],
)

cc_library(
    name = "stream",
    srcs = glob([
        "*.cpp",
    ]),
    deps = [
        ":stream_headers",
        "@grpc//:grpc++",
        # ],
        # implementation_deps = [
        ":stream_inner_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig:stream_proto_cc_proto",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service:service_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/kmonitor:kmonitor_client_cpp",
        "//rtp_llm/flexlb/navi_sched/vendor/network/arpc",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:object_tracer",
    ],
)
