load('//rtp_llm/flexlb/navi_sched/vendor/bazel:defs.bzl', 'cc_and_py_proto')

cc_and_py_proto(
    name = "proto",
    srcs = [
        "arpc/proto/msg_header.proto",
        "arpc/proto/rpc_extensions.proto",
    ],
    deps = [
    ],
    visibility = ["//visibility:public"],
    import_prefix = "arpc",
    strip_import_prefix = "arpc",
)

cc_library(
    name = "arpc",
    visibility = ["//visibility:public"],
    srcs = [
        "arpc/DataBufferInputStream.cpp",
        "arpc/DataBufferOutputStream.cpp",
        "arpc/MessageSerializable.cpp",
        "arpc/RPCServerAdapter.cpp",
        "arpc/UtilFun.cpp",
        "arpc/RPCServerWorkItem.cpp",
        "arpc/RPCChannelBase.cpp",
        "arpc/SyncClosure.cpp",
        "arpc/RPCMessageSerializable.cpp",
        "arpc/RPCRawMessageSerializable.cpp",
        "arpc/RPCServer.cpp",
        "arpc/RPCChannelManagerBase.cpp",
        "arpc/MessageCodec.cpp",
        "arpc/RPCServerClosure.cpp",
        "arpc/Tracer.cpp",
        "arpc/RPCServerList.cpp",
    ] + [
        "arpc/anet/ANetApp.cpp",
        "arpc/anet/ANetRPCMessageCodec.cpp",
        "arpc/anet/ANetRPCServerAdapter.cpp",
        "arpc/anet/ANetRPCServerClosure.cpp",
        "arpc/anet/ANetRPCServerWorkItem.cpp",
        "arpc/anet/ClientPacketHandler.cpp",
        "arpc/anet/SharedClientPacketHandler.cpp",
        "arpc/ANetRPCServer.cpp",
        "arpc/ANetRPCController.cpp",
        "arpc/ANetRPCChannel.cpp",
        "arpc/ANetRPCChannelManager.cpp",
        "arpc/PooledChannelManager.cpp",
    ] + glob([
        "arpc/common/*.cpp",
        "arpc/util/*.cpp",
        "arpc/anet/*.h",
    ]),
    hdrs = glob([
        "arpc/*.h",
        "arpc/anet/*.h",
        "arpc/util/*.h",
        "arpc/common/*.h",
    ]),
    deps = [
        ":proto_cc_proto",
        "//rtp_llm/flexlb/navi_sched/vendor/network/anet:anet",
        "//rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/metric:arpc_metric",
        "//rtp_llm/flexlb/navi_sched/vendor/alog:alog",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:NoCopyable",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:thread",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:time",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:lock",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:env_util",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:lock_free",
    ],
    copts = [
        "-Wformat-truncation=0",
    ],
    alwayslink = True,
)

# -----------------------------------------------------------------------------
# arpc/arpc/test












