package(default_visibility = ["//rtp_llm/flexlb/navi_sched/vendor/navi:__subpackages__"])

load('//rtp_llm/flexlb/navi_sched/vendor/bazel:defs.bzl', 'cc_proto')
load(":defs.bzl", "cc_proto_custom")




cc_library(
    name = "navi",
    data = [
        ":navi_python_home_scripts",
    ],
    implementation_deps = [
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":navi_headers",
        ":navi_inner_lib",
        ":navi_proto_cc_proto",
        "//rtp_llm/flexlb/navi_sched/vendor/kmonitor:kmonitor_client_cpp",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig:multi_call",
        "//rtp_llm/flexlb/navi_sched/vendor/network/http_arpc",
    ],
)

cc_library(
    name = "navi_headers",
    hdrs = glob([
        "common.h",
        "builder/*.h",
        "config/*.h",
        "engine/AsyncPipe.h",
        "engine/Data.h",
        "engine/CreatorRegistry.h",
        "engine/CreatorStats.h",
        "engine/GraphMetric.h",
        "engine/Kernel.h",
        "engine/KernelComputeContext.h",
        "engine/KernelConfigContext.h",
        "engine/KernelCreator.h",
        "engine/KernelInitContext.h",
        "engine/KernelMetric.h",
        "engine/ScheduleInfo.h",
        "engine/MapKernel.h",
        "engine/N2OneKernel.h",
        "engine/NamedData.h",
        "engine/Navi.h",
        "engine/NaviSnapshotStat.h",
        "engine/NaviConfigContext.h",
        "engine/NaviResult.h",
        "engine/NaviUserResult.h",
        "engine/PipeKernel.h",
        "engine/PortMergeKernel.h",
        "engine/PortSplitKernel.h",
        "engine/ScopeTerminatorKernel.h",
        "engine/Resource.h",
        "engine/ResourceConfigContext.h",
        "engine/ResourceCreator.h",
        "engine/ResourceInitContext.h",
        "engine/ResourceMap.h",
        "engine/CollectFlags.h",
        "engine/RunGraphParams.h",
        "engine/TimeoutChecker.h",
        "engine/Type.h",
        "engine/TypeContext.h",
        "engine/TypeInfo.h",
        "engine/PartInfo.h",
        "log/LoggingEvent.h",
        "log/NaviLogger.h",
        "log/NaviLoggerProvider.h",
        "log/TraceCollector.h",
        "log/LogBtFilter.h",
        "perf/NaviPerfResult.h",
        "resource/*.h",
        "ops/DefaultMergeKernel.h",
        "ops/DefaultSplitKernel.h",
        "rpc_server/NaviRpcServerR.h",
        "rpc_server/NaviArpcRequestData.h",
        "rpc_server/NaviArpcResponseData.h",
        "util/NaviTestPool.h",
        "util/NaviClosure.h",
        "util/ReadyBitMap.h",
        "util/BitMap.h",
        "util/EnvParam.h",
        "test_cluster/NaviTestCluster.h",
        "test_cluster/NaviFuncClosure.h",
    ]),
    include_prefix = "navi",
    deps = [
        ":navi_proto_cc_proto_headers",
        ":navi_builder_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:object_tracer",
        "//rtp_llm/flexlb/navi_sched/vendor/network/anet:anet",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:json",
    ] + select({
        "//:aios_open_source": [],
        "//conditions:default": [
            "//rtp_llm/flexlb/navi_sched/vendor/third_party/kernel_headers:kernel_headers",
        ],
    }),
)


cc_library(
    name = "navi_engine_testlib",
    testonly = False,
    hdrs = [
        "engine/test/MockAsyncPipe.h",
    ],
    include_prefix = "navi",
    visibility = ["//visibility:public"],
    deps = [
        ":navi",
        "//rtp_llm/flexlb/navi_sched/vendor/unittest_framework:unittest_lib",
    ],
)


cc_proto_custom(
    name = "navi_proto_peer_info",
    srcs = glob([
        "proto/PeerInfo.proto",
    ]),
    import_prefix = "navi",
    deps = [],
)

cc_proto(
    name = "navi_proto",
    srcs = glob([
        "proto/GraphDef.proto",
        "proto/GraphVis.proto",
        "proto/KernelDef.proto",
        "proto/NaviRpc.proto",
        "proto/GraphState.proto",
        "proto/NaviState.proto",
        "proto/NaviSummary.proto",
        "proto/NaviConfig.proto",
    ]),
    import_prefix = "navi",
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/network/arpc:proto",
        ":navi_proto_peer_info"
    ],
)

cc_proto(
    name = "navi_proto_inner",
    srcs = glob([
        "proto/NaviStream.proto",
    ]),
    import_prefix = "navi",
    deps = [
        ":navi_proto",
    ],
)

cc_library(
    name = "navi_builder_headers",
    hdrs = glob(
        [
            "builder/GraphBuilder.h",
            "builder/GraphDesc.h",
            "builder/BorderId.h",
            "builder/KernelDefBuilder.h",
            "builder/ResourceDefBuilder.h",
        ],
        exclude = [
        ],
    ),
    copts = [
        "-Werror",
        "-Wno-aligned-new",
    ],
    implementation_deps = [
    ],
    include_prefix = "navi",
    linkopts = [
        "-L/usr/lib64",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":navi_common",
        ":navi_proto_inner_cc",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:json",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig:multi_call",
    ],
    alwayslink = True,
)

cc_library(
    name = "navi_builder_inner",
    srcs = glob(
        [
            "builder/*.cpp",
        ],
        exclude = [
        ],
    ),
    hdrs = glob(
        [
            "builder/*.h",
        ],
        exclude = [
        ],
    ),
    copts = [
        "-Werror",
        "-Wno-aligned-new",
    ],
    include_prefix = "navi",
    linkopts = [
        "-L/usr/lib64",
    ],
    deps = [
        ":navi_common",
        ":navi_headers",
        ":navi_proto_peer_info_cc_proto_headers",
        ":navi_proto_cc_proto_headers",
        ":navi_proto_inner_cc_proto_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:json",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig:multi_call",
    ],
    alwayslink = True,
)

cc_library(
    name = "navi_common",
    srcs = [
        "common.cpp",
    ],
    hdrs = [
        "common.h",
        "util/NaviTestPool.h",
    ],
    copts = [
        "-Werror",
        "-Wno-aligned-new",
    ],
    include_prefix = "navi",
    linkopts = [
        "-L/usr/lib64",
    ],
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/autil:mem_pool_base",
        "//rtp_llm/flexlb/navi_sched/vendor/network/anet",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
    alwayslink = True,
)

cc_library(
    name = "navi_inner_lib",
    srcs = glob(
        [
            "log/*.cpp",
            "util/*.cpp",
            "config/*.cpp",
            "engine/*.cpp",
            "resource/*.cpp",
            "ops/*.cpp",
            "config_loader/*.cpp",
            "distribute/*.cpp",
            "perf/*.cpp",
            "memleak/*.cpp",
            "rpc_server/*.cpp",
            "test_cluster/*.cpp",
        ],
        exclude = [
            "test_cluster/NaviGraphRunner.cpp",
        ],
    ),
    hdrs = glob(
        [
            "engine/*.h",
            "log/*.h",
            "ops/*.h",
            "util/*.h",
            "resource/*.h",
            "config/*.h",
            "config_loader/*.h",
            "distribute/*.h",
            "perf/*.h",
            "memleak/*.h",
            "rpc_server/*.h",
            "test_cluster/*.h",
            "test_cluster/*.h",
        ],
        exclude = [
            "test_cluster/NaviGraphRunner.h",
        ],
    ),
    copts = [
        "-Werror",
        "-Wno-aligned-new",
    ],
    include_prefix = "navi",
    deps = [
        ":navi_builder_headers",
        ":navi_builder_inner",
        ":navi_common",
        ":navi_headers",
        ":navi_proto_inner_cc",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:compression",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:const_string_util",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:diagnostic",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:file_util",
        "//rtp_llm/flexlb/navi_sched/vendor/kmonitor:kmonitor_client_cpp",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig:multi_call",
        "//rtp_llm/flexlb/navi_sched/vendor/network/http_arpc",
        "//rtp_llm/flexlb/navi_sched/vendor/third_party/elfutils-libelf",
        "//rtp_llm/flexlb/navi_sched/vendor/third_party/libev",
        "//rtp_llm/flexlb/navi_sched/vendor/third_party/tbb",
        "@local_config_python//:python_headers",
        "@local_config_python//:python_lib",
    ],
    alwayslink = True,
)

cc_library(
    name = "navi_runtime_config",
    hdrs = ["pybind/NaviRuntimeConfig.h"],
    include_prefix = "navi",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "navi_pybind",
    srcs = glob(
        [
            "pybind/*.cpp",
        ],
    ),
    hdrs = glob(
        [
            "pybind/*.h",
        ],
    ),
    copts = [
        "-Werror",
        "-Wno-aligned-new",
    ],
    include_prefix = "navi",
    deps = [
        ":navi",
        ":navi_runtime_config",
        "@pybind11//:pybind11",
    ],
    alwayslink = True,
    visibility = ["//visibility:public"],
)






cc_library(
    name = "navi_ops_testlib",
    testonly = False,
    srcs = glob([
        "tester/*.cpp",
    ]),
    hdrs = glob([
        "tester/*.h",
    ]),
    include_prefix = "navi",
    visibility = ["//visibility:public"],
    deps = [
        ":navi_engine_testlib",
        ":navi_inner_lib",
        "//rtp_llm/flexlb/navi_sched/vendor/kmonitor:metrics_report_navi_resource",
    ],
    alwayslink = True,
)

filegroup(
    name = "navi_python_home_scripts",
    srcs = glob([
        "config_loader/python/*.py",
    ]),
)

# TODO
# add navi ut



















