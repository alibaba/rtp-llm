package(default_visibility = ["//visibility:public"])

cc_library(
    name = "pp_cache_collector_state",
    srcs = ["PPCacheCollectorState.cpp"],
    hdrs = ["PPCacheCollectorState.h"],
)

cc_test(
    name = "pp_cache_collector_state_test",
    srcs = ["PPCacheCollectorStateTest.cpp"],
    deps = [
        ":pp_cache_collector_state",
        "//aios/unittest_framework",
    ],
)

cc_library(
    name = "pp_cache_collector_data",
    srcs = ["PPCacheShardUpdateData.cpp"],
    hdrs = ["PPCacheShardUpdateData.h"],
    deps = [
        ":pp_cache_collector_state",
        "//aios/navi:navi",
        "//aios/autil:json",
    ],
    alwayslink = True,
)

cc_test(
    name = "pp_cache_collector_data_test",
    srcs = ["PPCacheShardUpdateDataTest.cpp"],
    deps = [
        ":pp_cache_collector_data",
        "//aios/unittest_framework",
    ],
)

cc_library(
    name = "pp_cache_collector_report",
    srcs = ["PPCacheShardReportR.cpp", "PPCacheShardSourceKernel.cpp", "PPRequestFinishedSinkKernel.cpp"],
    hdrs = ["PPCacheShardReportR.h"],
    deps = [
        ":pp_cache_collector_data",
        "//aios/navi:navi",
    ],
    alwayslink = True,
)

cc_test(
    name = "pp_cache_collector_report_test",
    srcs = ["PPCacheShardReportRTest.cpp"],
    deps = [
        ":pp_cache_collector_report",
        "//aios/unittest_framework",
    ],
)

cc_test(
    name = "pp_cache_collector_source_test",
    srcs = ["PPCacheShardSourceKernelTest.cpp"],
    deps = [
        ":pp_cache_collector_report",
        "//aios/navi:navi_ops_testlib",
        "//aios/unittest_framework",
    ],
)

cc_library(
    name = "pp_cache_collector_role",
    srcs = ["NaviSchedCollector.cpp"],
    hdrs = ["NaviSchedCollector.h"],
    deps = [
        ":pp_cache_resident_state",
        ":pp_cache_collector_report",
        "//aios/navi_sched:navi_sched_role_lib",
    ],
)

cc_test(
    name = "pp_cache_collector_role_test",
    srcs = ["NaviSchedCollectorTest.cpp"],
    deps = [
        ":pp_cache_collector_role",
        "//aios/unittest_framework",
    ],
)

cc_library(
    name = "pp_cache_collector_merge",
    srcs = ["PPCacheMergeR.cpp", "PPCacheMergeKernel.cpp"],
    hdrs = ["PPCacheMergeR.h"],
    deps = [
        ":pp_cache_collector_data",
        "//aios/navi:navi",
    ],
    alwayslink = True,
)

cc_test(
    name = "pp_cache_collector_merge_test",
    srcs = ["PPCacheMergeTest.cpp"],
    deps = [
        ":pp_cache_collector_merge",
        "//aios/navi:navi_ops_testlib",
        "//aios/unittest_framework",
    ],
)

cc_library(
    name = "pp_cache_collect_graph",
    srcs = ["PPCacheCollectGraph.cpp"],
    hdrs = ["PPCacheCollectGraph.h", "PPCacheAgentConfig.h"],
    deps = [
        "//aios/navi_sched:navi_sched_data_lib",
        ":pp_cache_collector_merge",
        ":pp_request_finished",
        "//aios/navi:navi",
    ],
)

cc_test(
    name = "pp_cache_collect_graph_test",
    srcs = ["PPCacheCollectGraphTest.cpp"],
    deps = [
        ":pp_cache_collect_graph",
        ":pp_cache_collector_role",
        "//aios/unittest_framework",
    ],
)

cc_library(
    name = "pp_cache_resident_state",
    srcs = ["PPCacheResidentState.cpp"],
    hdrs = ["PPCacheResidentState.h"],
    deps = [
        ":pp_cache_collector_state",
        "@rapidjson//:rapidjson",
    ],
)

cc_test(
    name = "pp_cache_resident_state_test",
    srcs = ["PPCacheResidentStateTest.cpp"],
    deps = [
        ":pp_cache_resident_state",
        "//aios/unittest_framework",
    ],
)

py_test(
    name = "pp_cache_collector_observer_test",
    srcs = ["V6DCollectorObserverTest.py"],
    main = "V6DCollectorObserverTest.py",
    deps = ["//aios/navi_sched:py_sched_lib"],
    python_version = "PY3",
)

cc_test(
    name = "pp_cache_agent_integration_test",
    srcs = ["PPCacheAgentIntegrationTest.cpp"],
    deps = [
        ":pp_cache_collector_role",
        ":pp_cache_collect_graph",
        "//aios/unittest_framework",
    ],
)

py_test(
    name = "pp_cache_collector_runtime_test",
    srcs = ["V6DPPCollectorRuntimeTest.py", "V6DCollectorObserverTest.py"],
    main = "V6DPPCollectorRuntimeTest.py",
    imports = ["."],
    deps = ["//aios/navi_sched:py_sched_lib"],
    python_version = "PY3",
)

cc_library(
    name = "pp_request_finished",
    srcs = ["PPRequestFinishedR.cpp", "PPRequestFinishedSourceKernel.cpp"],
    hdrs = ["PPRequestFinishedR.h"],
    deps = [
        ":pp_cache_collector_data",
        "//aios/navi:navi",
    ],
    alwayslink = True,
)

cc_test(
    name = "pp_request_finished_test",
    srcs = ["PPRequestFinishedRTest.cpp"],
    deps = [
        ":pp_request_finished",
        "//aios/unittest_framework",
    ],
)

cc_test(
    name = "pp_request_finished_kernel_test",
    srcs = ["PPRequestFinishedSourceKernelTest.cpp"],
    deps = [
        ":pp_request_finished",
        "//aios/navi:navi_ops_testlib",
        "//aios/unittest_framework",
    ],
)

cc_test(
    name = "pp_request_finished_sink_test",
    srcs = ["PPRequestFinishedSinkKernelTest.cpp"],
    deps = [
        ":pp_cache_collector_report",
        "//aios/navi:navi_ops_testlib",
        "//aios/unittest_framework",
    ],
)
