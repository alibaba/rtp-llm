package(default_visibility = ["//visibility:public"])
exports_files(["test/VipSubscriptionTest.cpp"])
load("//rtp_llm/flexlb/navi_sched/vendor/third_party/pybind11_bazel:build_defs.bzl", "pybind_extension")
load("//rtp_llm/flexlb/navi_sched/vendor/bazel:defs.bzl", "cc_proto")
load("//rtp_llm/flexlb/navi_sched/vendor/bazel:git_version.bzl", "git_version_library")

git_version_library(
    name = "navi_sched_git_version",
    namespace = "navi_sched_build",
)

cc_proto(
    name = "navi_sched_proto",
    srcs = [
        "proto/ReplayControl.proto",
        "proto/RouteMode.proto",
        "proto/SchedulerConfig.proto",
    ],
    import_prefix = "rtp_llm/flexlb/navi_sched",
    deps = ["//rtp_llm/flexlb/navi_sched/vendor/network/arpc:proto"],
)

cc_library(
    name = "navi_sched_data_lib",
    srcs = ["common/NaviSchedCommon.cpp"] + glob([
        "data/*.cpp",
    ], exclude = [
        "data/*Test.cpp",
    ]),
    hdrs = [
        "common/NaviSchedCommon.h",
        "common/NaviSchedConstants.h",
        "common/NaviSchedScheduleData.h",
    ] + glob([
        "data/*.h",
    ]),
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/navi:navi",
        "//rtp_llm/flexlb/navi_sched/vendor/navi:navi_headers",
        "//rtp_llm/flexlb/navi_sched/vendor/navi:navi_runtime_config",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:hash_util",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:json",
    ],
    alwayslink = True,
    visibility = ["//rtp_llm/flexlb/navi_sched:__subpackages__"],
)

cc_library(
    name = "navi_sched_replay_debug_lib",
    srcs = ["debug/NaviSchedScheduleReplayJson.cpp"],
    hdrs = ["debug/NaviSchedScheduleReplayJson.h"],
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/autil:json",
        ":navi_sched_data_lib",
    ],
    visibility = ["//rtp_llm/flexlb/navi_sched:__subpackages__"],
)

cc_binary(
    name = "traffic_replay_log_converter",
    testonly = True,
    srcs = ["test/simulator/TrafficReplayLogConverter.cpp"],
    deps = [
        "@rapidjson//:rapidjson",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:string_helper",
    ],
)

cc_binary(
    name = "latency_online_replay",
    testonly = True,
    srcs = [
        "resource/AgentSchedulerConfigR.h",
        "resource/LatencyHistorySampler.cpp",
        "resource/LatencyHistorySampler.h",
        "resource/LatencyResidualModel.cpp",
        "resource/LatencyResidualModel.h",
        "resource/NonLinearPrefillModelR.cpp",
        "resource/NonLinearPrefillModelR.h",
        "scheduler/AutoGrad.h",
        "test/LatencyOnlineReplay.cpp",
    ],
    copts = ["-fno-access-control"],
    linkopts = ["-Wl,-rpath,$$ORIGIN/navi_sched"],
    deps = [
        "@rapidjson//:rapidjson",
        "//rtp_llm/flexlb/navi_sched/vendor/kmonitor:kmonitor_client_cpp",
        "//rtp_llm/flexlb/navi_sched/vendor/kmonitor:metrics_report_navi_resource",
        "//rtp_llm/flexlb/navi_sched/vendor/navi:navi",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig:multi_call",
        ":navi_sched_data_lib",
    ],
)

cc_binary(
    name = "latency_formula_experiment",
    testonly = True,
    srcs = [
        "resource/AgentSchedulerConfigR.h",
        "resource/LatencyHistorySampler.cpp",
        "resource/LatencyHistorySampler.h",
        "resource/LatencyResidualModel.cpp",
        "resource/LatencyResidualModel.h",
        "resource/NonLinearPrefillModelR.cpp",
        "resource/NonLinearPrefillModelR.h",
        "scheduler/AutoGrad.h",
        "test/LatencyFormulaExperiment.cpp",
    ],
    copts = ["-fno-access-control"],
    linkopts = ["-Wl,-rpath,$$ORIGIN/navi_sched"],
    deps = [
        "@rapidjson//:rapidjson",
        "//rtp_llm/flexlb/navi_sched/vendor/kmonitor:kmonitor_client_cpp",
        "//rtp_llm/flexlb/navi_sched/vendor/kmonitor:metrics_report_navi_resource",
        "//rtp_llm/flexlb/navi_sched/vendor/navi:navi",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig:multi_call",
        ":navi_sched_data_lib",
    ],
)

cc_binary(
    name = "latency_unified_feature_experiment",
    testonly = True,
    srcs = [
        "resource/AgentSchedulerConfigR.h",
        "resource/LatencyHistorySampler.cpp",
        "resource/LatencyHistorySampler.h",
        "resource/LatencyResidualModel.cpp",
        "resource/LatencyResidualModel.h",
        "resource/NonLinearPrefillModelR.cpp",
        "resource/NonLinearPrefillModelR.h",
        "scheduler/AutoGrad.h",
        "test/LatencyUnifiedFeatureExperiment.cpp",
    ],
    copts = ["-fno-access-control"],
    deps = [
        "@rapidjson//:rapidjson",
        "//rtp_llm/flexlb/navi_sched/vendor/kmonitor:kmonitor_client_cpp",
        "//rtp_llm/flexlb/navi_sched/vendor/kmonitor:metrics_report_navi_resource",
        "//rtp_llm/flexlb/navi_sched/vendor/navi:navi",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig:multi_call",
        ":navi_sched_data_lib",
    ],
)

cc_library(
    name = "navi_sched_common_header",
    hdrs = [
        "common/NaviSchedCommon.h",
        "common/NaviSchedConstants.h",
    ],
    include_prefix = "navi_sched",
    strip_include_prefix = "common",
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/navi:navi",
        "//rtp_llm/flexlb/navi_sched/vendor/navi:navi_runtime_config",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "navi_sched_agent_header",
    hdrs = ["service/NaviSchedAgent.h"],
    include_prefix = "navi_sched",
    strip_include_prefix = "service",
    deps = [":navi_sched_common_header"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "navi_sched_scheduler_api",
    hdrs = [
        "scheduler/MasterScheduleContextData.h",
        "scheduler/ScheduleContext.h",
        "scheduler/SchedulerBaseKernel.h",
    ],
    deps = [
        ":navi_sched_data_lib",
        "//rtp_llm/flexlb/navi_sched/vendor/navi:navi",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "navi_sched_scheduler_extension_compile",
    srcs = [
        "resource/NonLinearPrefillModelR.h",
        "scheduler/AutoGrad.h",
        "scheduler/CostSchedulerConfig.cpp",
        "scheduler/CostSchedulerConfig.h",
        "scheduler/CostSchedulerColdStartStrategy.cpp",
        "scheduler/CostSchedulerColdStartStrategy.h",
        "scheduler/CostSchedulerKernel.cpp",
        "scheduler/FlexLBBaselineSchedulerKernel.cpp",
        "scheduler/MaxCacheHitSchedulerKernel.cpp",
        "scheduler/RandomSchedulerKernel.cpp",
        "scheduler/CostSchedulerKernelInternal.cpp",
        "scheduler/CostSchedulerKernelInternal.h",
    ],
    hdrs = [
        "scheduler/CostSchedulerKernel.h",
        "scheduler/FlexLBBaselineSchedulerKernel.h",
        "scheduler/MaxCacheHitSchedulerKernel.h",
        "scheduler/RandomSchedulerKernel.h",
    ],
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/autil:json",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig:multi_call",
        ":navi_sched_scheduler_api",
    ],
)

cc_library(
    name = "navi_sched_role_lib",
    srcs = glob([
        "adapters/**/*.cpp",
        "config/*.cpp",
        "graph/*.cpp",
        "kernel/*.cpp",
        "resource/*.cpp",
        "scheduler/*.cpp",
        "service/*.cpp",
    ], exclude = [
        "**/*Test.cpp",
    ]),
    hdrs = ["common/NaviSchedCommon.h"] + glob([
        "adapters/**/*.h",
        "config/*.h",
        "graph/*.h",
        "kernel/*.h",
        "resource/*.h",
        "scheduler/*.h",
        "service/*.h",
    ]),
    alwayslink = True,
    deps = [
        ":navi_sched_agent_header",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "//rtp_llm/flexlb/navi_sched/vendor/navi:navi",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:file_recorder",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:crc32c",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:hash_base",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:hash_map",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:json",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:lock_free",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:log",
        "//rtp_llm/flexlb/navi_sched/vendor/kmonitor:kmonitor_client_cpp",
        "//rtp_llm/flexlb/navi_sched/vendor/kmonitor:metrics_report_navi_resource",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig:multi_call",
        ":navi_sched_git_version",
        ":navi_sched_proto_cc_proto",
        ":navi_sched_data_lib",
        ":navi_sched_replay_debug_lib",
    ],
)

cc_binary(
    name = "libnavi_sched.so",
    additional_linker_inputs = ["scripts/navi_sched_cpp.lds"],
    linkshared = True,
    linkopts = [
        "-Wl,-Bsymbolic",
        "-Wl,--exclude-libs=ALL",
        "-Wl,--version-script",
        "$(location scripts/navi_sched_cpp.lds)",
    ],
    deps = [":navi_sched_role_lib"],
    visibility = ["//visibility:public"],
)

cc_import(
    name = "navi_sched_dynamic",
    hdrs = [
        "common/NaviSchedCommon.h",
        "adapters/rtp_llm/RtpLLMAdapter.h",
        "service/NaviSchedAgent.h",
    ],
    shared_library = ":libnavi_sched.so",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "navi_sched_public_api_compile",
    srcs = ["test/package/NaviSchedPublicApiCompileTest.cpp"],
    deps = [
        ":navi_sched_agent_header",
        ":navi_sched_role_lib",
    ],
)

cc_library(
    name = "rtp_llm_adapter_public_api_compile",
    srcs = ["test/package/RtpLLMAdapterPublicApiCompileTest.cpp"],
    deps = [
        ":navi_sched_role_lib",
        "//rtp_llm/flexlb/navi_sched/vendor/navi:navi_runtime_config",
    ],
)

cc_test(
    name = "navi_sched_rtp_llm_adapter_test",
    srcs = [
        "adapters/rtp_llm/RtpLLMAdapterTest.cpp",
    ],
    copts = ["-fno-access-control"],
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/unittest_framework",
        ":navi_sched_role_lib",
    ],
)

cc_test(
    name = "navi_sched_agent_response_r_test",
    srcs = ["resource/AgentResponseRTest.cpp"],
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/unittest_framework",
        ":navi_sched_role_lib",
    ],
)

cc_test(
    name = "replay_control_r_test",
    srcs = ["resource/ReplayControlRTest.cpp"],
    data = [":wheel_python_runtime"],
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/unittest_framework",
        ":navi_sched_role_lib",
    ],
)

cc_test(
    name = "latency_history_sampler_test",
    srcs = ["resource/LatencyHistorySamplerTest.cpp"],
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/unittest_framework",
        ":navi_sched_role_lib",
    ],
)

cc_test(
    name = "request_latency_tracker_test",
    srcs = [
        "common/NaviSchedConstants.h",
        "data/NaviSchedLatencyPrediction.h",
        "data/NaviSchedLatencySample.h",
        "resource/RequestLatencyTracker.cpp",
        "resource/RequestLatencyTracker.h",
        "resource/RequestLatencyTrackerTest.cpp",
    ],
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/autil:lock",
        "//rtp_llm/flexlb/navi_sched/vendor/unittest_framework",
    ],
)

cc_test(
    name = "agent_history_state_test",
    srcs = ["resource/AgentHistoryStateTest.cpp"],
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/navi:navi_ops_testlib",
        "//rtp_llm/flexlb/navi_sched/vendor/unittest_framework",
        ":navi_sched_role_lib",
    ],
)

cc_test(
    name = "navi_sched_client_response_r_test",
    srcs = ["resource/ClientResponseRTest.cpp"],
    data = [":wheel_python_runtime"],
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/unittest_framework",
        ":navi_sched_role_lib",
    ],
)

cc_test(
    name = "navi_sched_config_builder_test",
    srcs = ["test/NaviSchedConfigBuilderTest.cpp"],
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/unittest_framework",
        ":navi_sched_role_lib",
    ],
)

genrule(
    name = "wheel_navi_sched_lib",
    srcs = [":libnavi_sched.so"],
    outs = ["navi_sched/lib/libnavi_sched.so"],
    cmd = "cp $< $@",
)

genrule(
    name = "wheel_python_runtime",
    srcs = ["@python3.10_header//:lib/libpython3.10.so.1.0"],
    outs = ["navi_sched/libpython3.10.so.1.0"],
    cmd = "cp -L $< $@",
)

pybind_extension(
    name = "navi_sched/_navi_sched_core",
    additional_linker_inputs = ["scripts/navi_sched_pybind.lds"],
    srcs = glob([
        "bindings/*.cpp",
        "bindings/*.h",
    ]),
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/alog",
        "//rtp_llm/flexlb/navi_sched/vendor/navi:navi",
        "//rtp_llm/flexlb/navi_sched/vendor/navi:navi_pybind",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig:multi_call",
        ":navi_sched_replay_debug_lib",
        ":navi_sched_role_lib",
    ],
    visibility = ["//rtp_llm/flexlb/navi_sched:__subpackages__"],
    features = [
        "static_link_cpp_runtimes",
        "static_libgcc",
        "static_libcxx",
    ],
    linkopts = [
        "-Wl,-rpath,\\$$ORIGIN",
        "-Wl,-rpath,\\$$ORIGIN/lib",
        "-Wl,-rpath,/opt/conda310/lib",
        "-Wl,--version-script",
        "$(location scripts/navi_sched_pybind.lds)",
    ],
    copts = [
        "-fvisibility=hidden",
        "-fPIC",
        "-I/usr/local/include/python3",
    ],
)

py_library(
    name = "py_sched_lib",
    srcs = [
        "navi_sched_vllm_plugin.py",
        "navi_sched/__init__.py",
        "navi_sched/alog.py",
        "navi_sched/env.py",
        "navi_sched/agent.py",
        "navi_sched/adapters/__init__.py",
        "navi_sched/adapters/dashserving/__init__.py",
        "navi_sched/adapters/dashserving/navi_sched_dashserving_adapter.py",
        "navi_sched/adapters/rtp_llm/__init__.py",
        "navi_sched/adapters/rtp_llm/NaviSchedAdapterConfig.py",
        "navi_sched/adapters/rtp_llm/NaviSchedClient.py",
        "navi_sched/adapters/rtp_llm/SwitchableMasterClient.py",
        "navi_sched/adapters/rtp_llm/RtpLLMAdapterUtils.py",
        "navi_sched/adapters/rtp_llm/RtpLLMMaster.py",
        "navi_sched/adapters/rtp_llm/RtpLLMUnifiedConfig.py",
        "navi_sched/adapters/rtp_llm/ServiceDiscovery.py",
        "navi_sched/adapters/vllm/__init__.py",
        "navi_sched/adapters/vllm/NaviSchedAdapterConfig.py",
        "navi_sched/adapters/vllm/NaviSchedClient.py",
        "navi_sched/adapters/vllm/NaviSchedEndpointRouter.py",
        "navi_sched/adapters/vllm/NaviSchedProductionStackRouter.py",
        "navi_sched/adapters/vllm/NaviSchedVLLMPlugin.py",
        "navi_sched/adapters/vllm/VLLMBlockHash.py",
        "navi_sched/adapters/vllm/VLLMCachePolicy.py",
        "navi_sched/adapters/vllm/VLLMDeferredRenderer.py",
        "navi_sched/adapters/vllm/VLLMEngineObserver.py",
        "navi_sched/adapters/vllm/VLLMEngineTransport.py",
        "navi_sched/adapters/vllm/VLLMImageCacheMap.py",
        "navi_sched/adapters/vllm/VLLMNativePreparation.py",
        "navi_sched/adapters/vllm/VLLMAdapterUtils.py",
        "navi_sched/client.py",
        "navi_sched/master.py",
        "navi_sched/master_process.py",
        "navi_sched/replay_cli.py",
        "navi_sched/route_control.py",
        "navi_sched/util.py",
        "navi_sched/sched.py",
    ],
    data = [
        ":navi_sched/_navi_sched_core.so",
        ":wheel_python_runtime",
        "scripts/update_scheduler_config.sh",
    ],
    imports = ["."],
    srcs_version = "PY3",
)

alias(
    name = "navi_sched_wheel",
    actual = "//rtp_llm/flexlb/navi_sched/tools:navi_sched_wheel",
)

cc_test(
    name = "navi_sched_data_test",
    srcs = [
        "data/NaviSchedDataTest.cpp",
    ],
    data = [
        ":navi_sched_testdata",
        ":wheel_python_runtime",
    ],
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/navi:navi",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:json",
        "//rtp_llm/flexlb/navi_sched/vendor/unittest_framework",
        ":navi_sched_data_lib",
        ":navi_sched_role_lib",
    ],
)

cc_test(
    name = "agent_workload_ownership_test",
    srcs = [
        "resource/AgentWorkloadOwnership.cpp",
        "resource/AgentWorkloadOwnership.h",
        "resource/AgentWorkloadOwnershipTest.cpp",
    ],
    deps = [
        ":navi_sched_data_lib",
        "//rtp_llm/flexlb/navi_sched/vendor/unittest_framework",
    ],
)

cc_test(
    name = "master_batch_reservation_test",
    srcs = [
        "resource/MasterBatchReservation.cpp",
        "resource/MasterBatchReservation.h",
        "resource/MasterBatchReservationTest.cpp",
        "resource/PendingCacheUtil.cpp",
        "resource/PendingCacheUtil.h",
        "resource/RequestLatencyTracker.cpp",
        "resource/RequestLatencyTracker.h",
    ],
    copts = ["-fno-access-control"],
    data = [
        ":wheel_python_runtime",
        "testdata/pending_cache/master-batch-3516.json",
    ],
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/autil:hash_map",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:lock",
        "//rtp_llm/flexlb/navi_sched/vendor/unittest_framework",
        ":navi_sched_data_lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_test(
    name = "latency_residual_model_test",
    srcs = ["resource/LatencyResidualModelTest.cpp"],
    data = [":wheel_python_runtime"],
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/unittest_framework",
        ":navi_sched_role_lib",
    ],
)

cc_test(
    name = "nonlinear_prefill_model_r_test",
    srcs = ["resource/NonLinearPrefillModelRTest.cpp"],
    copts = ["-fno-access-control"],
    data = [":wheel_python_runtime"],
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/unittest_framework",
        ":navi_sched_role_lib",
    ],
)

cc_test(
    name = "agent_scheduler_config_test",
    srcs = ["resource/AgentSchedulerConfigRTest.cpp"],
    copts = ["-fno-access-control"],
    data = [":wheel_python_runtime"],
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/unittest_framework",
        ":navi_sched_role_lib",
    ],
)

cc_test(
    name = "navi_sched_role_test",
    srcs = [
        "resource/AgentPendingCacheRTest.cpp",
        "resource/MasterPendingCacheRTest.cpp",
        "resource/NaviSchedResourceTest.cpp",
        "scheduler/CostSchedulerKernelInternal.h",
        "scheduler/ScheduleContextReplayTest.cpp",
        "test/CostSchedulerIncidentTest.cpp",
        "test/NaviSchedClientTest.cpp",
        "test/NaviSchedMasterQueueTest.cpp",
        "test/ScheduleReplayRunner.cpp",
        "test/ScheduleReplayRunner.h",
    ],
    copts = ["-fno-access-control"],
    data = [":wheel_python_runtime", ":navi_sched_testdata"],
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/navi:navi",
        "//rtp_llm/flexlb/navi_sched/vendor/navi:navi_ops_testlib",
        "//rtp_llm/flexlb/navi_sched/vendor/unittest_framework",
        "//rtp_llm/flexlb/navi_sched/vendor/kmonitor:kmonitor_cpp_test_helper",
        ":navi_sched_data_lib",
        ":navi_sched_replay_debug_lib",
        ":navi_sched_role_lib",
    ],
)

cc_test(
    name = "navi_sched_kmon_exit_test",
    srcs = ["test/KmonExitTest.cpp"],
    copts = ["-fno-access-control"],
    env = {
        "LD_LIBRARY_PATH": "/opt/conda310/lib",
    },
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/unittest_framework",
        ":navi_sched_role_lib",
    ],
)

cc_binary(
    name = "schedule_replay_runner",
    testonly = True,
    srcs = [
        "test/ScheduleReplayCompare.cpp",
        "scheduler/CostSchedulerKernelInternal.h",
        "test/ScheduleReplayRunner.cpp",
        "test/ScheduleReplayRunner.h",
    ],
    deps = [
        ":navi_sched_data_lib",
        ":navi_sched_replay_debug_lib",
        ":navi_sched_role_lib",
        "//rtp_llm/flexlb/navi_sched/vendor/navi:navi",
    ],
)

cc_binary(
    name = "cost_scheduler_graph_benchmark",
    testonly = True,
    srcs = [
        "test/CostSchedulerGraphBenchmark.cpp",
        "test/ScheduleReplayRunner.cpp",
        "test/ScheduleReplayRunner.h",
    ],
    data = [":navi_sched_testdata"],
    deps = [
        ":navi_sched_data_lib",
        ":navi_sched_replay_debug_lib",
        ":navi_sched_role_lib",
        "//rtp_llm/flexlb/navi_sched/vendor/navi:navi",
    ],
)

cc_binary(
    name = "master_pending_cache_benchmark",
    testonly = True,
    srcs = ["test/MasterPendingCacheBenchmark.cpp"],
    deps = [
        ":navi_sched_role_lib",
        ":navi_sched_replay_debug_lib",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_binary(
    name = "cost_scheduler_master_graph_vis",
    testonly = True,
    srcs = [
        "test/CostSchedulerMasterGraphVis.cpp",
    ],
    data = [":navi_sched_testdata"],
    deps = [
        ":navi_sched_data_lib",
        ":navi_sched_role_lib",
        "//rtp_llm/flexlb/navi_sched/vendor/navi:navi",
    ],
)

cc_test(
    name = "navi_sched_cluster_test",
    srcs = [
        "test/NaviSchedClusterTest.cpp",
    ],
    copts = ["-fno-access-control"],
    data = [
        ":navi_sched_testdata",
        ":wheel_python_runtime",
    ],
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/navi:navi",
        "//rtp_llm/flexlb/navi_sched/vendor/unittest_framework",
        ":navi_sched_data_lib",
        ":navi_sched_role_lib",
    ],
)

cc_library(
    name = "navi_sched_simulator_lib",
    testonly = True,
    alwayslink = True,
    srcs = [
        "test/simulator/SimulatedAgentAccess.cpp",
        "test/simulator/SimulatedExecutionKernel.cpp",
        "test/simulator/SimulatedExecutionR.cpp",
        "test/simulator/TrafficReplayAccess.cpp",
        "test/simulator/TrafficReplayGraph.cpp",
        "test/simulator/TrafficReplayKernel.cpp",
        "test/simulator/TrafficReplayR.cpp",
        "test/simulator/TrafficReplayStartData.cpp",
        "test/simulator/TrafficReplayStartKernel.cpp",
        "test/simulator/TrafficReplayStartR.cpp",
    ],
    hdrs = [
        "test/simulator/SimulatedAgentAccess.h",
        "test/simulator/SimulatedExecutionR.h",
        "test/simulator/TrafficReplayAccess.h",
        "test/simulator/TrafficReplayGraph.h",
        "test/simulator/TrafficReplayR.h",
        "test/simulator/TrafficReplayStartData.h",
        "test/simulator/TrafficReplayStartR.h",
    ],
    deps = [
        "@rapidjson//:rapidjson",
        ":navi_sched_data_lib",
        ":navi_sched_replay_debug_lib",
        ":navi_sched_role_lib",
    ],
)

cc_test(
    name = "navi_sched_simulator_test",
    srcs = [
        "test/simulator/SimulatedExecutionRTest.cpp",
    ],
    data = [":wheel_python_runtime"],
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/navi:navi_ops_testlib",
        "//rtp_llm/flexlb/navi_sched/vendor/unittest_framework",
        ":navi_sched_simulator_lib",
    ],
)

pybind_extension(
    name = "navi_sched_simulate_core",
    testonly = True,
    additional_linker_inputs = ["scripts/navi_sched_pybind.lds"],
    srcs = [
        "bindings/NaviSchedBindings.h",
        "bindings/Pybindings.cpp",
        "test/simulator/SimulatedPybindings.cpp",
    ],
    copts = [
        "-fPIC",
        "-I/usr/local/include/python3",
    ],
    features = [
        "static_link_cpp_runtimes",
        "static_libgcc",
        "static_libcxx",
    ],
    linkopts = [
        "-Wl,-rpath,\\$$ORIGIN",
        "-Wl,-rpath,\\$$ORIGIN/lib",
        "-Wl,-rpath,/opt/conda310/lib",
        "-Wl,--version-script",
        "$(location scripts/navi_sched_pybind.lds)",
    ],
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/alog",
        "//rtp_llm/flexlb/navi_sched/vendor/navi:navi",
        "//rtp_llm/flexlb/navi_sched/vendor/navi:navi_pybind",
        "//rtp_llm/flexlb/navi_sched/vendor/network/gig:multi_call",
        ":navi_sched_role_lib",
        ":navi_sched_simulator_lib",
    ],
)

py_test(
    name = "navi_sched_python_role_test",
    srcs = [
        "test/NaviSchedPythonRoleTest.py",
    ],
    main = "test/NaviSchedPythonRoleTest.py",
    python_version = "PY3",
    env = {
        "LD_LIBRARY_PATH": "/opt/conda310/lib",
    },
    tags = ["exclusive"],
    deps = [
        ":py_sched_lib",
    ],
)

py_test(
    name = "navi_sched_rtp_llm_discovery_test",
    srcs = ["test/rtp_llm/RtpLLMDiscoveryTest.py"],
    main = "test/rtp_llm/RtpLLMDiscoveryTest.py",
    python_version = "PY3",
    env = {
        "LD_LIBRARY_PATH": "/opt/conda310/lib",
    },
    deps = [":py_sched_lib"],
)

py_test(
    name = "navi_sched_vllm_image_cache_map_test",
    srcs = ["test/vllm/VLLMImageCacheMapTest.py"],
    main = "test/vllm/VLLMImageCacheMapTest.py",
    python_version = "PY3",
    env = {
        "LD_LIBRARY_PATH": "/opt/conda310/lib",
    },
    deps = [":py_sched_lib"],
)

py_test(
    name = "navi_sched_vllm_engine_observer_test",
    srcs = ["test/vllm/VLLMEngineObserverTest.py"],
    main = "test/vllm/VLLMEngineObserverTest.py",
    python_version = "PY3",
    env = {
        "LD_LIBRARY_PATH": "/opt/conda310/lib",
    },
    deps = [":py_sched_lib"],
)

py_test(
    name = "navi_sched_vllm_production_stack_router_test",
    srcs = ["test/vllm/ProductionStackRouterTest.py"],
    main = "test/vllm/ProductionStackRouterTest.py",
    python_version = "PY3",
    env = {
        "LD_LIBRARY_PATH": "/opt/conda310/lib",
    },
    deps = [":py_sched_lib"],
)

py_test(
    name = "navi_sched_vllm_block_hash_test",
    srcs = ["test/vllm/VLLMBlockHashTest.py"],
    main = "test/vllm/VLLMBlockHashTest.py",
    python_version = "PY3",
    env = {
        "LD_LIBRARY_PATH": "/opt/conda310/lib",
    },
    deps = [":py_sched_lib"],
)

py_test(
    name = "navi_sched_vllm_cache_policy_test",
    srcs = ["test/vllm/VLLMCachePolicyTest.py"],
    main = "test/vllm/VLLMCachePolicyTest.py",
    python_version = "PY3",
    env = {
        "LD_LIBRARY_PATH": "/opt/conda310/lib",
    },
    deps = [":py_sched_lib"],
)

py_test(
    name = "navi_sched_vllm_adapter_end_to_end_test",
    srcs = ["test/vllm/VLLMAdapterEndToEndTest.py"],
    main = "test/vllm/VLLMAdapterEndToEndTest.py",
    python_version = "PY3",
    env = {
        "LD_LIBRARY_PATH": "/opt/conda310/lib",
    },
    tags = ["exclusive"],
    deps = [":py_sched_lib"],
)

py_test(
    name = "navi_sched_vllm_direct_transport_test",
    srcs = ["test/vllm/VLLMDirectTransportTest.py"],
    main = "test/vllm/VLLMDirectTransportTest.py",
    python_version = "PY3",
    env = {
        "LD_LIBRARY_PATH": "/opt/conda310/lib",
    },
    deps = [":py_sched_lib"],
)

py_test(
    name = "navi_sched_dashserving_adapter_test",
    srcs = ["test/dashserving/DashServingAdapterTest.py"],
    main = "test/dashserving/DashServingAdapterTest.py",
    python_version = "PY3",
    env = {
        "LD_LIBRARY_PATH": "/opt/conda310/lib",
    },
    deps = [":py_sched_lib"],
)

py_test(
    name = "flexlb_event_baseline_test",
    srcs = [
        "test/simulator/FlexLBEventBaselineTest.py",
        "test/simulator/flexlb_event_baseline.py",
    ],
    main = "test/simulator/FlexLBEventBaselineTest.py",
    python_version = "PY3",
)

py_test(
    name = "latency_model_state_test",
    srcs = [
        "test/simulator/LatencyModelStateTest.py",
        "test/simulator/latency_model_state.py",
    ],
    main = "test/simulator/LatencyModelStateTest.py",
    python_version = "PY3",
)

py_test(
    name = "export_latency_replay_jsonl_test",
    srcs = [
        "test/ExportLatencyReplayJsonl.py",
        "test/ExportLatencyReplayJsonlTest.py",
    ],
    main = "test/ExportLatencyReplayJsonlTest.py",
    python_version = "PY3",
)

py_test(
    name = "latency_online_replay_config_test",
    srcs = ["test/LatencyOnlineReplayConfigTest.py"],
    data = [":latency_online_replay"],
    main = "test/LatencyOnlineReplayConfigTest.py",
    python_version = "PY3",
)

py_test(
    name = "compare_latency_online_rls_test",
    srcs = [
        "test/CompareLatencyOnlineRls.py",
        "test/CompareLatencyOnlineRlsTest.py",
    ],
    main = "test/CompareLatencyOnlineRlsTest.py",
    python_version = "PY3",
)

py_test(
    name = "latency_downstream_scale_contract_test",
    srcs = [
        "test/AnalyzeLatencyFeatureGroups.py",
        "test/AnalyzeLatencyPhaseLeastSquares.py",
        "test/CompareLatencyAugmentedSquareShape.py",
        "test/CompareLatencyCrossWindow.py",
        "test/CompareLatencyFeatureCapacity.py",
        "test/CompareLatencyFixedCurvature.py",
        "test/CompareLatencyGlobalLeastSquares.py",
        "test/CompareLatencyShapeEstimators.py",
        "test/CompareLatencySingleIndex.py",
        "test/CompareLatencySquareShape.py",
        "test/LatencyDownstreamScaleContractTest.py",
        "test/PlotLatencyHighPressureU.py",
        "test/PlotLatencyShapeOperatingRange.py",
    ],
    main = "test/LatencyDownstreamScaleContractTest.py",
    python_version = "PY3",
)

py_test(
    name = "latency_independent_scale_contract_test",
    srcs = [
        "test/AnalyzeLatencyFeatureNeighbors.py",
        "test/AnalyzeLatencyPhaseLeastSquares.py",
        "test/AnalyzeVllmRequestLifecycles.py",
        "test/CompareLatencyOperatingMargin.py",
        "test/EvaluateDecodeProgressFeature.py",
        "test/LatencyIndependentScaleContractTest.py",
    ],
    main = "test/LatencyIndependentScaleContractTest.py",
    python_version = "PY3",
)

py_test(
    name = "ablate_latency_online_rates_test",
    srcs = [
        "test/AblateLatencyOnlineRates.py",
        "test/AblateLatencyOnlineRatesTest.py",
    ],
    main = "test/AblateLatencyOnlineRatesTest.py",
    python_version = "PY3",
)

py_test(
    name = "analyze_latency_shortlist_robustness_test",
    srcs = [
        "test/AnalyzeLatencyShortlistRobustness.py",
        "test/AnalyzeLatencyShortlistRobustnessTest.py",
    ],
    main = "test/AnalyzeLatencyShortlistRobustnessTest.py",
    python_version = "PY3",
)

py_test(
    name = "analyze_latency_adam_scale_test",
    srcs = [
        "test/AnalyzeLatencyAdamScale.py",
        "test/AnalyzeLatencyAdamScaleTest.py",
    ],
    main = "test/AnalyzeLatencyAdamScaleTest.py",
    python_version = "PY3",
)

py_test(
    name = "build_vllm_navi_images_test",
    srcs = [
        "scripts/build_vllm_navi_images.py",
        "scripts/build_vllm_navi_images_test.py",
    ],
    main = "scripts/build_vllm_navi_images_test.py",
    python_version = "PY3",
)

py_test(
    name = "scheduler_config_tools_test",
    srcs = [
        "scripts/navi_sched_config.py",
        "scripts/scheduler_config_tools_test.py",
    ],
    data = [
        "resource/SchedulerConfigR.cpp",
        "scripts/update_scheduler_config.sh",
        "scripts/update_scheduler_config_example.sh",
    ],
    main = "scripts/scheduler_config_tools_test.py",
    python_version = "PY3",
)

py_test(
    name = "replay_cli_test",
    srcs = ["scripts/replay_cli_test.py"],
    main = "scripts/replay_cli_test.py",
    python_version = "PY3",
    deps = [":py_sched_lib"],
)

py_test(
    name = "audit_kmonitor_dashboard_test",
    srcs = [
        "scripts/audit_kmonitor_dashboard.py",
        "scripts/audit_kmonitor_dashboard_test.py",
    ],
    main = "scripts/audit_kmonitor_dashboard_test.py",
    python_version = "PY3",
)

py_test(
    name = "create_kmonitor_dashboard_test",
    srcs = [
        "scripts/create_kmonitor_dashboard.py",
        "scripts/create_kmonitor_dashboard_test.py",
    ],
    main = "scripts/create_kmonitor_dashboard_test.py",
    python_version = "PY3",
)

py_test(
    name = "vllm_engine_transport_shutdown_test",
    srcs = ["test/vllm/VLLMEngineTransportShutdownTest.py"],
    data = ["navi_sched/adapters/vllm/VLLMEngineTransport.py"],
    main = "test/vllm/VLLMEngineTransportShutdownTest.py",
    python_version = "PY3",
)

filegroup(
    name = "navi_sched_testdata",
    testonly = True,
    srcs = glob(["testdata/**"]),
)

cc_test(
    name = "local_file_util_test",
    srcs = ["test/LocalFileUtilTest.cpp"],
    deps = [
        "//rtp_llm/flexlb/navi_sched/vendor/autil:file_util",
        "//rtp_llm/flexlb/navi_sched/vendor/autil:file_recorder",
    ],
)
