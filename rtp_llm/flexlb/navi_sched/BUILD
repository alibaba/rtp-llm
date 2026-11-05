package(default_visibility = ["//visibility:public"])
load("//third_party/pybind11_bazel:build_defs.bzl", "pybind_extension")
load("//bazel:defs.bzl", "cc_proto")
load("//bazel:git_version.bzl", "git_version_library")

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
    import_prefix = "aios/navi_sched",
    deps = ["//aios/network/arpc:proto"],
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
        ":tpu_prefill_execution_policy",
        "//aios/navi:navi",
        "//aios/navi:navi_runtime_config",
        "//aios/autil:hash_util",
        "//aios/autil:json",
    ],
    alwayslink = True,
    visibility = ["//aios/navi_sched:__subpackages__"],
)

cc_library(
    name = "navi_sched_replay_debug_lib",
    srcs = ["debug/NaviSchedScheduleReplayJson.cpp"],
    hdrs = ["debug/NaviSchedScheduleReplayJson.h"],
    deps = [
        "//aios/autil:json",
        ":navi_sched_data_lib",
    ],
    visibility = ["//aios/navi_sched:__subpackages__"],
)

cc_binary(
    name = "traffic_replay_log_converter",
    testonly = True,
    srcs = ["test/simulator/TrafficReplayLogConverter.cpp"],
    deps = [
        "@rapidjson//:rapidjson",
        "//aios/autil:string_helper",
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
        "test/legacy/LatencyLearningRateController.cpp",
        "test/legacy/LatencyLearningRateController.h",
    ],
    copts = ["-fno-access-control"],
    linkopts = ["-Wl,-rpath,$$ORIGIN/navi_sched"],
    deps = [
        "@rapidjson//:rapidjson",
        "//aios/kmonitor:kmonitor_client_cpp",
        "//aios/kmonitor:metrics_report_navi_resource",
        "//aios/navi:navi",
        "//aios/network/gig:multi_call",
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
        "//aios/kmonitor:kmonitor_client_cpp",
        "//aios/kmonitor:metrics_report_navi_resource",
        "//aios/navi:navi",
        "//aios/network/gig:multi_call",
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
        "//aios/kmonitor:kmonitor_client_cpp",
        "//aios/kmonitor:metrics_report_navi_resource",
        "//aios/navi:navi",
        "//aios/network/gig:multi_call",
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
        ":tpu_prefill_execution_policy",
        "//aios/navi:navi",
        "//aios/navi:navi_runtime_config",
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
        "//aios/navi:navi",
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
        "//aios/autil:json",
        "//aios/network/gig:multi_call",
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
        ":tpu_prefill_execution_policy",
        ":navi_sched_agent_header",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "//aios/navi:navi",
        "//aios/autil:file_recorder",
        "//aios/autil:crc32c",
        "//aios/autil:hash_base",
        "//aios/autil:hash_map",
        "//aios/autil:json",
        "//aios/autil:lock_free",
        "//aios/autil:log",
        "//aios/kmonitor:kmonitor_client_cpp",
        "//aios/kmonitor:metrics_report_navi_resource",
        "//aios/network/gig:multi_call",
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
        "-Wl,-rpath,\\$$ORIGIN",
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
    deps = [":navi_sched_role_lib"],
)

cc_test(
    name = "navi_sched_rtp_llm_adapter_test",
    srcs = [
        "adapters/rtp_llm/RtpLLMAdapterTest.cpp",
    ],
    copts = ["-fno-access-control"],
    deps = [
        "//aios/unittest_framework",
        ":navi_sched_role_lib",
    ],
)

cc_test(
    name = "navi_sched_agent_response_r_test",
    srcs = ["resource/AgentResponseRTest.cpp"],
    deps = [
        "//aios/unittest_framework",
        ":navi_sched_role_lib",
    ],
)

cc_test(
    name = "replay_control_r_test",
    srcs = ["resource/ReplayControlRTest.cpp"],
    data = [":wheel_python_runtime"],
    deps = [
        "//aios/unittest_framework",
        ":navi_sched_role_lib",
    ],
)

cc_test(
    name = "latency_history_sampler_test",
    srcs = ["resource/LatencyHistorySamplerTest.cpp"],
    deps = [
        "//aios/unittest_framework",
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
        ":navi_sched_data_lib",
        "//aios/autil:lock",
        "//aios/unittest_framework",
    ],
)

cc_library(
    name = "execution_cache_source_state_migration",
    srcs = ["packaging/dashllm_r50/ExecutionCacheSourceStateMigration.cpp"],
    hdrs = ["packaging/dashllm_r50/ExecutionCacheSourceStateMigration.h"],
    deps = [
        ":navi_sched_role_lib",
        "//aios/autil:crc32c",
        "//aios/autil:json",
    ],
)

cc_binary(
    name = "migrate_execution_cache_source_state",
    srcs = ["packaging/dashllm_r50/MigrateExecutionCacheSourceState.cpp"],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../py/navi_sched"],
    deps = [
        ":execution_cache_source_state_migration",
        "//aios/autil:json",
    ],
)

cc_test(
    name = "agent_history_state_test",
    srcs = ["resource/AgentHistoryStateTest.cpp"],
    data = [":migrate_execution_cache_source_state"],
    deps = [
        ":execution_cache_source_state_migration",
        "//aios/navi:navi_ops_testlib",
        "//aios/unittest_framework",
        ":navi_sched_role_lib",
    ],
)

cc_test(
    name = "navi_sched_client_response_r_test",
    srcs = ["resource/ClientResponseRTest.cpp"],
    data = [":wheel_python_runtime"],
    deps = [
        "//aios/unittest_framework",
        ":navi_sched_role_lib",
    ],
)

cc_test(
    name = "navi_sched_config_builder_test",
    srcs = ["test/NaviSchedConfigBuilderTest.cpp"],
    deps = [
        "//aios/unittest_framework",
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

genrule(
    name = "wheel_lockless_allocator",
    srcs = ["//aios/memory/lockless_allocator:llalloc"],
    outs = ["navi_sched/libllalloc.so"],
    cmd = "cp -L $< $@",
)

config_setting(
    name = "python312_wheel",
    define_values = {"navi_sched_python": "3.12"},
)

genrule(
    name = "wheel_python_runtime_py312",
    srcs = ["@python3.12_header//:lib/libpython3.12.so.1.0"],
    outs = ["navi_sched/libpython3.12.so.1.0"],
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
        "//aios/alog",
        "//aios/navi:navi",
        "//aios/navi:navi_pybind",
        "//aios/network/gig:multi_call",
        ":navi_sched_replay_debug_lib",
        ":navi_sched_role_lib",
    ],
    visibility = ["//aios/navi_sched:__subpackages__"],
    features = [
        "static_link_cpp_runtimes",
        "static_libgcc",
        "static_libcxx",
    ],
    linkopts = [
        "-Wl,-rpath,\\$$ORIGIN",
        "-Wl,-rpath,\\$$ORIGIN/lib",
    ] + select({
        ":python312_wheel": ["-Wl,-rpath,/usr/local/lib"],
        "//conditions:default": ["-Wl,-rpath,/opt/conda310/lib"],
    }) + [
        "-Wl,--version-script",
        "$(location scripts/navi_sched_pybind.lds)",
    ],
    copts = [
        "-fvisibility=hidden",
        "-fPIC",
    ] + select({
        ":python312_wheel": ["-I/usr/local/include/python3.12"],
        "//conditions:default": ["-I/usr/local/include/python3"],
    }),
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
        "navi_sched/adapters/rtp_llm/flexlb_info.py",
        "navi_sched/adapters/rtp_llm/SwitchableMasterClient.py",
        "navi_sched/adapters/rtp_llm/RtpLLMAdapterUtils.py",
        "navi_sched/adapters/rtp_llm/RtpLLMMaster.py",
        "navi_sched/adapters/rtp_llm/FlexlbPrefillMaster.py",
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
        "navi_sched/adapters/vllm/ExecutionTimingProbe.py",
        "navi_sched/adapters/vllm/VLLMEngineObserver.py",
        "navi_sched/adapters/vllm/VLLMExecutionGeometry.py",
        "navi_sched/adapters/vllm/VLLMEngineTransport.py",
        "navi_sched/adapters/vllm/VLLMImageCacheMap.py",
        "navi_sched/adapters/vllm/VLLMNativePreparation.py",
        "navi_sched/adapters/vllm/VLLMWorkerObserver.py",
        "navi_sched/adapters/vllm/VLLMAdapterUtils.py",
        "navi_sched/client.py",
        "navi_sched/master.py",
        "navi_sched/master_process.py",
        "navi_sched/spectrum_prefill_master.py",
        "navi_sched/spectrum_kmon.py",
        "navi_sched/spectrum_standalone_master.py",
        "navi_sched/replay_cli.py",
        "navi_sched/route_control.py",
        "navi_sched/util.py",
        "navi_sched/sched.py",
    ],
    data = [
        ":navi_sched/_navi_sched_core.so",
        ":wheel_lockless_allocator",
        "scripts/update_scheduler_config.sh",
    ] + select({
        ":python312_wheel": [":wheel_python_runtime_py312"],
        "//conditions:default": [":wheel_python_runtime"],
    }),
    imports = ["."],
    srcs_version = "PY3",
)

alias(
    name = "navi_sched_wheel",
    actual = "//aios/navi_sched/tools:navi_sched_wheel",
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
        "//aios/navi:navi",
        "//aios/autil:json",
        "//aios/unittest_framework",
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
        "//aios/unittest_framework",
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
        "//aios/autil:hash_map",
        "//aios/autil:lock",
        "//aios/unittest_framework",
        ":navi_sched_data_lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_test(
    name = "execution_utilization_controller_test",
    srcs = [
        "scheduler/ExecutionUtilizationController.h",
        "scheduler/ExecutionUtilizationControllerTest.cpp",
    ],
    deps = ["//aios/unittest_framework"],
)

cc_test(
    name = "latency_residual_model_test",
    srcs = ["resource/LatencyResidualModelTest.cpp"],
    data = [":wheel_python_runtime"],
    deps = [
        "//aios/unittest_framework",
        ":navi_sched_role_lib",
    ],
)

cc_test(
    name = "nonlinear_prefill_model_r_test",
    srcs = ["resource/NonLinearPrefillModelRTest.cpp"],
    copts = ["-fno-access-control"],
    data = select({
        ":python312_wheel": [":wheel_python_runtime_py312"],
        "//conditions:default": [":wheel_python_runtime"],
    }),
    deps = [
        "//aios/navi:navi_ops_testlib",
        "//aios/unittest_framework",
        ":navi_sched_role_lib",
    ],
)

cc_test(
    name = "legacy_v76_fit_parity_test",
    srcs = [
        "resource/LegacyV76FitParityTest.cpp",
        "resource/testdata/main_projection/MainProjectionReference.cpp",
        "resource/testdata/main_projection/MainProjectionReference.h",
        "resource/testdata/legacy_v76/LegacyV76NonLinearPrefillModelR.cpp",
        "resource/testdata/legacy_v76/LegacyV76NonLinearPrefillModelR.h",
    ],
    copts = ["-fno-access-control"],
    deps = [
        "//aios/navi:navi_ops_testlib",
        "//aios/unittest_framework",
        ":navi_sched_role_lib",
    ],
)

cc_test(
    name = "agent_scheduler_config_test",
    srcs = ["resource/AgentSchedulerConfigRTest.cpp"],
    copts = ["-fno-access-control"],
    data = [":wheel_python_runtime"],
    deps = [
        "//aios/unittest_framework",
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
    data = [":navi_sched_testdata"] + select({
        ":python312_wheel": [":wheel_python_runtime_py312"],
        "//conditions:default": [":wheel_python_runtime"],
    }),
    deps = [
        "//aios/navi:navi",
        "//aios/navi:navi_ops_testlib",
        "//aios/unittest_framework",
        "//aios/kmonitor:kmonitor_cpp_test_helper",
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
        "//aios/unittest_framework",
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
        "//aios/navi:navi",
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
        "//aios/navi:navi",
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
        "//aios/navi:navi",
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
        "//aios/navi:navi",
        "//aios/unittest_framework",
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
        "//aios/navi:navi_ops_testlib",
        "//aios/unittest_framework",
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
        "//aios/alog",
        "//aios/navi:navi",
        "//aios/navi:navi_pybind",
        "//aios/network/gig:multi_call",
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
    name = "navi_sched_spectrum_standalone_master_test",
    srcs = [
        "navi_sched/spectrum_standalone_master.py",
        "test/dashserving/SpectrumStandaloneMasterTest.py",
    ],
    main = "test/dashserving/SpectrumStandaloneMasterTest.py",
    python_version = "PY3",
)

sh_test(
    name = "navi_sched_standalone_master_bootstrap_test",
    srcs = ["test/dashserving/test_standalone_master_bootstrap.sh"],
    data = ["packaging/dashllm_r39/standalone_master_bootstrap.sh"],
)

sh_test(
    name = "navi_sched_prefill_bootstrap_cold_test",
    srcs = ["test/dashserving/test_prefill_bootstrap_cold.sh"],
    data = [
        "packaging/dashllm_r39/prefill_bootstrap.sh",
        "packaging/dashllm_r39/Dockerfile",
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
    name = "navi_sched_fixed_prefill_plan_test",
    srcs = ["test/vllm/VLLMFixedPrefillPlanTest.py"],
    main = "test/vllm/VLLMFixedPrefillPlanTest.py",
    data = [
        "packaging/dashllm_r50/runtime_overlay/vllm_torchtpu/core/pp_chunk_scheduler.py",
        "packaging/dashllm_r50/runtime_overlay/vllm_torchtpu/core/raiden_pd.py",
        "packaging/dashllm_r50/runtime_overlay/vllm_torchtpu/platforms/tpu_platform.py",
        "packaging/dashllm_r50/runtime_overlay/vllm/v1/core/sched/scheduler.py",
    ],
    python_version = "PY3",
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
    name = "navi_sched_python_gc_timing_probe_test",
    srcs = ["test/vllm/PythonGcTimingProbeTest.py"],
    main = "test/vllm/PythonGcTimingProbeTest.py",
    python_version = "PY3",
    deps = [":py_sched_lib"],
)

py_test(
    name = "navi_sched_vllm_worker_observer_test",
    srcs = ["test/vllm/VLLMWorkerObserverTest.py"],
    main = "test/vllm/VLLMWorkerObserverTest.py",
    python_version = "PY3",
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
    name = "navi_sched_vllm_request_registration_test",
    srcs = ["test/vllm/VLLMRequestRegistrationTest.py"],
    main = "test/vllm/VLLMRequestRegistrationTest.py",
    data = [
        "adapters/vllm/VLLMRequestRegistration.py",
        "testdata/vllm/r64_frontend_registration.json",
    ],
    python_version = "PY3",
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
    imports = ["test"],
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
    imports = ["test"],
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
    imports = ["test"],
    python_version = "PY3",
)

py_test(
    name = "ablate_latency_online_rates_test",
    imports = ["test"],
    srcs = [
        "test/AblateLatencyOnlineRates.py",
        "test/AblateLatencyOnlineRatesTest.py",
    ],
    main = "test/AblateLatencyOnlineRatesTest.py",
    python_version = "PY3",
)

py_test(
    name = "analyze_latency_shortlist_robustness_test",
    imports = ["test"],
    srcs = [
        "test/AnalyzeLatencyShortlistRobustness.py",
        "test/AnalyzeLatencyShortlistRobustnessTest.py",
    ],
    main = "test/AnalyzeLatencyShortlistRobustnessTest.py",
    python_version = "PY3",
)

py_test(
    name = "analyze_latency_adam_scale_test",
    imports = ["test"],
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
    imports = ["scripts"],
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

filegroup(
    name = "navi_sched_testdata",
    testonly = True,
    srcs = glob(["testdata/**"]),
)



cc_test(
    name = "navi_sched_agent_typed_cache_test",
    srcs = ["test/AgentTypedCacheResourceTest.cpp"],
    deps = [":navi_sched_role_lib", "//aios/unittest_framework"],
)

cc_test(
    name = "navi_sched_master_typed_pending_cache_test",
    srcs = ["test/MasterTypedPendingCacheTest.cpp"],
    deps = [":navi_sched_role_lib", "//aios/unittest_framework"],
)

cc_test(
    name = "navi_sched_master_typed_cache_state_test",
    srcs = ["test/MasterTypedCacheStateTest.cpp"],
    copts = ["-fno-access-control"],
    deps = [":navi_sched_role_lib", "//aios/unittest_framework"],
)

cc_test(
    name = "navi_sched_cache_key_type_test",
    srcs = ["data/CacheKeyTypeTest.cpp"],
    deps = [":navi_sched_data_lib", "//aios/unittest_framework"],
)

cc_test(
    name = "navi_sched_agent_typed_cache_policy_test",
    srcs = ["test/AgentTypedCachePolicyTest.cpp"],
    copts = ["-fno-access-control"],
    deps = [":navi_sched_role_lib", "//aios/unittest_framework"],
)

cc_test(
    name = "navi_sched_typed_pending_caller_test",
    srcs = ["test/TypedPendingCallerTest.cpp"],
    copts = ["-fno-access-control"],
    deps = [":navi_sched_role_lib", "//aios/unittest_framework"],
)

# Runtime policy arithmetic and its typed control-message representation.
cc_library(
    name = "tpu_prefill_execution_policy",
    srcs = ["common/TPUPrefillExecutionPolicy.cpp"],
    hdrs = ["common/TPUPrefillExecutionPolicy.h"],
    deps = ["//aios/autil:data_buffer"],
)

cc_test(
    name = "tpu_prefill_execution_policy_test",
    srcs = ["test/TPUPrefillExecutionPolicyTest.cpp", "test/r50_split_goldens.h"],
    deps = [":tpu_prefill_execution_policy", "//aios/unittest_framework"],
)

cc_test(
    name = "geometry_integration_test",
    srcs = ["resource/GeometryIntegrationTest.cpp", "test/DiscreteSearchBaseline.h"],
    copts = ["-fno-access-control"],
    deps = [
        "//aios/navi:navi_ops_testlib",
        ":navi_sched_role_lib",
        "//aios/unittest_framework",
    ],
)

py_test(
    name = "navi_sched_vllm_fixed_pp_scheduler_test",
    srcs = [
        "test/vllm/VLLMFixedPpSchedulerTest.py",
        "test/vllm/VLLMPDBatchPreparationTest.py",
        "test/vllm/VLLMV6DLocalCoverageTest.py",
    ],
    main = "test/vllm/VLLMFixedPpSchedulerTest.py",
    imports = ["test/vllm"],
    data = [
        ":navi_sched/_navi_sched_core.so",
        "navi_sched/adapters/vllm/ExecutionTimingProbe.py",
        "packaging/dashllm_r50/runtime_overlay/kvs_connector/v6d_object_connector.py",
        "packaging/dashllm_r50/runtime_overlay/vllm/v1/core/sched/scheduler.py",
        "packaging/dashllm_r50/runtime_overlay/vllm_torchtpu/core/pp_chunk_scheduler.py",
    ],
    python_version = "PY3",
)

py_test(
    name = "navi_sched_vllm_pd_batch_preparation_test",
    srcs = ["test/vllm/VLLMPDBatchPreparationTest.py", "test/vllm/VLLMV6DLocalCoverageTest.py"],
    main = "test/vllm/VLLMPDBatchPreparationTest.py",
    imports = ["test/vllm"],
    data = [
        "packaging/dashllm_r50/runtime_overlay/kvs_connector/v6d_object_connector.py",
        "packaging/dashllm_r50/runtime_overlay/vllm/v1/core/sched/scheduler.py",
        "navi_sched/adapters/vllm/VLLMEngineTransport.py",
    ],
    python_version = "PY3",
)

py_test(
    name = "navi_sched_v6d_local_coverage_test",
    srcs = ["test/vllm/VLLMV6DLocalCoverageTest.py"],
    main = "test/vllm/VLLMV6DLocalCoverageTest.py",
    data = ["packaging/dashllm_r50/runtime_overlay/kvs_connector/v6d_object_connector.py"],
    python_version = "PY3",
)

py_test(
    name = "navi_sched_v6d_lookup_patch_test",
    srcs = [
        "packaging/dashllm_r50/lookup_audit_patch.py",
        "packaging/dashllm_r50/preprocessing_audit_patch.py",
        "packaging/dashllm_r50/test_lookup_audit_patch.py",
        "packaging/dashllm_r50/test_preprocessing_audit_patch.py",
    ],
    main = "packaging/dashllm_r50/test_lookup_audit_patch.py",
    imports = ["packaging/dashllm_r50"],
    python_version = "PY3",
)

py_test(
    name = "navi_sched_v6d_read_plan_patch_test",
    srcs = [
        "packaging/dashllm_r50/read_plan_audit_patch.py",
        "packaging/dashllm_r50/preprocessing_audit_patch.py",
        "packaging/dashllm_r50/test_read_plan_audit_patch.py",
        "packaging/dashllm_r50/test_preprocessing_audit_patch.py",
    ],
    main = "packaging/dashllm_r50/test_read_plan_audit_patch.py",
    imports = ["packaging/dashllm_r50"],
    python_version = "PY3",
)

py_test(
    name = "navi_sched_v6d_read_plan_audit_test",
    srcs = ["test/vllm/VLLMV6DReadPlanAuditTest.py"],
    main = "test/vllm/VLLMV6DReadPlanAuditTest.py",
    data = ["packaging/dashllm_r50/runtime_overlay/v6d/server/peers/tiered_vineyard/peer.py"],
    python_version = "PY3",
)

py_test(
    name = "navi_sched_fa_prefix_read_patch_test",
    srcs = [
        "packaging/dashllm_r50/fa_prefix_read_patch.py",
        "packaging/dashllm_r50/checkpoint_probe_patch.py",
        "packaging/dashllm_r50/preprocessing_audit_patch.py",
        "packaging/dashllm_r50/test_preprocessing_audit_patch.py",
        "packaging/dashllm_r50/test_fa_prefix_read_patch.py",
    ],
    main = "packaging/dashllm_r50/test_fa_prefix_read_patch.py",
    imports = ["packaging/dashllm_r50"],
    python_version = "PY3",
)

py_test(
    name = "navi_sched_v6d_lookup_audit_test",
    srcs = ["test/vllm/VLLMV6DLookupAuditTest.py"],
    main = "test/vllm/VLLMV6DLookupAuditTest.py",
    data = [
        "packaging/dashllm_r50/runtime_overlay/kvs_connector/v6d_object_connector.py",
        "packaging/dashllm_r50/runtime_overlay/v6d/server/peers/tiered_vineyard/peer.py",
        "packaging/dashllm_r50/runtime_overlay/v6d/client/client.py",
    ],
    python_version = "PY3",
)

py_test(
    name = "navi_sched_r50_publication_test",
    srcs = ["test/vllm/VLLMR50PublicationTest.py"],
    main = "test/vllm/VLLMR50PublicationTest.py",
    data = ["packaging/dashllm_r50/runtime_overlay/kvs_connector/v6d_object_connector.py"],
    python_version = "PY3",
)

py_test(
    name = "navi_sched_vllm_cache_publication_test",
    srcs = ["test/vllm/VLLMCachePublicationTest.py", "test/vllm/VLLMEngineObserverTest.py"],
    main = "test/vllm/VLLMCachePublicationTest.py",
    python_version = "PY3",
    deps = [":py_sched_lib"],
)

py_test(
    name = "navi_sched_r50_tracker_publication_test",
    srcs = ["test/vllm/VLLMR50TrackerPublicationTest.py"],
    main = "test/vllm/VLLMR50TrackerPublicationTest.py",
    data = ["packaging/dashllm_r50/runtime_overlay/v6d/server/peers/tiered_vineyard/peer.py"],
    python_version = "PY3",
)

py_test(
    name = "navi_sched_r50_execution_geometry_test",
    srcs = ["test/vllm/VLLMExecutionGeometryTest.py"],
    main = "test/vllm/VLLMExecutionGeometryTest.py",
    data = ["navi_sched/adapters/vllm/VLLMExecutionGeometry.py"],
    python_version = "PY3",
)

py_test(
    name = "navi_sched_r52_workspace_status_test",
    srcs = ["packaging/dashllm_r50/test_workspace_status.py"],
    main = "packaging/dashllm_r50/test_workspace_status.py",
    data = ["packaging/dashllm_r50/workspace_status.py"],
    python_version = "PY3",
)

py_test(
    name = "navi_sched_r50_release_proof_test",
    srcs = ["packaging/dashllm_r50/test_build_release.py"],
    main = "packaging/dashllm_r50/test_build_release.py",
    data = [
        "packaging/dashllm_r50/build_release.py",
        "packaging/dashllm_r50/image-dashllm_cmd.sh",
        "packaging/dashllm_r50/base-launcher-manifest.json",
    ],
    python_version = "PY3",
)

py_test(
    name = "navi_sched_r50_stage_manifest_test",
    srcs = ["packaging/dashllm_r50/test_stage_prefill.py"],
    main = "packaging/dashllm_r50/test_stage_prefill.py",
    data = ["packaging/dashllm_r50/stage_prefill.py", "packaging/dashllm_r50/capture_prefill.py",
            "packaging/dashllm_r50/image-dashllm_cmd.sh", "packaging/dashllm_r50/build_release.py"],
    python_version = "PY3",
)

py_test(
    name = "navi_sched_source34_master_release_test",
    srcs = [
        "packaging/dashllm_r50/build_release.py",
        "packaging/dashllm_r50/build_master_release.py",
        "packaging/dashllm_r50/stage_prefill.py",
        "packaging/dashllm_r50/stage_master.py",
        "packaging/dashllm_r50/test_stage_master.py",
        "packaging/dashllm_r50/migrate_master_pool.py",
        "packaging/dashllm_r50/test_migrate_master_pool.py",
    ],
    main = "packaging/dashllm_r50/test_stage_master.py",
    python_version = "PY3",
)

py_test(
    name = "navi_sched_legacy_timing_probe_test",
    srcs = ["packaging/dashllm_r50/test_timing_diagnostic.py", "packaging/dashllm_r50/timing_diagnostic.py"],
    main = "packaging/dashllm_r50/test_timing_diagnostic.py",
    data = ["navi_sched/adapters/vllm/ExecutionTimingProbe.py", "navi_sched/adapters/vllm/VLLMEngineObserver.py",
            "packaging/dashllm_r50/stage_timing_diagnostic.py", "packaging/dashllm_r50/capture_prefill.py",
            "packaging/dashllm_r50/stage_prefill.py",
            "packaging/dashllm_r50/build_cache_snapshot_comparison.py",
            "packaging/dashllm_r50/build_output_clock_overlay.py",
        "packaging/dashllm_r50/build_fresh_legacy_overlay.py",
            "packaging/dashllm_r39/prefill_bootstrap.sh",
            "packaging/dashllm_r50/build_release.py"],
    python_version = "PY3",
)

py_test(
    name = "navi_sched_archived_v76_profile_test",
    srcs = ["packaging/dashllm_r50/test_archived_v76_profile.py"],
    main = "packaging/dashllm_r50/test_archived_v76_profile.py",
    data = [
        "packaging/dashllm_r50/archived_v76_profile.py",
        "packaging/dashllm_r50/build_archived_v76_overlay.py",
        "packaging/dashllm_r50/build_release.py",
        "packaging/dashllm_r50/stage_timing_diagnostic.py",
        "packaging/dashllm_r50/stage_prefill.py",
        "packaging/dashllm_r50/capture_prefill.py",
        "packaging/dashllm_r50/legacy_v76_sources/VLLMEngineObserver.py",
        "packaging/dashllm_r50/legacy_v76_sources/VLLMCachePolicy.py",
    ],
    python_version = "PY3",
)

py_test(
    name = "navi_sched_model35_learning_profile_test",
    srcs = ["packaging/dashllm_r50/test_learning_profile_update.py", "packaging/dashllm_r50/learning_profile_update.py"],
    main = "packaging/dashllm_r50/test_learning_profile_update.py",
    python_version = "PY3",
    imports = ["packaging/dashllm_r50"],
)

py_test(
    name = "navi_sched_peer_config_restore_test",
    srcs = ["packaging/dashllm_r50/test_restore_peer_cache_config.py", "packaging/dashllm_r50/restore_peer_cache_config.py", "packaging/dashllm_r50/learning_profile_update.py"],
    main = "packaging/dashllm_r50/test_restore_peer_cache_config.py",
    python_version = "PY3",
    imports = ["packaging/dashllm_r50"],
)

cc_binary(
    name = "frozen_feature_predict",
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
        "test/FrozenFeaturePredict.cpp",
    ],
    deps = [
        ":navi_sched_data_lib",
        "@rapidjson//:rapidjson",
        "//aios/kmonitor:kmonitor_client_cpp",
        "//aios/kmonitor:metrics_report_navi_resource",
        "//aios/navi:navi",
        "//aios/network/gig:multi_call",
    ],
)

py_test(
    name = "frozen_feature_predict_test",
    srcs = ["test/FrozenFeaturePredictTest.py"],
    data = [":frozen_feature_predict"],
    main = "test/FrozenFeaturePredictTest.py",
    python_version = "PY3",
)


# Offline arithmetic replay of the deployed model35 cache-resolved projector.
cc_binary(
    name = "cache_resolved_geometry_replay",
    testonly = True,
    srcs = ["test/CacheResolvedGeometryReplay.cpp"],
    deps = [":tpu_prefill_execution_policy", "@rapidjson//:rapidjson"],
)

py_test(
    name = "cache_resolved_geometry_replay_test",
    srcs = ["test/CacheResolvedGeometryReplayTest.py"],
    data = [":cache_resolved_geometry_replay"],
    main = "test/CacheResolvedGeometryReplayTest.py",
    python_version = "PY3",
)

py_test(
    name = "navi_sched_checkpoint_probe_patch_test",
    srcs = [
        "packaging/dashllm_r50/checkpoint_probe_patch.py",
        "packaging/dashllm_r50/preprocessing_audit_patch.py",
        "packaging/dashllm_r50/test_checkpoint_probe_patch.py",
        "packaging/dashllm_r50/test_preprocessing_audit_patch.py",
    ],
    main = "packaging/dashllm_r50/test_checkpoint_probe_patch.py",
    imports = ["packaging/dashllm_r50"],
    python_version = "PY3",
)

py_test(
    name = "navi_sched_native_fifo_patch_test",
    srcs = [
        "packaging/dashllm_r50/test_native_fifo_patch.py",
        "packaging/dashllm_r50/native_fifo_patch.py",
        "packaging/dashllm_r50/checkpoint_probe_patch.py",
        "packaging/dashllm_r50/stage_prefill.py",
    ],
    main = "packaging/dashllm_r50/test_native_fifo_patch.py",
    imports = ["packaging/dashllm_r50"],
    python_version = "PY3",
)

py_test(
    name = "navi_sched_tair_cold_read_policy_test",
    srcs = ["packaging/dashllm_r50/test_tair_cold_read_policy.py"],
    data = ["packaging/dashllm_r50/runtime_overlay/v6d/server/peers/tiered_vineyard/peer.py"],
    main = "packaging/dashllm_r50/test_tair_cold_read_policy.py",
    python_version = "PY3",
)

py_test(
    name = "navi_sched_tair_cold_read_patch_test",
    srcs = [
        "packaging/dashllm_r50/test_tair_cold_read_patch.py",
        "packaging/dashllm_r50/tair_cold_read_patch.py",
        "packaging/dashllm_r50/checkpoint_probe_patch.py",
        "packaging/dashllm_r50/stage_prefill.py",
    ],
    main = "packaging/dashllm_r50/test_tair_cold_read_patch.py",
    imports = ["packaging/dashllm_r50"],
    python_version = "PY3",
)

py_test(
    name = "navi_sched_gc_timing_audit_patch_test",
    srcs = [
        "packaging/dashllm_r50/test_gc_timing_audit_patch.py",
        "packaging/dashllm_r50/gc_timing_audit_patch.py",
        "packaging/dashllm_r50/checkpoint_probe_patch.py",
        "packaging/dashllm_r50/stage_prefill.py",
    ],
    main = "packaging/dashllm_r50/test_gc_timing_audit_patch.py",
    imports = ["packaging/dashllm_r50"],
    python_version = "PY3",
)

py_test(
    name = "navi_sched_output_phase_timing_probe_test",
    srcs = ["test/vllm/OutputPhaseTimingProbeTest.py"],
    main = "test/vllm/OutputPhaseTimingProbeTest.py",
    python_version = "PY3",
    deps = [":py_sched_lib"],
)

py_test(
    name = "navi_sched_output_phase_audit_patch_test",
    srcs = [
        "packaging/dashllm_r50/test_output_phase_audit_patch.py",
        "packaging/dashllm_r50/output_phase_audit_patch.py",
        "packaging/dashllm_r50/checkpoint_probe_patch.py",
        "packaging/dashllm_r50/stage_prefill.py",
    ],
    main = "packaging/dashllm_r50/test_output_phase_audit_patch.py",
    imports = ["packaging/dashllm_r50"],
    python_version = "PY3",
)

py_test(
    name = "navi_sched_worker_startup_gc_sweep_test",
    srcs = ["test/vllm/WorkerStartupGcSweepTest.py"],
    main = "test/vllm/WorkerStartupGcSweepTest.py",
    python_version = "PY3",
    deps = [":py_sched_lib"],
)

py_test(
    name = "navi_sched_worker_startup_gc_sweep_patch_test",
    srcs = [
        "packaging/dashllm_r50/test_worker_startup_gc_sweep_patch.py",
        "packaging/dashllm_r50/worker_startup_gc_sweep_patch.py",
        "packaging/dashllm_r50/checkpoint_probe_patch.py",
        "packaging/dashllm_r50/stage_prefill.py",
    ],
    main = "packaging/dashllm_r50/test_worker_startup_gc_sweep_patch.py",
    imports = ["packaging/dashllm_r50"],
    python_version = "PY3",
)

py_test(
    name = "navi_sched_worker_sampling_stage_audit_test",
    srcs = ["test/vllm/WorkerSamplingStageAuditTest.py"],
    main = "test/vllm/WorkerSamplingStageAuditTest.py",
    python_version = "PY3",
    deps = [":py_sched_lib"],
)

py_test(
    name = "navi_sched_worker_sampling_stage_audit_patch_test",
    srcs = [
        "packaging/dashllm_r50/test_worker_sampling_stage_audit_patch.py",
        "packaging/dashllm_r50/worker_sampling_stage_audit_patch.py",
        "packaging/dashllm_r50/checkpoint_probe_patch.py",
        "packaging/dashllm_r50/stage_prefill.py",
    ],
    main = "packaging/dashllm_r50/test_worker_sampling_stage_audit_patch.py",
    imports = ["packaging/dashllm_r50"],
    python_version = "PY3",
)

py_test(
    name = "navi_sched_worker_draft_compile_audit_test",
    srcs = ["test/vllm/WorkerDraftCompileAuditTest.py"],
    main = "test/vllm/WorkerDraftCompileAuditTest.py",
    python_version = "PY3",
    deps = [":py_sched_lib"],
)

py_test(
    name = "navi_sched_worker_draft_compile_audit_patch_test",
    srcs = [
        "packaging/dashllm_r50/test_worker_draft_compile_audit_patch.py",
        "packaging/dashllm_r50/worker_draft_compile_audit_patch.py",
        "packaging/dashllm_r50/checkpoint_probe_patch.py",
        "packaging/dashllm_r50/stage_prefill.py",
    ],
    main = "packaging/dashllm_r50/test_worker_draft_compile_audit_patch.py",
    imports = ["packaging/dashllm_r50"],
    python_version = "PY3",
)

py_test(
    name = "navi_sched_async_diagnostic_logging_test",
    srcs = ["test/vllm/AsyncDiagnosticLoggingTest.py"],
    data = [
        "testdata/vllm/observer_hot_paths_original_sync.py",
        "testdata/vllm/observer_hot_paths_merged_sync.py",
        "packaging/dashllm_r50/runtime_overlay/vllm/v1/engine/core.py",
        "packaging/dashllm_r50/runtime_overlay/vllm_torchtpu/core/pp_chunk_scheduler.py",
        "packaging/dashllm_r50/runtime_overlay/kvs_connector/v6d_object_connector.py",
    ],
    main = "test/vllm/AsyncDiagnosticLoggingTest.py",
    python_version = "PY3",
    deps = [":py_sched_lib"],
)

py_test(
    name = "navi_sched_async_diagnostic_logging_patch_test",
    srcs = [
        "packaging/dashllm_r50/test_async_diagnostic_logging_patch.py",
        "packaging/dashllm_r50/async_diagnostic_logging_patch.py",
        "packaging/dashllm_r50/checkpoint_probe_patch.py",
        "packaging/dashllm_r50/stage_prefill.py",
    ],
    main = "packaging/dashllm_r50/test_async_diagnostic_logging_patch.py",
    imports = ["packaging/dashllm_r50"],
    python_version = "PY3",
)

# Historical replay controller fixture only; excluded from serving source globs.
cc_test(
    name = "latency_learning_rate_controller_test",
    srcs = [
        "test/legacy/LatencyLearningRateController.cpp",
        "test/legacy/LatencyLearningRateController.h",
        "test/legacy/LatencyLearningRateControllerTest.cpp",
    ],
    data = [":wheel_python_runtime"],
    deps = ["//aios/unittest_framework", ":navi_sched_role_lib"],
)
