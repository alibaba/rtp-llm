load("//rtp_llm/test/smoke:defs.bzl", "custom_smoke_test")

package(default_visibility = ["//visibility:public"])

# Independent pure-Python oracle for EPLB window ratios.  No torch, no CUDA:
# it is the trusted reference for GPU recorder tests.
py_library(
    name = "window_ratio_oracle",
    srcs = ["window_ratio_oracle.py"],
)

# Shared synthetic-model/router builders, also imported by EP2 worker
# processes (module name must stay expert_load_recorder_test).
py_library(
    name = "expert_load_recorder_test_lib",
    testonly = True,
    srcs = ["expert_load_recorder_test.py"],
    deps = [
        ":window_ratio_oracle",
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
)

# Single-GPU tests of the REAL production recorder
# (rtp_llm.models_py.utils.expert_load_monitor + PureTpRouterNoQuant):
# direct helper UTs (strided/int32/int64/sentinel/chunk/2**31), explicit-input
# views and storage validation, CUDA graph capture/replay padding, streams.
py_test(
    name = "expert_load_recorder_test",
    srcs = ["expert_load_recorder_test.py"],
    deps = [
        ":window_ratio_oracle",
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "1"},
    exec_properties = {
        "gpu": "L20_CU13",
        "gpu_count": "1",
    },
    tags = ["L20_CU13"],
)

# EP2 destination ownership after NCCL dispatch scaffolding. Engine window
# lifecycle and actual DeepEP dispatch are covered separately.
py_test(
    name = "expert_load_recorder_ep2_test",
    srcs = ["expert_load_recorder_ep2_test.py"],
    deps = [
        ":expert_load_recorder_test_lib",
        ":window_ratio_oracle",
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "2"},
    exec_properties = {
        "gpu": "L20_CU13",
        "gpu_count": "2",
    },
    tags = ["L20_CU13"],
)

# Fixed-length requests must produce new prefill and decode EPLB samples,
# independently of warmup, in eager and CUDA graph/async execution.
custom_smoke_test(
    name = "service_smoke_test",
    main = "service_smoke_test.py",
    smoke_args = "--act_type BF16 --tp_size 1 --ep_size 1 --world_size 1 --max_seq_len 2048 --kv_cache_mem_mb 2048 --concurrency_limit 4 --max_context_batch_size 1 --seq_size_per_block 64 --warm_up 1",
    gpu_type = ["L20_CU13"],
    deps = ["//rtp_llm:requests"],
)

py_test(
    name = "deepep_router_monitor_test",
    srcs = ["deepep_router_monitor_test.py"],
    deps = [
        ":window_ratio_oracle",
        "//rtp_llm/models_py/standalone:py_standalone_testlib",
    ],
    env = {"GPU_COUNT": "1"},
    exec_properties = {"gpu": "L20_CU13", "gpu_count": "1"},
    tags = ["L20_CU13"],
)

custom_smoke_test(
    name = "service_ep2_smoke_test",
    main = "service_smoke_test.py",
    smoke_args = "--act_type BF16 --tp_size 2 --ep_size 2 --world_size 2 --max_seq_len 256 --kv_cache_mem_mb 2048 --concurrency_limit 256 --max_context_batch_size 1 --seq_size_per_block 64 --warm_up 1 --use_deepep_moe 1 --use_deepep_low_latency 1 --moe_strategy no_auant_ep_low_latency",
    gpu_type = ["H20_CU13"],
    deps = ["//rtp_llm:requests"],
)

# Run the monitor's CPU/GPU acceptance checks without launching a service.
test_suite(
    name = "monitoring_tests",
    tests = [
        ":expert_load_recorder_test",
        ":deepep_router_monitor_test",
        ":expert_load_recorder_ep2_test",
        "//rtp_llm/cpp/models/eplb/test:expert_load_metrics_test",
        "//rtp_llm/cpp/models/eplb/test:expert_load_monitor_test",
        "//rtp_llm/models_py/distributed/test:expert_load_collective_test",
        "//rtp_llm/models_py/modules/factory/fused_moe/defs/test:fused_moe_allreduce_contract_test",
    ],
)
