load("//rtp_llm/test/smoke:defs.bzl", "smoke_test")
load("//deps:kvcm.bzl", "KVCM_SOURCE_LOCK")

_PACE_SOURCE_ID = ":".join([KVCM_SOURCE_LOCK[key] for key in [
    "internal_commit", "opensource_commit", "pace_commit",
]])

def _pace_smoke(name, task_info, smoke_args, gpu_type="L20_CU13", backend="pace",
                kvcm_envs=[], kill_remote=False, metadata_check=True, sleep_time_qr=10,
                model_events_check=False):
    return smoke_test(
        name = name,
        task_info = task_info,
        smoke_args = smoke_args,
        gpu_type = [gpu_type],
        tags = ["requires-pace", "no-remote"],
        data = [
            "//3rdparty/remote_kv_cache_manager:remote_kv_cache_manager_server_bin",
            "@remote_kv_cache_manager_server//:KVCM_SOURCE_ID",
            "@remote_kv_cache_manager_server//:KVCM_ARTIFACT_SHA256",
            "@remote_kv_cache_manager_server//:etc/default_startup_config.json",
            "@remote_kv_cache_manager_client_rpm//:KVCM_SOURCE_ID",
            "@remote_kv_cache_manager_client_rpm//:KVCM_ARTIFACT_SHA256",
            "@remote_kv_cache_manager_client_rpm//:KVCM_CLIENT_VARIANT",
        ],
        kvcm_envs = ["PACE_REQUIRED=true", "PACE_BACKEND=" + backend,
                     "PACE_METADATA_CHECK=" + str(metadata_check),
                     "PACE_MODEL_EVENTS_CHECK=" + str(model_events_check), "KVCM_LOG_LEVEL=DEBUG"] + kvcm_envs,
        kill_remote = kill_remote,
        sleep_time_qr = sleep_time_qr,
        test_env = {"KVCM_EXPECTED_SOURCE_ID": _PACE_SOURCE_ID, "KVCM_SMOKE_CLIENT_VARIANT": "cuda"},
        env_inherit = ["KVCM_PACE_FIXTURE"],
        deps = ["//rtp_llm/cpp/model_rpc/proto:model_rpc_service_py_proto", "//rtp_llm/cpp/model_rpc:grpcio"],
    )

def _pace_suites():
    base = "--warm_up 0 --reuse_cache 1 --act_type FP16 --seq_size_per_block 8 --enable_remote_cache true" + REMOTE_CACHE_DEVICE_STORE_ARGS
    tests = []
    for suffix, task, args, config, check in [
        ("basic", "q_r_l20_remote_cache.json", "", [], True),
        ("batch", "q_r_l20_remote_cache.json", " --kvcm_default_query_type 1 --kvcm_query_type 0", [], True),
        ("backend", "q_r_l20_remote_cache.json", " --kvcm_read_backend_type 3 --kvcm_min_replica_count 1", [], True),
        ("tp2", "q_r_l20_remote_cache_tpsize_2.json", " --tp_size 2", [], True),
        ("edge", "q_r_l20_cache_edge_case_1_remote_cache.json", " --seq_size_per_block 4", [], True),
        ("match_fail", "q_r_l20_remote_cache_match_failure.json", "", ["ENABLE_DEBUG_SERVICE=true", "TEST_MATCH_FAILURE=1"], False),
        ("write_start_fail", "q_r_l20_remote_cache_start_and_finish_failure.json", "", ["ENABLE_DEBUG_SERVICE=true", "TEST_START_WRITE_FAILURE=1"], False),
        ("write_finish_fail", "q_r_l20_remote_cache_start_and_finish_failure.json", "", ["ENABLE_DEBUG_SERVICE=true", "TEST_FINISH_WRITE_FAILURE=1"], False),
    ]:
        tests.append(_pace_smoke(
            "remote_cache_pace_" + suffix, "data/model/qwen25/" + task,
            base + args, kvcm_envs = config, metadata_check = check, sleep_time_qr = 20,
        ))
    tests.append(_pace_smoke(
        "remote_cache_pace_publisher", "data/model/qwen25/worker_status_reuse_cache.json",
        # Keep local keys resident so the test can verify real HBM announcements.
        "--warm_up 0 --reuse_cache 1 --act_type FP16 --seq_size_per_block 8 --enable_remote_cache true --enable_device_cache 1 --enable_memory_cache 0 --enable_disk_cache 0 --test_block_num 500",
        model_events_check = True, sleep_time_qr = 1,
    ))
    tests.append(_pace_smoke(
        "remote_cache_pace_kill", "data/model/qwen25/q_r_l20_remote_cache_kill_remote.json",
        base, kill_remote = True, metadata_check = False,
    ))
    tests.append(_pace_smoke(
        "remote_cache_pace_pd", "data/model/qwen25/q_r_l20_remote_cache_pd_sep.json",
        {
            "prefill": base + " --role_type PREFILL",
            "decode": base + " --role_type DECODE",
        }, metadata_check = False, sleep_time_qr = 20,
    ))
    tests.append(_pace_smoke(
        "remote_cache_pace_hybrid_tp2",
        "data/model/qwen3_next/q_r_next_fp8_tp2_long_input_reuse_remote_only.json",
        "--tp_size 2 --act_type BF16 --seq_size_per_block 2048 --linear_step 2 --reuse_cache 1 --enable_remote_cache 1 --kvcm_default_query_type 4 --kvcm_query_type 4 --kvcm_put_timeout_ms 17000 --kvcm_get_timeout_ms 17000 --kvcm_get_broadcast_timeout 20000 --kvcm_put_broadcast_timeout 20000" + REMOTE_CACHE_DEVICE_STORE_ARGS,
        gpu_type = "H20_CU13", sleep_time_qr = 20,
    ))
    native.test_suite(name = "smoke_cuda_remote_cache_pace", tests = tests)
    native.test_suite(
        name = "smoke_kvcm_p1_gpu",
        tests = [":smoke_cuda_remote_cache_pace"] + [
            "//rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test:" + name
            for name in ["client_wrapper_test", "kvcm_config_test", "kvcm_internal_test",
                         "kvcm_mock_only_full_test", "kvcm_mock_full_linear_test", "kvcm_independent_pool_test"]
        ] + ["//rtp_llm/cpp/cache/events/test:kv_cache_event_publisher_test"],
    )
    native.test_suite(name = "smoke_kvcm_p1_cpu", tests = [":remote_cache_pace_contract"])
    native.test_suite(name = "smoke_kvcm_p1_cpu_ssd", tests = [":remote_cache_pace_ssd_contract"])
    native.test_suite(
        name = "smoke_kvcm_p1_gpu_ssd",
        tests = [_pace_smoke(
            "remote_cache_pace_ssd", "data/model/qwen25/q_r_l20_remote_cache.json",
            base + " --kvcm_read_backend_type 9", backend = "pace_ssd",
        )],
    )

# Remote uploads originate from DEVICE inserts. With no lower local tier, drop
# every DEVICE tree entry after insertion so later queries must read the backend.
# 500 physical blocks leave 499 usable: ceil(499 * 0.002) = 1 triggers eviction,
# and floor(499 * 0.001) = 0 retains no local entries. Request/storage references
# keep the physical payload alive while the remote upload completes.
REMOTE_CACHE_DEVICE_STORE_ARGS = (
    " --enable_device_cache 1 --enable_memory_cache 0 --enable_disk_cache 0" +
    " --test_block_num 500" +
    " --block_tree_device_evict_low_watermark_ratio 0.001" +
    " --block_tree_device_evict_high_watermark_ratio 0.002"
)

def remote_cache_suites():
    _pace_suites()

    # PPU Remote Cache (with KVCM server)
    native.test_suite(
        name = "smoke_cuda_remote_cache",
        tests = [
            smoke_test(
                name = "remote_cache_basic",
                data = ["//3rdparty/remote_kv_cache_manager:remote_kv_cache_manager_server_bin"],
                gpu_type = ["L20_CU13"],
                kvcm_envs = ["SEQ_SIZE_PER_BLOCK=8", "KVCM_LOG_LEVEL=DEBUG"],
                # Exact reuse counts require the previous asynchronous upload to be published.
                sleep_time_qr = 10,
                smoke_args = "--warm_up 0 --reuse_cache 1 --act_type FP16 --seq_size_per_block 8 --enable_remote_cache true" + REMOTE_CACHE_DEVICE_STORE_ARGS,
                task_info = "data/model/qwen25/q_r_l20_remote_cache.json",
            ),
            smoke_test(
                name = "remote_cache_basic_async",
                data = ["//3rdparty/remote_kv_cache_manager:remote_kv_cache_manager_server_bin"],
                gpu_type = ["L20_CU13"],
                kvcm_envs = ["SEQ_SIZE_PER_BLOCK=8", "KVCM_LOG_LEVEL=DEBUG"],
                sleep_time_qr = 10,
                smoke_args = "--warm_up 0 --reuse_cache 1 --act_type FP16 --seq_size_per_block 8 --enable_remote_cache true" + REMOTE_CACHE_DEVICE_STORE_ARGS,
                task_info = "data/model/qwen25/q_r_l20_remote_cache.json",
            ),
            smoke_test(
                name = "remote_cache_kill",
                data = ["//3rdparty/remote_kv_cache_manager:remote_kv_cache_manager_server_bin"],
                gpu_type = ["L20_CU13"],
                kill_remote = True,
                kvcm_envs = ["SEQ_SIZE_PER_BLOCK=8", "KVCM_LOG_LEVEL=DEBUG"],
                sleep_time_qr = 10,
                smoke_args = "--warm_up 0 --reuse_cache 1 --act_type FP16 --seq_size_per_block 8 --enable_remote_cache true" + REMOTE_CACHE_DEVICE_STORE_ARGS,
                task_info = "data/model/qwen25/q_r_l20_remote_cache_kill_remote.json",
            ),
            smoke_test(
                name = "remote_cache_tp2",
                data = ["//3rdparty/remote_kv_cache_manager:remote_kv_cache_manager_server_bin"],
                gpu_type = ["L20_CU13"],
                kvcm_envs = ["SEQ_SIZE_PER_BLOCK=8", "KVCM_LOG_LEVEL=DEBUG"],
                sleep_time_qr = 20,
                smoke_args = "--warm_up 0 --reuse_cache 1 --act_type FP16 --seq_size_per_block 8 --tp_size 2 --enable_remote_cache true --kvcm_put_timeout_ms 12000 --kvcm_get_timeout_ms 12000 --kvcm_get_broadcast_timeout 15000 --kvcm_put_broadcast_timeout 15000" + REMOTE_CACHE_DEVICE_STORE_ARGS,
                task_info = "data/model/qwen25/q_r_l20_remote_cache_tpsize_2.json",
            ),
            smoke_test(
                name = "remote_cache_pd",
                data = ["//3rdparty/remote_kv_cache_manager:remote_kv_cache_manager_server_bin"],
                gpu_type = ["L20_CU13"],
                kvcm_envs = ["SEQ_SIZE_PER_BLOCK=8", "KVCM_LOG_LEVEL=DEBUG"],
                sleep_time_qr = 20,
                smoke_args = {
                    "prefill": "--warm_up 0  --reuse_cache 1 --role_type PREFILL --act_type FP16 --seq_size_per_block 8 --enable_remote_cache true --kvcm_put_timeout_ms 12000 --kvcm_get_timeout_ms 12000 --kvcm_get_broadcast_timeout 15000 --kvcm_put_broadcast_timeout 15000" + REMOTE_CACHE_DEVICE_STORE_ARGS,
                    "decode": "--warm_up 0  --reuse_cache 1 --role_type DECODE --act_type FP16 --seq_size_per_block 8 --enable_remote_cache true --kvcm_put_timeout_ms 12000 --kvcm_get_timeout_ms 12000 --kvcm_get_broadcast_timeout 15000 --kvcm_put_broadcast_timeout 15000" + REMOTE_CACHE_DEVICE_STORE_ARGS,
                },
                task_info = "data/model/qwen25/q_r_l20_remote_cache_pd_sep.json",
            ),
            smoke_test(
                name = "remote_cache_match_fail",
                data = ["//3rdparty/remote_kv_cache_manager:remote_kv_cache_manager_server_bin"],
                gpu_type = ["L20_CU13"],
                kvcm_envs = [
                    "SEQ_SIZE_PER_BLOCK=8",
                    "KVCM_LOG_LEVEL=DEBUG",
                    "ENABLE_DEBUG_SERVICE=TRUE",
                    "TEST_MATCH_FAILURE=1",
                ],
                sleep_time_qr = 10,
                smoke_args = "--warm_up 0 --reuse_cache 1 --act_type FP16 --seq_size_per_block 8 --enable_remote_cache true" + REMOTE_CACHE_DEVICE_STORE_ARGS,
                task_info = "data/model/qwen25/q_r_l20_remote_cache_match_failure.json",
            ),
            smoke_test(
                name = "remote_cache_write_start_fail",
                data = ["//3rdparty/remote_kv_cache_manager:remote_kv_cache_manager_server_bin"],
                gpu_type = ["L20_CU13"],
                kvcm_envs = [
                    "SEQ_SIZE_PER_BLOCK=8",
                    "KVCM_LOG_LEVEL=DEBUG",
                    "ENABLE_DEBUG_SERVICE=TRUE",
                    "TEST_START_WRITE_FAILURE=1",
                ],
                sleep_time_qr = 10,
                smoke_args = "--warm_up 0 --reuse_cache 1 --act_type FP16 --seq_size_per_block 8 --enable_remote_cache true" + REMOTE_CACHE_DEVICE_STORE_ARGS,
                task_info = "data/model/qwen25/q_r_l20_remote_cache_start_and_finish_failure.json",
            ),
            smoke_test(
                name = "remote_cache_write_finish_fail",
                data = ["//3rdparty/remote_kv_cache_manager:remote_kv_cache_manager_server_bin"],
                gpu_type = ["L20_CU13"],
                kvcm_envs = [
                    "SEQ_SIZE_PER_BLOCK=8",
                    "KVCM_LOG_LEVEL=DEBUG",
                    "ENABLE_DEBUG_SERVICE=TRUE",
                    "TEST_FINISH_WRITE_FAILURE=1",
                ],
                sleep_time_qr = 10,
                smoke_args = "--warm_up 0 --reuse_cache 1 --act_type FP16 --seq_size_per_block 8 --enable_remote_cache true" + REMOTE_CACHE_DEVICE_STORE_ARGS,
                task_info = "data/model/qwen25/q_r_l20_remote_cache_start_and_finish_failure.json",
            ),
            smoke_test(
                name = "remote_cache_edge",
                data = ["//3rdparty/remote_kv_cache_manager:remote_kv_cache_manager_server_bin"],
                gpu_type = ["L20_CU13"],
                kvcm_envs = ["SEQ_SIZE_PER_BLOCK=4", "KVCM_LOG_LEVEL=DEBUG"],
                sleep_time_qr = 10,
                smoke_args = "--warm_up 0  --reuse_cache 1 --act_type FP16 --seq_size_per_block 4 --enable_remote_cache true" + REMOTE_CACHE_DEVICE_STORE_ARGS,
                task_info = "data/model/qwen25/q_r_l20_cache_edge_case_1_remote_cache.json",
            ),
        ],
    )
