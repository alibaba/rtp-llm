# Multi-Node Perf Benchmark Refactor Design

## Goal

Refactor the performance benchmark scripts under `rtp_llm/test/perf_test/multi_node/` so that `local_server_runner.py` and its dependencies are fully self-contained within the `multi_node/` directory. This eliminates coupling with `rtp_llm/test/perf_test/` level test files (which are used by smoke tests) and prevents unintended side effects.

The initial focus is single-node benchmarking, but the multi-node coordination framework (TCPStore, gang_config, barrier) from the reference commit is preserved for future expansion.

## Reference

- Reference commit: https://github.com/yykzjh/rtp-llm/commit/20b40a19a98dde4897cd0e090cdb344e114924e4
- The reference commit modifies shared files under `rtp_llm/test/perf_test/`; this refactor copies the needed logic into `multi_node/` instead.

## Directory Structure

After refactoring, `rtp_llm/test/perf_test/multi_node/` will contain:

```
rtp_llm/test/perf_test/multi_node/
├── local_server_runner.py      # Refactored main entry (enhanced per reference commit)
├── server_manager.py           # Copied from maga_server_manager.py, simplified
├── perf_runner.py              # Extracted run_single() from batch_decode_test.py
├── perf_impl.py                # Copied BatchPerfImpl from batch_perf_impl.py
├── perf_dataclass.py           # Copied data classes from dataclass.py
├── perf_util.py                # Copied create_query/get_prompt from test_util.py (no ODPS)
├── multi_benchmark_config.yaml # Unchanged
├── multi_benchmark.py          # Unchanged
├── multi_runner.sh             # Unchanged
├── multi_local_executor.sh     # Unchanged
└── analyzer/                   # Unchanged
```

## File Specifications

### perf_dataclass.py

**Source:** `rtp_llm/test/perf_test/dataclass.py`

**Contents (copied as-is):**
- `ResponseInfo` class — parses HTTP response from the inference server
- `TestResultMetrics` dataclass — aggregated performance metrics
- `analyze_results()` — computes avg/max/var metrics from a list of `ResponseInfo`
- `MetricState` class — associates metrics with input_len and batch_size
- `TableType` enum — Prefill vs Decode
- `create_metrics_table()` — generates PrettyTable report and dumps JSON

**Dependencies:** `prettytable`

### perf_util.py

**Source:** `rtp_llm/test/perf_test/test_util.py`

**Contents (subset, ODPS removed):**
- `_load_tokenizer()` — loads tokenizer with GLM-5 compatible path
- `get_prompt()` — generates a prompt of exact token length
- `create_query()` — builds test input data for each input_len using binary search

**Removed:** `write_odps()` function and `odps` dependency

**Dependencies:** `transformers`, `rtp_llm.utils.fuser.fetch_remote_file_to_local`

### perf_impl.py

**Source:** `rtp_llm/test/perf_test/batch_perf_impl.py` (reference commit version)

**Contents (reference commit version with multi-node support):**
- `_curl_server_single_worker()` — sends a single HTTP request to the server; uses `(host, port)` from `tp0_endpoints` instead of hardcoded `127.0.0.1`
- `_curl_server_batch_worker()` — ThreadPoolExecutor-based concurrent request handler
- `BatchPerfImpl` class:
  - Constructor accepts `gang_config_string`, `local_world_size`, `request_tpot`, `connection_timeout`, `retry_times`, `retry_interval`
  - `_get_all_dp_tp0_frontends()` — discovers all DP groups' tp_rank==0 endpoints from gang_config_string
  - `_set_concurrency()` — multi-threaded POST to `/update_scheduler_info` on all endpoints with retry
  - `run()` — executes warmup → measure → profile (3 passes)
  - `_curl_server()` — distributes requests across processes

**Dependencies:** `requests`, `rtp_llm.distribute.distributed_server.members_from_test_env`, `rtp_llm.utils.util.check_with_info`, local `perf_dataclass`

### perf_runner.py

**Source:** `rtp_llm/test/perf_test/batch_decode_test.py` (reference commit version)

**Contents (only `run_single()` function, reference commit version):**
- `run_single()` — orchestrates benchmark execution:
  - Accepts `gang_config_string`, `local_world_size`, `request_tpot`, `connection_timeout`, `retry_times`, `retry_interval` parameters
  - Iterates over (batch_size, input_len) combinations
  - Creates `BatchPerfImpl` instances and runs them
  - Generates metrics table and JSON output

**Removed:** `RunningConfig`, `write_odps_wrapper`, `start_server`, `parse_args`, `merge_state`, `run_normal_test`, `run_disaggregate_test`, `create_test_env`, `main` — all smoke-test-specific logic

**Dependencies:** `tqdm`, local `perf_impl`, local `perf_dataclass`

### server_manager.py

**Source:** `rtp_llm/test/utils/maga_server_manager.py`

**Contents (copied and simplified):**
- `LocalServerManager` class (renamed from `MagaServerManager`):
  - `__init__(port, log_dir)` — takes port and log directory
  - `start_server()` — launches `python -m rtp_llm.start_server` via `subprocess.Popen`, redirects logs to file
  - `stop_server()` — recursively kills all child processes using `psutil`
  - `print_process_log()` — prints process log for debugging
  - `wait_sever_done()` — polls `/health` endpoint until server is ready

**Removed:**
- `get_free_port()` / `PortManager` dependency — port is provided via env var `START_PORT`
- `visit()` — not needed for benchmark
- `smoke_args_str` / `_smoke_args_str` — smoke test specific
- `_role_name` complexity — simplified to single role

**Dependencies:** `psutil`, `requests`, `subprocess`

### local_server_runner.py

**Source:** Current `local_server_runner.py` + reference commit enhancements

**Key changes from current version:**
1. Imports changed from `rtp_llm.test.perf_test.*` to local modules (`perf_runner`, `perf_util`, etc.)
2. Import `LocalServerManager` from local `server_manager` instead of `MagaServerManager`
3. Added `setup_logging()` + `patch_logging_stream_handler()` (from reference commit)
4. Added TCPStore-based coordination (`_init_startup_store`, `_store_set_safe`, `_store_check_failed`, `_store_check_ok`, `_store_barrier`)
5. Added `wait_world_server_startup()` — health check all servers across nodes
6. Enhanced `wait_master_done()` with configurable retry parameters
7. Removed `test_main()` wrapper — calls `run_single()` directly
8. Main block restructured per reference commit flow

**Dependencies (all framework-level, no test-level deps):**
- `rtp_llm.config.log_config.setup_logging`
- `rtp_llm.config.py_config_modules.PyEnvConfigs`
- `rtp_llm.config.server_config_setup.setup_and_configure_server`
- `rtp_llm.distribute.distributed_server.members_from_test_env`
- `rtp_llm.server.server_args.server_args.setup_args`
- `rtp_llm.utils.fuser.fetch_remote_file_to_local` (in perf_util.py)
- `rtp_llm.utils.util.check_with_info` (in perf_impl.py)
- `rtp_llm.utils.import_util.has_internal_source` (existing)
- `torch.distributed.TCPStore`

## Execution Flow (Single Node)

```
Environment variables → local_server_runner.py __main__
    ├── 1. setup_logging() + patch_logging_stream_handler()
    ├── 2. Read env vars (BATCH_SIZE_LIST, INPUT_LEN_LIST, IS_DECODE, etc.)
    ├── 3. Set runtime env vars (GEN_TIMELINE_SYNC, MAX_SEQ_LEN, FAKE_BALANCE_EXPERT, etc.)
    ├── 4. setup_args() + setup_and_configure_server() → py_env_configs
    ├── 5. Create log/output directory
    ├── 6. Generate gang_config_string (single node: "name:perf_part0,ip:127.0.0.1,port:{START_PORT}")
    ├── 7. create_query() → build test input data
    ├── 8. Init TCPStore (single node: self is master and only worker)
    ├── 9. LocalServerManager.start_server() → launch service
    ├── 10. wait_world_server_startup() → wait for all services ready
    ├── 11. _store_barrier() → cross-node sync (single node: passes immediately)
    ├── 12. run_single() → execute benchmark
    │       ├── For each (batch_size, input_len):
    │       │   └── BatchPerfImpl.run()
    │       │       ├── _set_concurrency() → POST /update_scheduler_info
    │       │       ├── _curl_server(warmup) → warm up
    │       │       ├── _curl_server(measure) → measure performance
    │       │       └── _curl_server(profile) → dump trace JSON
    │       └── create_metrics_table() → PrettyTable report + JSON file
    ├── 13. server.stop_server()
    └── 14. script_exit()
```

## Output Artifacts

In `log_dir_path` directory:
- `Decode_Result.json` / `Prefill_Result.json` — JSON performance metrics
- `normal_*.json` — torch profiler trace JSON files
- `main_logs/process.log` — server process log

## Environment Variables

### Required (user must set):

**Model config:**
- `MODEL_TYPE` — e.g., `qwen35_moe`
- `TOKENIZER_PATH` — local path to tokenizer
- `CHECKPOINT_PATH` — local path to model checkpoint
- `QUANTIZATION` — e.g., `FP4_PER_GROUP_QUARK`
- `ACT_TYPE` — e.g., `bf16`
- `HACK_LAYER_NUM` — e.g., `5`

**Parallelism config:**
- `WORLD_SIZE`, `DP_SIZE`, `TP_SIZE`, `EP_SIZE`
- `HIP_VISIBLE_DEVICES` (or `CUDA_VISIBLE_DEVICES`)

**Test config:**
- `BATCH_SIZE_LIST` — e.g., `[128]`
- `INPUT_LEN_LIST` — e.g., `[2048]`
- `IS_DECODE` — `1` or `0`
- `DECODE_TEST_LENGTH` — e.g., `20`
- `START_PORT` — e.g., `10666`
- `CONCURRENCY_LIMIT` — e.g., `128`

**Performance tuning:**
- `SEQ_SIZE_PER_BLOCK`, `KERNEL_SEQ_SIZE_PER_BLOCK`
- `ENABLE_CUDA_GRAPH`, `DECODE_CAPTURE_CONFIG`
- `DEVICE_RESERVE_MEMORY_BYTES`, `RESERVER_RUNTIME_MEM_MB`
- `MAX_SEQ_LEN`
- `LOAD_PYTHON_MODEL`, `LOAD_METHOD`

**Framework config:**
- `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `LD_PRELOAD`
- NCCL/RCCL related env vars

### Auto-set by local_server_runner.py:
- `USE_BATCH_DECODE_SCHEDULER=1`
- `FAKE_BALANCE_EXPERT=1`
- `GEN_TIMELINE_SYNC=1`
- `MAX_SEQ_LEN` (computed from INPUT_LEN_LIST + DECODE_TEST_LENGTH)
- `WORKER_INFO_PORT_NUM=10`
- `TORCH_CUDA_PROFILER_DIR`

## Impact Analysis

### Files NOT modified (protecting smoke tests):
- `rtp_llm/test/perf_test/batch_decode_test.py`
- `rtp_llm/test/perf_test/batch_perf_impl.py`
- `rtp_llm/test/perf_test/dataclass.py`
- `rtp_llm/test/perf_test/test_util.py`
- `rtp_llm/test/perf_test/test_entry.py`
- `rtp_llm/test/perf_test/defs.bzl` / `BUILD`
- `rtp_llm/test/utils/maga_server_manager.py`

### Files NOT modified in multi_node/:
- `multi_benchmark.py`
- `multi_benchmark_config.yaml`
- `multi_runner.sh`
- `multi_local_executor.sh`
- `analyzer/`

### Modified files:
- `rtp_llm/test/perf_test/multi_node/local_server_runner.py` — refactored main entry

### New files:
- `rtp_llm/test/perf_test/multi_node/server_manager.py`
- `rtp_llm/test/perf_test/multi_node/perf_runner.py`
- `rtp_llm/test/perf_test/multi_node/perf_impl.py`
- `rtp_llm/test/perf_test/multi_node/perf_dataclass.py`
- `rtp_llm/test/perf_test/multi_node/perf_util.py`

## Testing Plan

Test on ssh 211 remote machine in the container environment set up via `setup-mi355x-env` skill:

1. Set all required environment variables (model, parallelism, test config, performance tuning, framework)
2. Run `python rtp_llm/test/perf_test/multi_node/local_server_runner.py`
3. Verify:
   - Service starts successfully
   - Benchmark data is sent and processed
   - PrettyTable performance report is printed to stdout
   - Output directory contains `*_Result.json` files
   - Output directory contains `normal_*.json` trace files
