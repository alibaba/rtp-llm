# NaviSched RTP-LLM Adapter

## 1. Integration Boundary

NaviSched replaces only RTP-LLM's node-selection step. RTP-LLM continues to
own model tokenization, request serialization, model RPC, generation,
streaming, cancellation, KV cache storage, and CUDA execution.

The complete request path is:

1. RTP-LLM frontend tokenizes the request into `GenerateInput.token_ids`.
2. `BackendRPCServerVisitor` calls the NaviSched client when
   `RTP_LLM_MASTER_TYPE=navi_sched`.
3. The adapter passes prompt length, RTP-LLM's rolling block cache keys, and
   request semantics to NaviSched Client. Client sends them to Master through a
   distributed `runGraph` call.
4. Master returns an Agent's advertised model RPC IP and port.
5. The adapter converts that endpoint to RTP-LLM `RoleAddr`.
6. RTP-LLM's existing `ModelRpcClient.enqueue(input)` sends the unchanged
   request through its native RPC protocol.
7. The C++ Agent reports lifecycle, latency, inflight count, and KV cache
   changes to Master through one long-running collector graph.

No model request or generated token passes through NaviSched. This preserves
RTP-LLM's transport behavior and keeps NaviSched independent of model runtime
types.

The integration is verified against RTP-LLM commit
`695eb3e1a08330f96a505ab3cff91f68a6695755` plus the local integration commit
described in section 8.

## 2. Package Boundary

The RTP backend links the standalone NaviSched C++ package. The package
contains public headers, `libnavi_sched.so`, its Python 3.10 and native runtime
libraries, Bazel targets, and a C++ ABI marker. RTP does not import the
NaviSched Python wheel in the backend process.

```bash
cd /home/zq103303/alibaba/aios/navi_sched
./build_release.sh
mkdir -p /home/zq103303/RTP-LLM/github-opensource/.navi_sched/cpp
tar -xzf release/navi_sched_cpp-0.1.0.tar.gz \
  --strip-components=1 \
  -C /home/zq103303/RTP-LLM/github-opensource/.navi_sched/cpp
```

The RTP frontend uses the NaviSched Python wheel because its interception point
is Python. The wheel contains the common C++ implementation; Python only maps
RTP request and address types.

```bash
/opt/conda310/bin/python3 -m pip install --no-deps --force-reinstall \
  /home/zq103303/alibaba/aios/navi_sched/release/\
navi_sched-0.1.0-cp310-cp310-linux_x86_64.whl
```

The wheel declares no Torch dependency and does not replace the framework's
Torch, CUDA, or model packages.

## 3. Client Configuration

FlexLB remains the default. Set `RTP_LLM_MASTER_TYPE=navi_sched` to select the
NaviSched path.

Production deployments configure NaviSched with one JSON environment variable.
`MODEL_SERVICE_CONFIG` remains RTP-LLM's independent service-discovery input:

```bash
export RTP_LLM_NAVI_SCHED_CONFIG_JSON='{
  "log_path":"/home/admin/logs",
  "master":{
    "grpc_port":7003,
    "enable_kmon":false,
    "scheduler_name":"cost",
    "scheduler_config":{
      "lambda":0.0,
      "alpha":512.0,
      "alpha_decay":1.0,
      "min_alpha":0.0,
      "max_loop_count":10,
      "parallelism":0
    },
    "max_wait_schedule_count":30,
    "max_wait_schedule_time_ms":30,
    "runtime_config":{"thread_num":64}
  },
  "agent":{
    "batch_timeout_ms":20,
    "latency_fit_batch_size":4,
    "latency_fit_interval":0,
    "latency_learning_rate":0.011,
    "latency_cold_start_fit_epochs":100,
    "latency_history_limit":2000,
    "latency_persist_interval":256,
    "latency_warmup_sample_count":10,
    "latency_validation":true,
    "latency_model_state_file":"",
    "cache_total_blocks":0
  },
  "client":{"target_role":"PREFILL"}
}'
```

The unified JSON takes precedence over the legacy per-option environment
variables. Those variables remain supported for existing local scripts.

| Environment variable | Meaning |
| --- | --- |
| `RTP_LLM_MASTER_TYPE` | `flexlb` or `navi_sched`; default `flexlb`. |
| `RTP_LLM_NAVI_SCHED_MASTER_BIZ_NAME` | Optional local override; normally derived from `master_endpoint.address`. |
| `RTP_LLM_NAVI_SCHED_BIZ_NAME` | Optional override; normally derived from `prefill_endpoint.address`. |
| `RTP_LLM_NAVI_SCHED_TARGET_ROLE` | Role assigned to the selected `RoleAddr`; default `PREFILL`. |
| `RTP_LLM_NAVI_SCHED_LATENCY_FIT_BATCH_SIZE` | Completed execution batches averaged into one model update; default `4`. |
| `RTP_LLM_NAVI_SCHED_LATENCY_COLD_START_FIT_EPOCHS` | Epochs used to refit retained history after formula or binary mismatch; default `100`. |
| `RTP_LLM_NAVI_SCHED_LATENCY_HISTORY_LIMIT` | Rolling execution-batch history retained for restart recovery; default `2000`. |
| `RTP_LLM_NAVI_SCHED_LATENCY_PERSIST_INTERVAL` | New samples between state-file updates; default `256`. |
| `RTP_LLM_NAVI_SCHED_LATENCY_MODEL_STATE_FILE` | Stable model state path; set this independently of per-process log directories. |

Master discovery reuses `MODEL_SERVICE_CONFIG.master_endpoint`. A production
VipServer domain is passed to Gig subscription directly and remains dynamic;
the adapter does not store resolved IPs. Local tests use
`use_local=true` and a direct `IP:PORT` address:

```bash
export RTP_LLM_MASTER_TYPE=navi_sched
export RTP_LLM_NAVI_SCHED_TARGET_ROLE=PREFILL
export MODEL_SERVICE_CONFIG='{
  "service_id":"test",
  "master_endpoint":{
    "type":"VipServer",
    "address":"127.0.0.1:19010",
    "protocol":"http",
    "path":"/"
  },
  "role_endpoints":[{
    "group":"default",
    "prefill_endpoint":{
      "type":"VipServer",
      "address":"127.0.0.1:18010",
      "protocol":"http",
      "path":"/"
    }
  }],
  "use_local":true
}'
```

The request timeout is selected from RTP's positive TTFT timeout, then its
general request timeout, then a 1000ms default. The Master configuration
selects one scheduler for all requests; the adapter does not carry policy.

The combined Master image accepts the same `RTP_LLM_MASTER_TYPE` switch as
Frontend and Prefill. Put scheduler-specific options in
`NAVI_SCHED_MASTER_CONFIG_JSON`; `NAVI_SCHED_MASTER_CONFIG_FILE` remains a
fallback for local deployments, and JSON takes precedence when both are set.

## 4. Agent Integration

`LocalRpcServer` owns one generic C++ `NaviSchedAgent`. It starts before
`NormalEngine` for every eligible Prefill role, independently of
`RTP_LLM_MASTER_TYPE`. The Master type only selects the frontend routing path,
so switching a Master does not require restarting Prefill. Non-Prefill and
non-primary ranks remain excluded by RTP's role predicate. The Agent uses:

- `server_config.rpc_server_port` as the selected model RPC port;
- `server_config.ip` as the selected model RPC IP, or automatic IP detection
  when empty;
- model RPC port plus 2 as the default collector graph port;
- `prefill_endpoint.address` as the default published biz.

Master gRPC uses the FlexLB Master HTTP port plus 2. Master subscribes to the
Prefill VipServer domain and maps an HTTP endpoint to Agent Gig with `P+3`, or
a gRPC endpoint with `P+2`. Explicit NaviSched address, biz, and port variables
remain local-test overrides rather than production requirements.

`GenerateStream` passes the request's original ordered block hashes to Agent.
Agent predicts the continuous cache-hit prefix from its own cache state;
the framework does not calculate a second prediction. At batch completion,
`NormalEngine` reports the reuse tokens actually applied by execution.
Inflight counting is a core Agent capability and is included in every state
update. `KVCacheManager` and `SharedBlockCache` report one atomic cache delta
containing additions and removals.

The Agent is stopped after request execution has ended. It sends the version-0
EOF control data before closing its long-running graph. Master marks that Agent
inactive after acknowledging the stop. A later process using the same Agent IP
and port publishes a new `navi_sched.agent_instance_r` meta value, so Master
rebuilds its collector before accepting the fresh handshake. When the model state file remains
stable across processes, the new handshake carries the restored formula,
binary version, revision, and nine nonlinear-model parameters.

## 5. Cache Semantics

RTP tokenization and cache identity remain authoritative. Before routing,
RTP-LLM already computes rolling block cache keys for its FlexLB request. The
adapter passes those unsigned 64-bit values unchanged to NaviSched. Cache insert
and eviction hooks pass the same RTP keys to Agent, so Master compares values
from one framework-owned key space. NaviSched does not tokenize, choose block
boundaries, or calculate production cache hashes.

Agent keeps the complete local cache key set. A new Master collector receives a
handshake followed by a full cache snapshot, including an empty snapshot; later
updates are incremental. This lets a restarted Master rebuild exact Agent cache
state without a framework-specific initialization API.

High-frequency reports contain integer hashes and counters rather than repeated
strings. Handshake data carries the server endpoint, formula state, and version.
The current protocol version is `0`; no block-size or hash-algorithm negotiation
and no compatibility branch are active in this first version.

The real test runs two identical prompts. The second request must expose a
positive RTP `reuse_len` or `usage.prompt_tokens_details.cached_tokens`, and
Master must expose a positive `cached_block_count`.

## 6. Scheduling Policy Boundary

The scheduler Kernel selected in the Master graph handles cache scoring,
latency prediction, load feedback, batch grouping, and future policies. The
framework adapter only submits request features and uses the returned target.

## 7. Build and Runtime

The verified RTP build uses CUDA 12.9 and disables the optional remote KV cache
library, whose current prebuilt artifact requires CUDA 13:

```bash
cd /home/zq103303/RTP-LLM/github-opensource
bazelisk build //rtp_llm:rtp_llm //rtp_llm:rtp_llm_package_libs \
  --verbose_failures \
  --config=debug \
  --config=cuda12_9 \
  --config=sm9x \
  --define=use_remote_kv_cache=false \
  --test_output=errors \
  --test_env=LOG_LEVEL=INFO \
  --jobs=128
```

Put `/usr/local/nvidia/lib64` before `/usr/lib64`. Do not prepend CUDA 11.8 or
`/usr/local/cuda/compat` to this CUDA 12.9 build.

For the tested Qwen2.5 model, RTP's current FlashInfer reuse-cache path raises
an illegal-memory-access error on the second repeated request. The framework's
supported `DISABLE_FLASH_INFER=1` setting selects its stable attention path
without changing NaviSched or cache semantics. With that setting, repeated
cache-enabled requests complete and report the exact reuse count.

## 8. Verification

RTP adapter unit tests:

```bash
cd /home/zq103303/RTP-LLM/github-opensource
PYTHONDONTWRITEBYTECODE=1 /opt/conda310/bin/python3 \
  -m unittest -v rtp_llm.server.test.navi_sched_adapter_test
```

Real GPU end-to-end test:

```bash
cd /home/zq103303/alibaba/aios/navi_sched
/opt/conda310/bin/python3 test/rtp_llm/RtpLLMEndToEndTest.py \
  --rtp-source /home/zq103303/RTP-LLM/github-opensource \
  --model /home/zq103303/Qwen2.5-0.5B-Instruct \
  --gpu 6
```

The test verifies framework tokenization, Master routing, native model RPC,
generated output, repeated-request cache reuse, Agent reports, EOF removal,
same-address restart, and a request after restart. Logs and snapshots remain in
`tmp/rtp_llm_e2e`.

The 2026-07-29 and 2026-07-30 release verifications used the Qwen2.5 0.5B
model on H20 GPUs. The latest run used the packaged NaviSched wheel and the
RTP-LLM C++ server linked against the matching standalone C++ package.
The single-node run verified grouped execution, partial-batch timeout, a late
request, 768 reused KV tokens, 14 synchronized cache blocks, model revision 2,
Agent removal, same-address restart, and restoration of the fitted model
revision and nine parameters. The two-node run verified both Agents
online, four routed requests, removal of one endpoint, successful failover to
the remaining endpoint and restoration of both collectors after restart.
