# FlexLB Online Evaluation

This tool chain evaluates a running SpringBoot `flexlb-api` against a mock
rtp-llm engine cluster. It is intended to run inside the `luoli_gpu` container,
where `grpcio`, `grpcio-tools`, and `protobuf` are available.

## One-command run

Run inside `luoli_gpu`:

```bash
docker exec -it luoli_gpu bash
cd <repo-root>

rtp_llm/flexlb/tools/online_eval/run_online_eval.sh
```

The run directory defaults to `rtp_llm/flexlb/tools/online_eval/run/<timestamp>/`.
After completion, the important outputs are:

- `load_client/summary.json`
- `load_client/per_request.jsonl`
- `load_client/report.md`
- `mock_engine.log`
- `flexlb.log`

Common overrides:

```bash
PROCESS_CONFIG_FILE=rtp_llm/flexlb/tools/online_eval/data/config/master_fixed_window.json \
DURATION_S=300 \
LIMIT=5000 \
REPLAY_SPEED=20 \
N_PREFILL=4 \
N_DECODE=16 \
SLA_TTFT_MS=800 \
FLEXLB_CONFIG='{"deploy":"DISAGGREGATED","loadBalanceStrategy":"SHORTEST_TTFT","prefillBatchWaitTimeMs":200,"kvCache":"LOCAL_STATIC","staticCacheBlockSize":1024,"batchSize":32,"prefillLbTimeoutMs":300,"prefillGenerateTimeoutMs":30000,"enableGrpcPrefillMaster":true,"decodeConcurrencyLimit":64}' \
rtp_llm/flexlb/tools/online_eval/run_online_eval.sh
```

If `flexlb-api` is already running, use:

```bash
START_FLEXLB=0 \
FLEXLB_HTTP_ADDR=127.0.0.1:7001 \
rtp_llm/flexlb/tools/online_eval/run_online_eval.sh
```

If the default jar is not built, the script runs `./mvnw -pl flexlb-api -am package -DskipTests`.
The script auto-selects Java 21 from system alternatives when available; otherwise set `JAVA21_HOME` or `JAVA_HOME`.
It also defaults to `MAVEN_PROFILES=opensource,!internal` so an adjacent `internal_source` directory does not accidentally activate internal-only dependencies.

## Data layout

- `data/online_logs/trace_30min.jsonl`: sanitized replay shape derived from online logs.
- `data/online_logs/pod1_arrivals.tsv`: sanitized relative-time arrival analysis source.
- `data/online_logs/sample_access.json`: sanitized request-shape fixture with pseudonymous token IDs.
- `data/performance/dsv4_flash_performance.sample.json`: mock latency model.
- `data/config/master_fixed_window.json`: master process env config for the fixed-window baseline.

## Manual flow

### 1. Start mock engines

```bash
java -jar rtp_llm/flexlb/flexlb-mock-engine/target/flexlb-mock-engine-1.0.0-SNAPSHOT-all.jar \
  --n-prefill 2 \
  --n-decode 4 \
  --base-grpc-port 55151 \
  --performance rtp_llm/flexlb/tools/online_eval/data/performance/dsv4_flash_performance.sample.json \
  --endpoint-file rtp_llm/flexlb/tools/online_eval/run/endpoints.json \
  --env-file rtp_llm/flexlb/tools/online_eval/run/flexlb_env.txt
```

The cluster writes:

- `rtp_llm/flexlb/tools/online_eval/run/endpoints.json`
- `rtp_llm/flexlb/tools/online_eval/run/flexlb_env.txt`

Use the `env ... <your-flexlb-api-start-command>` snippet from
`flexlb_env.txt` when starting `flexlb-api`. The `DOMAIN_ADDRESS:*`
environment keys contain `:`, so they must be passed through `env`; bash cannot
`export` them directly.

The cluster also serves an HTTP control API on `--base-grpc-port - 1` (55150 here);
see `flexlb-mock-engine/README.md` for the endpoint schema.

### 2. Start flexlb-api

Start the full SpringBoot `flexlb-api` with the environment variables generated
by the mock cluster. FlexLB's own gRPC port is `server.port + 2`.

Backend mock engines use the rtp-llm convention `http_port + 1 == grpc_port`.
The generated service route uses `"protocol": "http"`, so FlexLB treats the
service discovery port as the engine HTTP port and derives gRPC as `http + 1`.

### 3. Run load client

```bash
TRACE_FILE=rtp_llm/flexlb/tools/online_eval/data/online_logs/trace_30min.jsonl \
TARGET_ADDR=127.0.0.1:7001 \
REPLAY_SPEED=10 \
LIMIT=1000 \
OUTPUT_DIR=rtp_llm/flexlb/tools/online_eval/run/load_client \
java -cp rtp_llm/flexlb/flexlb-mock-engine/target/flexlb-mock-engine-1.0.0-SNAPSHOT-all.jar \
  org.flexlb.mockengine.JavaLoadClient
```

The client is configured entirely through environment variables
(`JavaLoadClient.Config.fromEnv`); the full list lives in
`flexlb-mock-engine/README.md`.

For master-enqueued batch requests, the client follows the frontend behavior:
it calls `FetchResponse` on the selected prefill engine. For direct requests, it
calls `GenerateStreamCall`.

Outputs:

- `summary.json`: throughput, latency percentiles, SLA violations, load balance.
- `per_request.jsonl`: one row per request with routing and latency details.
- `report.md`: readable report for comparing FlexLB configs.

## Validation

```bash
python3 -m unittest discover -s rtp_llm/flexlb/tools/online_eval/tests
```

Raw online logs must not be committed. Generate sanitized fixtures before adding data:

```bash
node rtp_llm/flexlb/tools/online_eval/sanitize_online_log_fixtures.mjs \
  /path/to/raw/online_logs \
  rtp_llm/flexlb/tools/online_eval/data/online_logs
```

The sanitizer drops request/header identity data, converts timestamps to relative
time, pseudonymizes endpoints and block hashes, and remaps plus shuffles token IDs.
The random mapping is not written to disk.

## Capacity reading

Use `completed_qps`, not only `offered_qps`, as the throughput signal. A config
is healthy only if:

- `completed_qps` tracks offered load;
- TTFT p99 stays under the target SLA;
- error and timeout rate remain low;
- prefill/decode load distribution is not strongly skewed;
- mock engine running/available-KV snapshots do not show unbounded backlog.

For capacity search, run the same trace with increasing `REPLAY_SPEED` or use
different trace slices. The practical capacity point is the highest completed
QPS before TTFT p99, error rate, or queue backlog bends upward.
