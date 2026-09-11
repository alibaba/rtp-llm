# Sleep and wake up

Sleep releases registered GPU memory without restarting the backend processes.
Keep the instance out of service discovery throughout sleep and wake. For
prefill/decode deployments, coordinate both roles before restoring traffic.
The lifecycle endpoints are administrative APIs: restrict access to trusted
control-plane callers; do not expose them to inference clients.

## Startup configuration

| CLI argument | Environment variable | Default |
| --- | --- | --- |
| `--enable-sleep-mode` | `ENABLE_SLEEP_MODE` | `0` (disabled) |
| `--sleep-mode-level` | `SLEEP_MODE_LEVEL` | `1` |
| `--sleep-release-collective-memory` | `SLEEP_RELEASE_COLLECTIVE_MEMORY` | `0` |

Underscored argument aliases are also accepted. Set the same options on every
backend rank. CLI values override environment values; invalid levels fail at
startup. Sleep requires a compatible CUDA/VMM runtime and the
`torch_memory_saver` preload hook configured **before** starting the processes.
Check `/sleep_status`: enabling the flag alone does not guarantee `effective=true`.

Level 1 keeps a pinned-host backup of weights. Level 2 discards weights without
writing a backup and reloads the original checkpoint in place on wake, including
supported derived and checkpoint-backed MTP weights. Keep that checkpoint
accessible and unchanged. Level 2 saves host memory but makes wake slower.
Startup checks reject unmerged/multiple LoRA adapters, local multimodal ViT,
EPLB and redundant experts for level 2; runtime LoRA mutations and weight
updates are rejected as well. Embedding engines do not support either level.
Both levels discard KV/prefix-cache contents. The level is fixed at startup,
not switchable per request.

NCCL release is an independent opt-in. It needs a compatible NCCL suspend/resume
runtime and uses pinned-host backups. Verify success on **all ranks**, since an
unsupported runtime may skip this step. CUDA contexts, loaded device code and
some communication memory remain; sleep does not promise zero GPU usage.

## HTTP API

Call the frontend HTTP port, not a backend gRPC port. These examples assume that
your frontend listens on `127.0.0.1:8080` and was started with level 2:

```bash
curl -fsS http://127.0.0.1:8080/sleep_status
curl -fsS -X POST http://127.0.0.1:8080/sleep \
  -H 'Content-Type: application/json' \
  -d '{"level":2,"mode":"wait","timeout_ms":30000}'
curl -fsS http://127.0.0.1:8080/is_sleeping
curl -fsS -X POST http://127.0.0.1:8080/wake_up \
  -H 'Content-Type: application/json' -d '{}'
```

`POST /sleep` accepts `level` (default 1, must match startup), `mode` (`wait` or
`abort`, default `wait`), `timeout_ms` (default 3600000, non-negative), and optional
diagnostic `reason`. `tags` must be absent, null, or empty: partial resource sleep
is not implemented and non-empty tags are rejected. Level 0 is unimplemented;
its error reports the actual backend capabilities. `POST /wake_up` takes an
empty object. Both operations return `{"status":"ok"}` only after convergence.
Do not pass the internal `phase`, `prepare_only`, or `commit_only` fields.

Sleep has two phases. Prepare closes admission and drains requests/transfers;
`wait` preserves in-flight work, while `abort` requests cancellation. A drain
timeout happens before GPU release: the frontend rolls all ranks back and
verifies `RUNNING`, then reports failure. Zero timeout means an immediate check.
Once commit starts releasing resources, cancellation cannot roll it back.
Commit and wake continue toward a terminal state even if the HTTP task is
cancelled. This protection does not survive frontend/backend process death.

`GET /sleep_status` reports the aggregated state, `effective`, `disabled_reason`,
`supported_levels`, `supported_modes`, resource states, admission/transfer counts
and `sleep_epoch`. `GET /is_sleeping` is a smaller capability/state response.
The normal sequence is `RUNNING → DRAINING → SUSPENDING → SLEEPING → WAKING_UP → RUNNING`.
Do not restore traffic until all required roles report `RUNNING` with valid
GPU/KV resources. HTTP success alone is not a post-wake correctness test.

| HTTP status | Meaning for sleep/wake |
| --- | --- |
| 400 | Invalid request, unsupported tags/internal controls, or startup-level mismatch |
| 409 | Precondition failure, lease contention, or failed drain/transition; inspect the body |
| 501 | Disabled/ineffective sleep or unimplemented level 0 |
| 500 | Other backend/transport failure; inspect rank details |

Status-query failures return 500. If a response has `recovery_required=true`, or
ranks disagree after an interrupted operation, keep traffic removed and restart
the complete instance/communication group. Do not clear a stranded instance
lease or force individual ranks back to service. After a successfully verified
drain rollback, a later sleep may be retried. Allow enough HTTP timeout for
checkpoint reload; `timeout_ms` bounds drain, not the entire sleep/wake operation.

## Distributed control addresses

The coordinator must reach the backend **gRPC address of every world rank**,
including non-serving TP/EP/CP ranks, not just DP leaders. It uses distributed
membership or an explicit `RTP_LLM_SLEEP_CONTROL_ADDRESSES` override (comma- or
semicolon-separated addresses, or a JSON string list). All frontend workers
must use the same complete membership and instance TCPStore. The TCPStore is
on the master start port minus one. With the standard worker layout, a rank's
gRPC port is `start_port + local_rank * worker_info_port_num + 1`; check your
actual deployment configuration instead of reusing example ports.

Incomplete coverage fails closed. The
`RTP_LLM_SLEEP_INFER_CONTROL_ADDRESSES` gang-metadata fallback is test-only and
must not replace authoritative production membership. Single-rank direct gRPC
calls are not a substitute for the frontend's all-rank two-phase protocol.
