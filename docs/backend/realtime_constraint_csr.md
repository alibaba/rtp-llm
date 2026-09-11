# Runtime CSR constraint trees (single Master MVP)

The Master builds the tree, not the inference Worker. Submit the complete eligible
SID set to one Master. It builds a compact CSR artifact in the background, retains
current/backup in memory, discovers Workers and pushes the binary over HTTP.
Workers validate/load in the background and atomically activate the snapshot.
Existing requests retain their original snapshot until completion. No DFS is used.

## Configuration

Run **one active Master** for this deployment. Recommended initial settings:

```text
CONSTRAINT_TREE_BUILD_THREADS=10
CONSTRAINT_TREE_RECONCILE_INTERVAL_SECONDS=60
CONSTRAINT_TREE_PUBLISH_CONCURRENCY=2
CONSTRAINT_TREE_PUBLISH_TIMEOUT_SECONDS=120
MAX_IN_MEMORY_SIZE=-1
```

`MAX_IN_MEMORY_SIZE=-1` disables the Master's WebFlux aggregation limit for the
full JSON batch. Size the Master heap, VIP/request limits and client timeout for
the actual batch size; the parsed input and build buffers coexist temporarily.
This is not a promise of unlimited capacity. Only one current artifact and one
backup artifact are retained after successful construction.

On every inference Worker:

```text
CONSTRAINT_TREE_REQUIRED=true
WARM_UP=0
ACT_TYPE=bf16
```

Remove the legacy `TREE_DECODE_CONFIG=prefix_config.json` setting. Start the
Workers, publish the first tree, and verify actual Worker activation before
sending business traffic. A required-tree Worker rejects generation before its
first runtime snapshot is available.

The Master must discover the actual inference model/service name and each
Worker's C++ HTTP port (normally `START_PORT + 5`). This need not be the deployment
or biz name. The local real-model E2E uses model `engine_service`.

## Publish a complete SID set

```bash
curl -X POST "http://MASTER/rtp_llm/constraint_tree/build" \
  -H 'Content-Type: application/json' \
  -d '{
    "version": 1001,
    "model": "engine_service",
    "sids": ["C123C456", "C789C012"]
  }'
```

The SARO contract is a full set of **two-level C-token SIDs**. Preserve the original
symbols, including leading zeros; `C012` is not silently rewritten to `C12`.
Each symbol must exist in the deployed tokenizer's mapping. Examples do not
guarantee those symbols exist in every model. SARO does not supply token IDs or
start/end tokens. The Worker exports an immutable mapping and SHA-256 fingerprint;
the Master probes all target Workers before each build, fetches the full mapping
only on a cache miss/change, and converts in its background task. Mixed Worker
fingerprints prevent the build until model rollout is consistent.

Each publication replaces the complete allowed set, not a delta. Use increasing
versions from one publisher. Same version and content is idempotent (SID ordering
and duplicates do not change content), including after a failed task: repeated POST
does not implicitly retry a failed build. Different content at the same version
returns HTTP 409 with `{"error":"..."}`. Older versions return HTTP 200 with
`state=STALE_VERSION`. Empty SID sets are rejected.

Legacy numeric `rq_token_ids` and underscore SID inputs remain available for
internal callers and variable-length trie tests; they are not the SARO contract.
Production artifacts from either input path are bound to the Worker mapping.

An accepted build is **not yet a completed rollout**. Check:

```bash
curl "http://MASTER/rtp_llm/constraint_tree/status"
curl "http://WORKER_CPP_HTTP/constraint_tree_status"
```

The Master should report `READY`, the expected `active_version`, and matching
published/target Worker counts. Workers should report `ready` and the same
`version`. The Master periodically checks actual state and republishes to
restarted Workers. Old versions and malformed artifacts do not replace a good
active tree. A failed background load retains the old snapshot.

Wire format v2 includes the mapping fingerprint and canonical input SHA-256. The
Worker checks the mapping before accepting/loading and reports both digests with
the active version. Master acknowledgement requires matching version and digests,
not just HTTP 200. A v1 artifact cannot bypass a configured mapping. Upgrade all
Workers to the mapping-aware release before enabling the new Master.

Operator-only retry (not needed for ordinary SARO network retries):

```bash
curl -X POST "http://MASTER/rtp_llm/constraint_tree/retry" \
  -H 'Content-Type: application/json' \
  -d '{"version":1001,"model":"engine_service"}'
```

Only the latest model/version can be retriggered. A failed build repeats mapping
verification and construction using retained input. A built artifact is repushed
without rebuilding. Corrected SID data must use a new version.

Normal inference requests do not carry the tree or a download address. They use
the currently active snapshot. Both fixed `num_beams` and positive
`variable_num_beams` schedules are supported by the existing sampler. The root
must contain at least the first scheduled width of distinct candidates, not the
maximum width. Later expansion selects across the surviving parents; insufficient
valid candidates or non-finite selected scores fail that request closed. No
automatic beam shrinking or duplicate padding is introduced. Finished beams
remain EOS-only while longer siblings complete; other invalid transitions fail
the request closed.

Sparse CSR masks use negative infinity for disallowed tokens. Beam search reuses
the upstream mask-aware TopK optimization (0190dfffb8): masked ties do not require
stable index ordering, while finite candidates retain their original ordering.
Selected non-finite scores are still rejected; the optimization does not make
masked candidates valid. The binary-search CSR mask kernel is unchanged.

## Batched frontend and beam-history transport

The optimized Worker requires no additional environment switches. Upgrade the
Worker image as a unit (Python frontend and C++ engine); the Master/tree format is
unchanged.

- Python clients advertise `accept_batched_output` in the generation RPC. New
  engines stack compatible token/score/optional tensors into the additive
  `batched_output` field. Legacy clients still receive the existing per-output
  tensors; new clients also accept legacy responses. Ragged or partially present
  tensors fall back per field, without padding or dropping results. This preserves
  this branch's legacy RPC schema; it does not promise compatibility with unrelated
  releases that changed the meaning of protobuf field 2.
- Beam detokenization uses the tokenizer's batch API, preserving SID lengths,
  stop handling and custom tokenizer overrides. Non-beam incremental decoding
  retains its existing state. Beam rows are not reused as stable identities after
  parent reordering or width changes.
- `generate_config.aux_info` defaults to `true`. Setting it to `false` omits
  diagnostic metadata, **not requested cumulative/softmax scores or lengths
  required by stop/logits-index processing**. Leave it enabled when collecting
  latency metrics.
- CUDA beam search returns selected tokens and parent IDs without constructing
  and copying a full output history from GPU to CPU. The sampler restores the
  existing CPU history contract. Backends without compact outputs retain the
  full-history fallback. Request history allocation is bounded by input length
  plus `max_new_tokens` and speculative reserve; mixed-length batches copy only
  each request's valid history.
- Access logs use the upstream asynchronous rotating handler with bounded queues,
  drop counters and shutdown draining. Files are `access_r<R>_s<S>.log` and
  `query_access_r<R>_s<S>.log`. Frontends use IDs `[0, frontend_server_count)` and
  backend uses the next ID to avoid cross-process rotation. Update log collectors
  to match these filenames; `aggregate_logs.py` is copied to the log directory for
  manual inspection. Queue overflow may drop logs, not block inference.

These changes do not alter CSR admission, snapshot pinning, version reconciliation
or fail-closed behavior. Validate both full HTTP latency/throughput and engine
latency before increasing traffic; a component microbenchmark is not a capacity
test.

## Recovery and scope

- Master current/backup are in memory, not durable storage. After Master restart,
  the upstream publisher must resubmit the latest full set. Existing Workers keep
  their loaded tree while running; simultaneous Master/Worker restart requires
  resubmission before inference can resume.
- Backup is retained for inspection/recovery, not automatic rollback. To restore
  an older SID set, republish that set using a new, higher version.
- Single-Master operation and the existing binary-search GPU mask
  kernel are intentional MVP limits. No cross-Master election or DFS is included.

## Real-model E2E

`rtp_llm/test/constraint_tree_model_e2e.py` is an opt-in live-service test, not an
offline unit test. It **publishes test trees**, so use dedicated test services.
Run it inside the development container, as a non-root user, after starting a
real model Worker with an empty runtime tree and the Java Master with discovery
pointing to that Worker:

```bash
CSR_E2E_MASTER=http://127.0.0.1:18770 \
CSR_E2E_WORKER=http://127.0.0.1:18765 \
python rtp_llm/test/constraint_tree_model_e2e.py
```

The test IDs match the sample semantic-recall model; other models require their
own valid IDs and EOS. Allow at least 256 tokens of sequence length. Checks cover
required-tree rejection, mixed-length/prefix-overlapping SIDs, fixed beams 1/2/3,
hot publication, backup, stale artifacts, synchronous/asynchronous load failures,
and an in-flight long request spanning a snapshot switch. It prints the final
version/SID set for a separate cold-Worker-restart recovery check.
Variable-beam checks cover growth, shrinking, repeated single-beam steps, candidate
shortage followed by a successful request, and 512-to-3500 expansion with EOS.
The CUDA sampler regression separately compares sparse results against the Torch
reference and checks stable finite ties with small and production-size vocabularies.

For a Java-to-C++ protocol-only test without model weights, use
`ConstraintTreeCrossLanguageE2ETest` with `CONSTRAINT_TREE_CPP_WORKER_BINARY` set
to the built `constraint_tree_test_server`. This does not replace real GPU
inference validation. Follow the repository's test-execution skill for all builds
and container execution.

`ConstraintTreeMappedE2ETest` additionally exercises the Java HTTP receiver and
production mapping-aware publisher against two C++ Worker HTTP processes, including
mapping disagreement, same-content retry/conflict, manual retry, Worker restart and
current/backup. These protocol tests do not measure inference latency.
