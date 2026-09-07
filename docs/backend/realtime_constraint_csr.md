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
```

On every inference Worker:

```text
CONSTRAINT_TREE_REQUIRED=true
WARM_UP=0
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
    "start_token_id": 1699,
    "end_token_id": 151645,
    "sids": ["169967_216546", "169968_215835_215836", "169969"]
  }'
```

SID strings contain model token IDs separated by underscores; length need not
be two. Do not include the start/end token in the SID itself. Token IDs must match
the deployed model's vocabulary. Each publication replaces the complete allowed
set; it is not a delta update. Use increasing versions from one publisher.

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

Normal inference requests do not carry the tree or a download address. They use
the currently active snapshot. Use fixed `num_beams`, with at least that many
distinct root candidates. Variable beam expansion is unsupported. Finished beams
remain EOS-only while longer siblings complete; other invalid transitions fail
the request closed.

## Recovery and scope

- Master current/backup are in memory, not durable storage. After Master restart,
  the upstream publisher must resubmit the latest full set. Existing Workers keep
  their loaded tree while running; simultaneous Master/Worker restart requires
  resubmission before inference can resume.
- Backup is retained for inspection/recovery, not automatic rollback. To restore
  an older SID set, republish that set using a new, higher version.
- Single-Master operation, fixed beams, and the existing binary-search GPU mask
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

For a Java-to-C++ protocol-only test without model weights, use
`ConstraintTreeCrossLanguageE2ETest` with `CONSTRAINT_TREE_CPP_WORKER_BINARY` set
to the built `constraint_tree_test_server`. This does not replace real GPU
inference validation. Follow the repository's test-execution skill for all builds
and container execution.
