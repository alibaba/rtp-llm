# KVCM EMB Python client

`RtpKvMetaObjectClient` adapts RTP's existing `RECO_*` settings to KVCM's
variable-size object API. It is a standalone Python facade. This module does
not connect the inference/multimodal request path, implement the E→P receipt
protocol, or change TairMempool.

## Service contract

The companion KVCM implementation uses the existing TairMempool/PACE DRAM or
SSD allocation and read/write APIs. The EMB instance group must use those
storage candidates. The former NFS/file-only integration harness is not
compatible with this implementation.

- Enable `kvcm.kv_meta.enabled=true` on KVCM's existing RPC port.
- Install a matching `kvcm_py_client` wheel exporting
  `KV_META_OBJECT_API_VERSION=2` in Python and native code.
- Create the derived instance group before constructing the client. For
  `RECO_INSTANCE_GROUP=pace_group_m3`, create `kve_pace_group_m3`.
- Use the existing PACE SDK configuration and provision the group quota.
- For recovery testing, persist both the KVCM registry and metadata. An
  ephemeral registry/metadata deployment cannot establish restart recovery.

API version 2 establishes a compatible interface; it is not evidence that
allocation reclamation, timeout behavior or restart recovery have been tested
on a deployed PACE revision. Record both source revisions and the actual
server/wheel artifacts used in the test.

## Quick start

```python
import torch
from rtp_llm.multimodal.kvcm import RtpKvMetaObjectClient

source = torch.arange(24, dtype=torch.float32).reshape(3, 8)
destination = torch.empty_like(source)

with RtpKvMetaObjectClient() as client:
    key = "request-unique-object-key"
    client.save_one(key, source)
    client.load_one(key, destination)
    assert torch.equal(source, destination)
    client.remove_one(key)  # only after the final reader has completed
```

Create one client when the worker starts and reuse it. The generic client
accepts concurrent callers, but its current native data-plane operations are
serialized per client. `close()` rejects new operations and waits for already
admitted native calls. A closed client cannot be reopened.

| Operation | One object | Batch |
|---|---|---|
| Save immutable bytes | `save_one(key, tensor)` | `save(keys, tensors)` |
| Load into caller-owned memory | `load_one(key, output)` | `load(keys, outputs)` |
| Load, returning `False` for exact `NOT_FOUND` | `try_load_one(key, output)` | `try_load(keys, outputs)` |
| Remove | `remove_one(key)` | `remove(keys)` |

`load_one` returns the same destination tensor. Batch methods return no tensor
list. `try_load` returns one boolean for the entire object set, not a hit mask.
The generic Python client validates the complete logical input before native
I/O and splits requests at 64 objects and 4 GiB per service batch. An object
must contain at least one byte and may be at most 1 GiB, or the smaller
configured `max_object_bytes`.

## Configuration

No EMB-specific production environment variables are required. The no-argument
constructor snapshots RTP's existing settings and derives a separate identity:

```text
RECO_ENABLE_VIPSERVER=1
RECO_VIPSERVER_DOMAIN=<existing KVCM domain>
RECO_INSTANCE_GROUP=pace_group_m3
RECO_MODEL_SDK_CONFIG=[{"type":"pace","sdk_log_file_path":"logs/pace_client.log","sdk_log_level":"INFO"}]
RECO_GET_TIMEOUT_MS=12000
RECO_PUT_TIMEOUT_MS=12000
```

Both the group and instance ID receive the `kve_` prefix. Without an instance
salt the instance ID is derived from the original group. With
`RECO_INSTANCE_ID_SALT=model-a`, it becomes `kve_model-a`. A non-empty
`RECO_CLIENT_CONFIG` takes precedence over split settings; a map with several
entries requires an empty-key primary entry.

The adapter deep-copies the selected configuration. It preserves SDK/backend
settings, changes the schema to `block_size=1` and
`location_spec_infos={"value":1}`, raises only the derived SDK queue floor to
64, and replaces `model_deployment` with the object marker while retaining its
`user_data`. The fixed-block client's configuration is not mutated. VIPServer
resolution happens once and is frozen into the derived endpoint list.

Use `from_kv_cache_config(py_env_configs.kv_cache_config)` when RTP has already
parsed CLI overrides. Use `from_env(environ=...)` for an explicit environment
mapping. Both methods accept `max_object_bytes`, `call_timeout_ms`,
`put_timeout_ms`, and `get_timeout_ms` overrides.

The adapter derives a write lease strictly above two SDK Put budgets plus
three metadata call budgets, with a 30-second minimum and an 1800-second
maximum. SDK timeout values are not a hard end-to-end deadline: a timed-out
operation can wait for accepted backend work to drain, and calls exceeding
one native batch can consume several sequential budgets. Match timeout
configuration to the actual PACE build and measure elapsed time under faults.

In the current KVCM client, `ER_SDK_TIMEOUT` destroys the transfer wrapper.
That client cannot perform further data-plane I/O. Stop admitting work to it,
close it, and construct a new client before a controlled retry. Recreating a
client does not resolve an earlier write whose outcome is unknown.

## Bytes, receipts and object lifetime

KVCM stores opaque bytes. It checks exact byte count, not tensor shape, dtype,
device or semantic identity. Two tensors with the same byte count but different
shape or dtype can be loaded without a schema error. RTP must validate the
receipt and allocate the correct contiguous output before loading. For GPU
buffers, the caller must also establish the required device/stream ordering;
the CPU integration suite below does not qualify CUDA/MUSA behavior.

For E→P handoff, E saves under an unguessable request-owned key. Its receipt
must carry the key, exact byte count, tensor shape/dtype/layout, and any schema
version needed by P. P loads and acknowledges completion before the final
owner removes the key. Separate producer and consumer clients must derive the
same KVCM instance ID and group.

For a reusable cache, derive a compact semantic digest from tenant isolation,
encoder/model and preprocessing revisions, media content, tensor role,
shape/dtype/layout and cache schema. Same-key/same-size `save` is an immutable
hit: it does not compare or overwrite existing bytes. A caller must never use
that behavior to repair a corrupted value silently. Normal readers must not
remove a shared key; invalidation requires ownership and coordinated readers.

The current native load obtains metadata and then reads the returned storage
URI. It does not hold a read lease or perform a post-transfer generation check.
A concurrent Remove/Reclaimer and a sufficiently delayed read can therefore
outlive the allocation's retirement grace. Successful loading alone does not
prove safety under that race. Production use with concurrent reclamation
requires separate verification of the reader lifetime and deletion contract.

If load raises, or `try_load` returns `False`, discard all supplied destinations.
An earlier batch may already have overwritten its outputs before a later
batch fails. Recompute into fresh or completely overwritten tensors. Only
exact `ER_SERVICE_NOT_FOUND` becomes `False`; timeout, size mismatch and other
failures remain exceptions.

## Errors and controlled retries

Configuration failures occur before client registration. `RtpKvMetaObjectConfigError`
indicates invalid settings; unsupported/missing wheel capabilities raise
`RuntimeError`. Invalid keys, duplicate keys, noncontiguous tensors, sequence
length mismatches and object-size violations raise `TypeError` or `ValueError`.
An empty `remove([])` is a no-op.

Import `KvMetaObjectClientError` from `kv_cache_manager.client`. The facade
preserves the original error object with these fields:

- `operation` and `code`: the failing operation and native result.
- `unknown_outcome`: whether a mutation may have happened despite an uncertain
  response. False does not undo an already completed earlier batch.
- `batch_index`, `batch_count`, `batch_start`, `batch_size`: the failed service
  batch in the logical call.
- `completed_items`: the confirmed prefix before that failed batch.

There is no `failed_batches` field in the companion client. `save`, `load` and
`remove` all stop at the first failed batch; later batches are not attempted.
For cleanup, retain all request-owned keys and make separately tracked removal
attempts so one failure cannot silently strand the remaining keys. A logical
Remove acknowledgment is not proof that PACE physically freed the allocation.

Do not retry an unknown write blindly or immediately remove a shared key. Retain
its operation identity and keys until the write session has converged, then
reconcile against the authoritative state. Known size/input failures may be
retried after correcting the input; successfully saved prefixes are immutable
and can be checked before retrying a logical multi-batch operation.

The facade supplies no application retry queue, receipt transport, singleflight,
fallback encoder, repair controller or deployment recovery policy. Those
belong to the RTP component integrating this API.

## Verification on a development machine

The live suite connects to a separately prepared, isolated KVCM/PACE deployment.
It does not launch a file backend, restart services, change PACE, or touch the
inference path. Build and install KVCM's server and wheel from the same tested
revision, then create `kve_emb_it` with PACE storage and sufficient quota.

```bash
RTP_KVCM_RUN_INTEGRATION=1 \
RTP_KVCM_ENDPOINT=127.0.0.1:19001 \
RTP_KVCM_INSTANCE_GROUP=emb_it \
RTP_KVCM_SDK_CONFIG='[{"type":"pace"}]' \
RTP_KVCM_EVIDENCE_DIR=/tmp/rtp-kvcm-evidence \
python -m unittest -v \
  rtp_llm.multimodal.test.mm_kvcm_emb_client_integration_test
```

Use the real SDK configuration for the deployed PACE build. The test-only
`RTP_KVCM_CLIENT_CONFIG` accepts a complete existing `RECO_CLIENT_CONFIG` map
and takes precedence over endpoint/group/SDK variables. Optional test variables
are `RTP_KVCM_INSTANCE_ID_SALT`, `RTP_KVCM_CALL_TIMEOUT_MS`,
`RTP_KVCM_PUT_TIMEOUT_MS`, `RTP_KVCM_GET_TIMEOUT_MS`, and
`RTP_KVCM_WORKER_TIMEOUT_SECONDS` (default 300). These do not add production
EMB configuration knobs.

The suite covers:

- Independent producer and consumer processes, 67 objects, five CPU dtypes,
  more than 30 exact byte sizes, odd sizes, byte equality and SHA-256 evidence.
- Mixed existing/new writes and a miss after a successfully loaded 64-item
  prefix; every failed logical load's destinations are discarded.
- Exact-size failures, preservation of the old bytes, immutable same-size hits
  and retries with corrected input.
- A deterministic second-batch save failure with a verified 64-item committed
  prefix, followed by a controlled retry.
- Logical removal, repeated removal, request-key reuse, and close/new-client
  continuity of committed bytes.

These tests do not establish backend timeout recovery, response-loss handling,
concurrent read/free safety, physical allocation reclamation, quota reuse, GPU
ordering or server/Provider restart recovery. Test those separately with fault
injection and PACE allocation/byte evidence. Do not report a skipped suite or
successful client initialization as a completed end-to-end test.
