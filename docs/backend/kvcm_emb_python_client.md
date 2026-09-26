# KVCM EMB Python client

This module is a small RTP-facing facade for storing variable-size multimodal
embedding tensors in KVCM's KVMeta object service. It contains only the Python
client and configuration adapter: it does not select an RTP transport, change
the multimodal RPC protocol, or modify the C++ inference path.

## Quick start

If RTP's fixed-block KVCM client is already configured, no EMB-specific
environment variables are required. `RtpKvMetaObjectClient()` reuses the
existing `RECO_*` settings, derives a separate `kve_` identity, and registers
the client automatically.

Before starting RTP, make sure that:

- KVCM enables `kvcm.kv_meta.enabled=true` on the existing RPC port.
- `kvcm_py_client` with KVMeta object support is installed.
- KVCM already has the derived instance group. For example,
  `RECO_INSTANCE_GROUP=pace_group_m3` requires `kve_pace_group_m3`.
- The derived group uses a KVMeta exact-object backend (details below).

Create one client when the RTP worker starts. A reusable embedding-cache lookup
is one boolean branch; a miss is not an exception and the Reclaimer owns
capacity eviction:

```python
import torch

from rtp_llm.multimodal.kvcm import RtpKvMetaObjectClient


emb_cache = RtpKvMetaObjectClient()


def get_embedding(stable_semantic_key, expected_shape, encode):
    output = torch.empty(expected_shape, dtype=torch.float32)
    if emb_cache.try_load_one(stable_semantic_key, output):
        return output

    embedding = encode()
    emb_cache.save_one(stable_semantic_key, embedding)
    return embedding


# During worker shutdown: emb_cache.close()
```

The key must digest all inputs that can change the bytes: tenant isolation,
encoder/preprocess versions, media content, tensor role/shape/dtype/layout and
cache schema. Do not call `remove_one()` after a reusable-cache hit; that races
other readers and disables reuse. The production fallback/error pattern is in
the detailed section below.

For production use:

- Create one client when the worker starts, reuse it across requests, and
  close it when the worker stops.
- For a reusable embedding cache, derive a stable immutable key from tenant,
  encoder/preprocess revisions, input-content digest, and tensor schema. Load
  before encoding and do not remove the shared key after each reader; KVCM's
  Reclaimer owns normal capacity eviction.
- For request-scoped E→P handoff, use a fresh unguessable key and remove it
  only after the final consumer has finished loading.
- `load_one()` writes into the supplied tensor; allocate it with the expected
  shape, dtype, device, and byte size before loading. If it raises, discard
  that destination tensor even if it looks complete; it may contain partial or
  concurrently retired data.

## Detailed reference

### Prerequisites

- KVCM enables `kvcm.kv_meta.enabled=true`. KVMeta and the fixed-block
  MetaService use the same `kvcm.service.rpc_port`.
- The process has a `kvcm_py_client` wheel that exports
  `kv_cache_manager.client.KvMetaObjectClient` with
  `KV_META_OBJECT_API_VERSION=2`. Older capability levels fail before instance
  registration.
- If the fixed-block instance group is `pace_group_m3`, KVCM administrators
  create the separate `kve_pace_group_m3` group before starting this client.
  The client registers its instance automatically, but does not create groups.
  That group must use KVCM's `local` or `cached` metadata mode so reads refresh
  LRU heat; KVMeta intentionally rejects direct Redis metadata, which would
  otherwise reclaim capacity without preserving hot embeddings. `cached`
  requires a non-empty valid URI and a local hot-cache layer. `local` is
  process memory only. A production shared TairMempool/PACE cache must use
  `cached` with Redis/async Redis as its persistent layer, plus a persistent
  KVCM Registry; otherwise a KVCM restart loses allocation ownership and GC
  cannot reconstruct it. Reserve `local` for tests or an explicitly ephemeral
  deployment backed by an independently verified namespace TTL/sweeper.
- Every storage candidate in the derived group must implement KVCM's KVMeta
  exact-object lifecycle. Current new-write admission supports NFS and
  TairMempool DRAM/SSD. A fixed-block group using HF3FS, VCNS-HF3FS,
  Mooncake, Dummy, or EventReport cannot simply be reused as an EMB group.

### Existing online configuration

The client reuses the same `RECO_*` variables as RTP's fixed-block KV cache
client. No EMB-specific endpoint, SDK, identity, or timeout variable is needed:

```text
RECO_ENABLE_VIPSERVER=1
RECO_VIPSERVER_DOMAIN=kvcm-na130-m3-bailian-grpc-2.vipserver
RECO_INSTANCE_GROUP=pace_group_m3
RECO_PUT_TIMEOUT_MS=100000
RECO_GET_TIMEOUT_MS=100000
RECO_MODEL_SDK_CONFIG=[{"type":"pace","sdk_log_file_path":"logs/pace_client.log","sdk_log_level":"INFO"}]
TAIR_MEMPOOL_KMONITOR_SINK_ADDRESS=${HIPPO_SLAVE_IP}:4141
KVCM_LOG_LEVEL=INFO
```

For this configuration, the client resolves the existing VIPServer domain and
derives:

```text
instance_group = kve_pace_group_m3
instance_id    = kve_pace_group_m3
```

If `RECO_INSTANCE_ID_SALT=model-a` is present, the instance id is
`kve_model-a`. `RECO_CLIENT_CONFIG`, when non-empty, has the same precedence as
the fixed-block client; the client prepends `kve_` to the selected original
group and id. Its SDK, backend, timeout and unknown top-level fields are
preserved, while `model_deployment` is replaced with the KVMeta object marker;
only `model_deployment.user_data` is carried over from the fixed-block entry.
The derived client requires at least one SDK backend. If the existing SDK
`queue_size` is below KVCM's 64-object native batch limit, only the deep-copied
`kve_` client config raises it to 64; the fixed-block client and its parsed
configuration are not mutated, and no new environment variable is required.

Static `RECO_SERVER_ADDRESS` values use `hostname:port`, `IPv4:port`, or
`[IPv6]:port`. Ports must be in the range 1-65535. VIPServer IPv4 and IPv6
results are both supported and are frozen into the derived client config.

### Bound cache latency independently

The inherited online values `RECO_GET_TIMEOUT_MS=100000` and
`RECO_PUT_TIMEOUT_MS=100000` are 100-second upper bounds. They remain the
default for compatibility, but they are not a safe failure budget for an
optional reusable cache: a failed lookup must fall back before waiting costs
more than recomputing the embedding. No additional environment variables are
needed; pass cache-specific limits when constructing this client:

```python
client = RtpKvMetaObjectClient.from_env(
    call_timeout_ms=500,  # KVMeta control-plane RPC budget
    get_timeout_ms=2_000, # example for a backend with a proven <=2s drain
    put_timeout_ms=3_000, # best-effort publish budget for that backend
)
```

These numbers are examples, not universal defaults. Set them from measured
object sizes, backend tail latency, encoder recompute cost, and the request
SLO. The override is copied only into the derived `kve_` client; it does not
mutate RTP's parsed config or the fixed-block KVCache client. KVCM derives a
write lease that strictly exceeds two effective Put windows plus three
metadata-call windows. The two Put windows cover queue/admission time and a
backend call that starts immediately before the outer deadline; accepted work
is drained before caller-owned tensor memory can be released.
Running a native call in an unbounded Python future is not an equivalent
timeout because cancellation cannot prove that backend I/O stopped touching
the caller-owned buffer. These SDK limits apply to each native batch; a
logical call above KVCM's 64-object boundary can consume multiple sequential
budgets, so RTP must also bound object-set size against its end-to-end SLO.

For TairMempool/PACE, these values must also be strictly greater than the
effective `TAIR_MEMPOOL_SYNC_TIMEOUT_MS` (10 seconds by default in the PACE
revision currently linked by KVCM), and PACE's complete inner timeout hierarchy
must be valid. The native KVMeta client fails initialization otherwise; the
fixed-block KVCache client is unchanged. Therefore the sub-second values often
used for an optional cache are not currently a hard PACE deadline. The existing
100-second `RECO_GET_TIMEOUT_MS`/`RECO_PUT_TIMEOUT_MS` pass the hierarchy check,
but they are still too large for fail-fast inference unless measured recompute
cost and request SLO justify them. A smaller PACE budget requires configuring
and validating the PACE inner timeouts together, not only overriding this
facade.

This hierarchy check protects the local caller-buffer/write-lease budget. It
does not prove that a timeout has drained already submitted remote RDMA/Commit.
KVCM therefore keeps a failed TairMempool write invisible and charged for a
server-side 180-second quarantine after the client commit deadline; a repeated
failed Finish is idempotent and a later successful Finish cannot resurrect it.
No new RTP environment variable is required. If an existing deployment sets
`TAIR_MEMPOOL_QUARANTINE_TTL_MS`, however, the variable-size PACE client requires
it to be positive and at most 180000; a longer client quarantine would outlive
KVCM's address quarantine and initialization fails before PACE I/O. This guard
does not change the fixed-block client. The server quarantine does not change
the derived client lease (for the listed online 100-second Put / 1.5-second
metadata settings, it remains 205 seconds). KVCM invokes its TairMempool
cleanup extension once. The extension chunks large batches at 256 identities
and uses generation-aware exact free/query routes on the storage candidate's
configured PACE MetaService HTTP endpoint. That provider control endpoint is
separate from the shared KVCM gRPC port; the latter is where legacy KVCM meta
and KVMeta object RPCs coexist.
Provider records the logical release on the allocation descriptor, so a lost
response retried against the same Provider process cannot decrement the same
reference twice; only manager-owned physical finalization may resume. A
Provider restart creates a new incarnation and first imports surviving backend
allocations as recovery-owned records, so requests carrying the old
incarnation can never target a newly allocated successor at the same address.
After an acknowledged mutation the adapter retries only the read-only absence
query. If acknowledgement was lost, it may resend the same idempotent exact
free once. A complete, non-partial targeted proof must show every address
absent; malformed, duplicate, partial, or still-present results fail closed.

Before production use, the deployed PACE revision must still pass fault
injection for `timeout -> failed PutFinish -> quarantine -> Free -> same-address
reuse`. The measured worst-case late RDMA/Commit lifetime must be below 180
seconds. Until that is proved, legacy PACE DRAM/direct-RDMA is suitable only for
an isolated functional canary, not a correctness-qualified reusable embedding
cache.

### Client lifecycle and batch API

The no-argument constructor snapshots the current `RECO_*` values, validates
the complete configuration, resolves VIPServer once, and constructs the
generic KVCM client. KVCM instance registration happens during construction.

`load_one()` fills and returns the supplied tensor; it never allocates a
replacement. `try_load_one()` and its batch form `try_load()` provide the
normal reusable-cache lookup: they return `True` on a complete
generation-fenced load and `False` only for KVCM's exact `NOT_FOUND` result.
They do not flatten timeout, size, service, or malformed-client failures.
Shape, dtype and device come from RTP's receipt layer, which owns that
protocol. If a load API raises, or if a `try_load*` API returns `False`, every
supplied destination may already be partially or fully overwritten and is
invalid. A request-handoff caller uses a fresh object key; a reusable
cache caller uses a stable semantic key whose complete inputs always produce
the same bytes. If `save_one()` reports an unknown mutation outcome, retain
that key for controlled reconciliation after the write session has converged.

The context manager above only demonstrates the complete lifecycle. A
production RTP worker should construct one client during component startup,
reuse that thread-safe client across requests, and call `close()` during worker
shutdown. Independent native operations on one client may run concurrently;
bound cache concurrency at worker level so optional I/O cannot starve
inference. Once shutdown starts, new operations are rejected and `close()`
waits for admitted synchronous calls before releasing registered memory. Do
not register a new client for every embedding.

For an RTP request containing several objects, use the corresponding batch
operations. They validate the complete logical call before the first KVCM I/O
and automatically split it at KVCM service limits:

```python
with RtpKvMetaObjectClient() as client:
    client.save(keys, tensors)
    client.load(keys, preallocated_tensors)
    client.remove(keys)
```

| Operation | One object | Batch |
|---|---|---|
| Save | `save_one(key, tensor)` | `save(keys, tensors)` |
| Strict load into caller-owned tensors | `load_one(key, tensor)` | `load(keys, tensors)` |
| Cache lookup (`False` only for `NOT_FOUND`) | `try_load_one(key, tensor)` | `try_load(keys, tensors)` |
| Remove | `remove_one(key)` | `remove(keys)` |

In a separated deployment, E and P construct their own client using the same
`RECO_*` values. E calls `save`; the control receipt carries the generated key,
shape, dtype, and exact byte size; P allocates a matching tensor and calls
`load`. RTP's transport owns the object key/receipt contract, successful-use
release routing, request-level retry, and reconciliation of unknown client
outcomes. KVCM's dedicated KVMeta Reclaimer independently owns cache-capacity
watermarks and LRU retirement. Every reclamation first persists an exact-owner
`DELETING` tombstone, then proves the matching physical generation absent, and
only then removes metadata and releases charged capacity. It must be enabled
and healthy even when RTP normally calls `remove` after consumption.

### Choose the lifecycle before integration

The facade is an exact-object primitive, not an RTP scheduler or
`get_or_compute` implementation. Two valid integrations use different key and
deletion rules:

| Integration | Key | Read/write order | Removal |
|---|---|---|---|
| Reusable embedding cache | Canonical digest of tenant isolation scope + encoder/model revision + preprocessing/tokenizer revision + media-content digest + tensor role/chunk index/shape/dtype/layout + cache schema | Load first; on any load failure recompute; publish the recomputed result best-effort | Never per reader; let the KVCM Reclaimer evict it, or invalidate by changing the versioned namespace/key |
| Request-scoped E→P handoff | Fresh unguessable UUID per logical object | E saves; receipt carries key and tensor metadata; P loads | Only the known final consumer removes it; Reclaimer is the crash/timeout safety net |

Do not combine a stable shared key with per-request removal: one request can
delete an object while another is loading it. KVMeta V1 has a configured read
grace for automatic reclamation but no server-side read lease, and explicit
`remove` has no grace. Conversely, a fresh UUID followed by immediate removal
is a mailbox/transport, not a cache, because later requests cannot hit it.

A reusable-cache integration should keep all cache failure handling outside
the model's correctness path. The output shape and dtype must be derivable
from the request/preprocessing contract so the destination can be allocated
before `load_one()`:

```python
from kv_cache_manager.client import KvMetaObjectClientError


def get_or_compute_embedding(
    client,
    semantic_key,
    empty_output,
    encode,
    request_id,
    on_cache_error,
):
    try:
        if client.try_load_one(semantic_key, empty_output, trace_id=request_id):
            return empty_output
    except KvMetaObjectClientError as error:
        # Timeout or backend failure: recomputation is authoritative. This
        # hook must be non-throwing.
        on_cache_error("load", semantic_key, error)

    # A normal miss and every failed load invalidate empty_output: a
    # post-transfer generation race can report NOT_FOUND after writing bytes.
    embedding = encode()

    try:
        client.save_one(semantic_key, embedding, trace_id=request_id)
    except KvMetaObjectClientError as error:
        # A cache write is best effort. For unknown_outcome, the hook must
        # enqueue bounded reconciliation; it must never blindly retry/remove a
        # stable shared key or raise into inference.
        on_cache_error("save", semantic_key, error)
    return embedding
```

`on_cache_error` is an RTP-owned, non-throwing metrics/reconciliation hook. For
`error.unknown_outcome`, it should retain the operation and key in a bounded
queue until state can be queried safely. The semantic key should be a compact
digest (not raw media or tenant data) and must stay within KVCM's key limit.
Count a business hit only after the complete object set has loaded and its
digest has passed, not when metadata lookup alone succeeds.

A recomputed result does not by itself repair a committed bad entry. KVCM
treats a same-key, same-size `save_one()` as an immutable-value hit, and its
metadata lookup refreshes LRU before the later data-plane read is known to be
good. A durable backend-not-found or checksum mismatch must therefore enter a
bounded per-key repair controller: one designated repairer removes the bad
key, waits for that mutation to converge, and then permits republish. Ordinary
timeouts, overload, and transient I/O failures only fall back for the current
request; they must not make every reader remove a shared key. Until RTP owns
that repair loop, persistent poisoned entries are an explicit limitation of
reusable-cache mode.

RTP still needs to supply bounded per-key singleflight/jitter, whole-object-set
fallback, digest verification, optional worker-local L1, hit/byte-hit and
encoder-skip metrics, and the actual scheduler/transport wiring. Those
application concerns are intentionally not hidden inside this client.

### Error handling

The RTP facade intentionally does not catch and flatten KVCM failures. Handle
errors according to the phase and exception type:

| Failure | Exception | Observable state and caller action |
|---|---|---|
| Invalid `RECO_*` values, malformed JSON/address/VIPServer results, or unsafe limits | `RtpKvMetaObjectConfigError` | Construction stops before instance registration or object I/O. Fail the KVMeta component initialization; an optional-cache integration may disable only this feature and continue inference. Do not retry in the request path. |
| `environ` is not a mapping, `kv_cache_config` is `None`, or an internal config object has the wrong type | `TypeError` | This is a caller programming error. Fail the KVMeta component initialization; do not turn it into per-request retries. |
| `kv_cache_manager.client` cannot import the high-level KVMeta object classes | `RuntimeError` with an `ImportError` cause | Deploy the KVCM wheel that contains KVMeta object support. Unused RTP paths remain unaffected because the wheel is imported lazily. |
| KVCM native API validation, client configuration, or instance registration fails | Original KVCM exception, such as `ImportError` or `KvMetaObjectClientError(operation="init")` | Client construction fails and no facade is returned. Preserve the exception; either fail the explicitly required feature rollout or disable only optional cache use. |
| Empty/mismatched sequences, duplicate/invalid keys, invalid `trace_id`, non-contiguous or unsupported tensors, or an object above `max_object_bytes` | `TypeError` or `ValueError` from the generic client | The complete logical `save`, `load`, or non-empty `remove` call is validated before native I/O. Fix the request; automatic retry cannot make it valid. An empty `remove([])` is a no-op. |
| Native metadata or data-plane failure | `KvMetaObjectClientError` | Inspect the structured fields below. The facade propagates the same exception object without retrying, wrapping, or discarding progress information. |
| Valid non-empty operation after the client is closed | `RuntimeError` | The client cannot be reopened; create a new worker-lifetime client. `close()` itself is idempotent. |

Import `KvMetaObjectClientError` from `kv_cache_manager.client`. Its stable
diagnostic fields are:

- `operation` and `code`: failed operation and native client error code.
- `unknown_outcome`: for `save` or `remove`, the mutation may already have
  committed even though the response was lost or malformed.
- `batch_index`, `batch_count`, `batch_start`, and `batch_size`: location of the
  failed service-sized batch in the logical request.
- `completed_items`: items confirmed complete in other batches. It does not
  imply that any item in the failed batch committed.
- `failed_batches`: normally one; `remove` can report more because it attempts
  every service batch once and aggregates failures.

`save` and `load` stop at the first failed batch. `remove` attempts all batches
once so cleanup can make maximum progress. No mutation is automatically
retried or rolled back. In particular, if `unknown_outcome` is true, do not
blindly repeat `save` under the same key. Retain the operation and object key
in a bounded reconciliation queue; query state after the write session
converges. Only request-handoff ownership may lead to cleanup removal. A
reusable shared key must not be blindly removed because another request can be
using the committed value.
RTP owns that application-lifecycle cleanup and any higher-level retry policy;
KVCM's KVMeta Reclaimer remains the independent capacity and physical-GC
safety net.

A load into a destination with the wrong byte size fails with
`ER_SERVICE_SIZE_MISMATCH`; a missing or already released object fails with
`ER_SERVICE_NOT_FOUND`. Load failures do not mutate KVCM metadata, but they can
partially or fully overwrite caller-owned destination tensors before a data
error or the post-transfer generation check fails. Discard every destination
from the failed logical load and recompute; never publish it to inference.
`try_load_one()`/`try_load()` convert only the exact
`ER_SERVICE_NOT_FOUND` result to `False`; all other failures retain their
original exception. Every destination from a `False` call is still invalid
and must be overwritten or discarded.
The high-level client performs that second metadata `Get` and accepts the load
only when every key remains committed with the same canonical generation URI.
Code that directly splits `KvMetaClient.Get` and `KvMetaTransferClient.Load`
must implement the same fence. Always attach a request-correlated `trace_id`
when one is available.

For example, preserve structured progress instead of converting every failure
to a boolean cache miss:

```python
from kv_cache_manager.client import KvMetaObjectClientError


try:
    client.save_one(key, embedding, trace_id=request_id)
except (TypeError, ValueError):
    # Invalid caller input: report/fail the request; retrying is not useful.
    raise
except KvMetaObjectClientError as error:
    if error.unknown_outcome:
        cleanup_queue.record(key, request_id)
    raise
```

When leaving a context manager normally, a close failure is raised and the
client still transitions to closed state. If another exception is already
unwinding the `with` body, the generic client preserves that original
exception and emits a sanitized cleanup warning instead.

### Use an already parsed RTP configuration

When RTP has already applied environment variables and CLI overrides, pass its
canonical `KVCacheConfig` instead of reading the environment again:

```python
from rtp_llm.multimodal.kvcm import RtpKvMetaObjectClient


client = RtpKvMetaObjectClient.from_kv_cache_config(
    py_env_configs.kv_cache_config
)
```

This entry point does not mutate `py_env_configs.kv_cache_config`.
Tests that have not built RTP's parsed config can inject an isolated environment
without changing process-global state:

```python
client = RtpKvMetaObjectClient.from_env(environ=test_reco_environment)
```

### Object contract

- `save_one`, `load_one`, `try_load_one` and `remove_one` are the concise
  one-embedding API; `save`, `load`, `try_load` and `remove` are the batch
  operations.
- `save(keys, tensors)` and `load(keys, tensors)` require equal-length,
  non-empty sequences with unique UTF-8 keys.
- Tensors must be non-empty, contiguous CPU or CUDA tensors. Load destinations
  must have exactly the stored byte size. Any failed logical load invalidates
  all of its caller-owned destinations, which may already contain transferred
  bytes.
- The generic client splits operations at KVCM's limits of 64 objects and
  4 GiB per service request.
- CUDA producers must synchronize their producing stream before `save`; the
  generic client receives raw pointers and cannot infer framework stream order.
- Invalid RTP/KVCM startup settings raise `RtpKvMetaObjectConfigError` before
  client registration. A missing optional KVCM Python package fails with a
  stable `RuntimeError`; errors raised later by KVCM configuration or native
  registration keep their original type and cause.
- Data-plane and metadata failures remain KVCM's structured
  `KvMetaObjectClientError`; the facade does not hide fields such as error code,
  completed batch position or unknown mutation outcome.
- Mutating calls are not retried automatically when the outcome is ambiguous.
  Handoff mode should retain fresh UUID keys for controlled cleanup; reusable
  cache mode should retain stable keys for bounded reconciliation and must not
  blindly overwrite or remove a key that another request can share.
- `close()` is idempotent, rejects new work once closing begins, and waits for
  already-admitted operations. Prefer a worker-lifetime client; the
  context-manager form is best for bounded examples/tests.

### Tests

The 59 fast tests do not require the KVCM wheel. They cover configuration and
provider failures, lazy dependency loading, exception identity and structured
progress preservation, no implicit mutation retry/cleanup, and lifecycle
errors in addition to the successful paths:

```bash
python -m unittest -v \
  rtp_llm.multimodal.test.mm_kvcm_emb_config_test \
  rtp_llm.multimodal.test.mm_kvcm_emb_client_test
```

The five manual cross-repository tests start real KVCM services, construct
independent E/P clients, register the derived instance, and verify 67 objects
(five dtypes and more than 30 exact byte sizes, including odd sizes) across the
64-object batch boundary. They also verify the legacy MetaService route on the
shared port, exact-size mismatch, release, and post-release miss. A dedicated
small-quota test drives the real Reclaimer through two LRU cycles and checks
its metrics, logical usage release, physical object deletion, surviving-object
readability, and successful writes after capacity is reclaimed:

```bash
RTP_KVCM_RUN_INTEGRATION=1 \
RTP_KVCM_SOURCE_ROOT=/path/to/KVCacheManager/github-opensource \
python -m unittest -v \
  rtp_llm.multimodal.test.mm_kvcm_emb_client_integration_test
```
