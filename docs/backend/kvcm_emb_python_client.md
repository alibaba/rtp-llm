# KVCM EMB Python client

This module is a small RTP-facing facade for storing variable-size multimodal
embedding tensors in KVCM's KVMeta object service. It contains only the Python
client and configuration adapter: it does not select an RTP transport, change
the multimodal RPC protocol, or modify the C++ inference path. For the optional
shared ViT embedding cache built on this facade, see
[ViT embedding remote cache](vit_embedding_remote_cache.md).

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

Then use the client directly:

```python
import uuid

import torch

from rtp_llm.multimodal.kvcm import RtpKvMetaObjectClient


key = f"rtp-mm-{uuid.uuid4().hex}"
embedding = torch.arange(24, dtype=torch.float32).reshape(3, 8).contiguous()
output = torch.empty_like(embedding)

with RtpKvMetaObjectClient() as client:
    client.save_one(key, embedding)
    client.load_one(key, output)
    client.remove_one(key)

torch.testing.assert_close(output, embedding)
```

For production use:

- Create one client when the worker starts, reuse it across requests, and
  close it when the worker stops.
- For request-owned transport objects, use a unique key and remove it after
  all consumers finish. Shared ViT cache objects instead reuse the existing
  multimodal input key and are not removed by a consuming worker.
- `load_one()` writes into the supplied tensor; allocate it with the expected
  shape, dtype, device, and byte size before loading.

## Detailed reference

### Prerequisites

- KVCM enables `kvcm.kv_meta.enabled=true`. KVMeta and the fixed-block
  MetaService use the same `kvcm.service.rpc_port`.
- The process has a `kvcm_py_client` wheel that exports
  `kv_cache_manager.client.KvMetaObjectClient`.
- If the fixed-block instance group is `pace_group_m3`, KVCM administrators
  create the separate `kve_pace_group_m3` group before starting this client.
  The client registers its instance automatically, but does not create groups.

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

Static `RECO_SERVER_ADDRESS` values use `hostname:port`, `IPv4:port`, or
`[IPv6]:port`. Ports must be in the range 1-65535. VIPServer IPv4 and IPv6
results are both supported and are frozen into the derived client config.

### Client lifecycle and batch API

The no-argument constructor snapshots the current `RECO_*` values, validates
the complete configuration, resolves VIPServer once, and constructs the
generic KVCM client. KVCM instance registration happens during construction.

`load_one()` fills and returns the supplied tensor; it never allocates a
replacement. Shape, dtype and device come from RTP's receipt layer, which owns
that protocol. The caller must use a fresh object key. If `save_one()` reports
an unknown mutation outcome, retain that key for controlled cleanup after the
write session has converged.

The context manager above only demonstrates the complete lifecycle. A
production RTP worker should construct one client during component startup,
reuse that thread-safe client across requests, and call `close()` during worker
shutdown. Do not register a new client for every embedding.

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
| Load into caller-owned tensors | `load_one(key, tensor)` | `load(keys, tensors)` |
| Remove | `remove_one(key)` | `remove(keys)` |

In a separated deployment, E and P construct their own client using the same
`RECO_*` values. E calls `save`; the control receipt carries the generated key,
shape, dtype, and exact byte size; P allocates a matching tensor and calls
`load`. Object ownership, receipt format, release routing, retry policy, and GC
remain responsibilities of RTP's transport implementation.

### Error handling

The RTP facade intentionally does not catch and flatten KVCM failures. Handle
errors according to the phase and exception type:

| Failure | Exception | Observable state and caller action |
|---|---|---|
| Invalid `RECO_*` values, malformed JSON/address/VIPServer results, or unsafe limits | `RtpKvMetaObjectConfigError` | Construction stops before instance registration or object I/O. Fix the deployment configuration; do not retry in the request path. |
| `environ` is not a mapping, `kv_cache_config` is `None`, or an internal config object has the wrong type | `TypeError` | This is a caller programming error. Fail startup or the owning component initialization. |
| `kv_cache_manager.client` cannot import the high-level KVMeta object classes | `RuntimeError` with an `ImportError` cause | Deploy the KVCM wheel that contains KVMeta object support. Unused RTP paths remain unaffected because the wheel is imported lazily. |
| KVCM native API validation, client configuration, or instance registration fails | Original KVCM exception, such as `ImportError` or `KvMetaObjectClientError(operation="init")` | Client construction fails and no facade is returned. Preserve the exception and fail component startup. |
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
blindly repeat `save` under the same key. Retain the UUID key in a bounded
cleanup queue and reconcile or remove it after the write session converges.
RTP owns any higher-level retry and GC policy.

A load into a destination with the wrong byte size fails with
`ER_SERVICE_SIZE_MISMATCH`; a missing or already released object fails with
`ER_SERVICE_NOT_FOUND`. Load failures are non-mutating, but the caller must
still decide whether retrying is useful. Always attach a request-correlated
`trace_id` when one is available.

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

- `save_one`, `load_one` and `remove_one` are the concise one-embedding API;
  `save`, `load` and `remove` are their batch counterparts.
- `save(keys, tensors)` and `load(keys, tensors)` require equal-length,
  non-empty sequences with unique UTF-8 keys.
- Tensors must be non-empty, contiguous CPU or CUDA tensors. Load destinations
  must have exactly the stored byte size.
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
  RTP should use fresh UUID keys and retain failed keys for controlled cleanup.
- `close()` is idempotent. Prefer the context-manager form.

### Tests

The 52 fast tests do not require the KVCM wheel. They cover configuration and
provider failures, lazy dependency loading, exception identity and structured
progress preservation, no implicit mutation retry/cleanup, and lifecycle
errors in addition to the successful paths:

```bash
python -m unittest -v \
  rtp_llm.multimodal.test.mm_kvcm_emb_config_test \
  rtp_llm.multimodal.test.mm_kvcm_emb_client_test
```

The manual cross-repository test starts a real KVCM service, constructs
independent E/P clients, registers the derived instance, and verifies 67
objects (five dtypes and more than 30 exact byte sizes, including odd sizes)
across the 64-object batch boundary. It also verifies the legacy MetaService
route on the shared port, exact-size mismatch, release, and post-release miss:

```bash
RTP_KVCM_RUN_INTEGRATION=1 \
RTP_KVCM_SOURCE_ROOT=/path/to/KVCacheManager/github-opensource \
python -m unittest -v \
  rtp_llm.multimodal.test.mm_kvcm_emb_client_integration_test
```
