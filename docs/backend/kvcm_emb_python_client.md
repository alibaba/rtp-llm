# KVCM EMB Python client

This module is a small RTP-facing facade for storing variable-size multimodal
embedding tensors in KVCM's KVMeta object service. It contains only the Python
client and configuration adapter: it does not select an RTP transport, change
the multimodal RPC protocol, or modify the C++ inference path.

## Prerequisites

- KVCM enables `kvcm.kv_meta.enabled=true`. KVMeta and the fixed-block
  MetaService use the same `kvcm.service.rpc_port`.
- The process has a `kvcm_py_client` wheel that exports
  `kv_cache_manager.client.KvMetaObjectClient`.
- If the fixed-block instance group is `pace_group_m3`, KVCM administrators
  create the separate `kve_pace_group_m3` group before starting this client.
  The client registers its instance automatically, but does not create groups.

## Existing online configuration

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
group and id.

## Minimal usage

The no-argument constructor snapshots the current `RECO_*` values, validates
the complete configuration, resolves VIPServer once, and constructs the
generic KVCM client. KVCM instance registration happens during construction.

```python
import uuid

import torch

from rtp_llm.multimodal.kvcm import RtpKvMetaObjectClient


key = f"rtp-mm-{uuid.uuid4().hex}"
embedding = torch.arange(24, dtype=torch.float32).reshape(3, 8).contiguous()

with RtpKvMetaObjectClient() as client:
    client.save_one(key, embedding)
    try:
        loaded = client.load_one(key, torch.empty_like(embedding))
    finally:
        # Once save succeeds, release the object even if downstream work fails.
        client.remove_one(key)

torch.testing.assert_close(loaded, embedding)
```

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

## Use an already parsed RTP configuration

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

## Object contract

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
  client registration. A missing or incompatible optional KVCM wheel fails at
  construction with a stable `RuntimeError`.
- Data-plane and metadata failures remain KVCM's structured
  `KvMetaObjectClientError`; the facade does not hide fields such as error code,
  completed batch position or unknown mutation outcome.
- Mutating calls are not retried automatically when the outcome is ambiguous.
  RTP should use fresh UUID keys and retain failed keys for controlled cleanup.
- `close()` is idempotent. Prefer the context-manager form.

## Tests

Fast tests do not require the KVCM wheel:

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
