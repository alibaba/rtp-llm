# ViT embedding remote cache

The optional KVCM tier stores one complete multimodal result under the existing
`MultimodalInput.cache_key()` (URL, multimodal type and preprocessing config).
The value contains `(embedding, position_ids, extra_input)`, including all
tensors inside these fields. It does not split these fields into separate keys.

## Read and write path

- Local GPU and CPU caches keep their existing priority and promotion behavior.
- On a local miss, the request that owns the existing in-flight entry tries
  KVCM. Other requests for the same key share that entry.
- A remote hit restores tensor values, shapes, dtypes, list/tuple/dict structure
  and CPU/CUDA placement. CUDA tensors go to the consuming worker's device.
  Preprocessing and ViT forward are skipped; normal local cache publication
  still runs. Feature hashes are computed from the restored embeddings using
  the existing algorithm.
- A remote miss, timeout, malformed object or unavailable service falls back
  to the existing preprocessing and ViT forward path.
- CPU capacity eviction asynchronously uploads a detached snapshot. It does
  not retain a pointer to a reusable local pool slot. GPU eviction still first
  demotes to CPU. Cache clear, failed requests and local removal do not upload.

GreenNet verification and request cancellation remain in the existing engine
path. A remote hit does not imply GreenNet approval. FlexLB routing and
frontend/ViT RPC protocols are unchanged. Remote availability is not advertised
as a local resident embedding by cache-status probes.

## Configuration

The feature is disabled by default. Set these variables on the ViT worker:

| Variable | Default | Meaning |
|---|---:|---|
| `MM_REMOTE_CACHE_ENABLE` | `false` | Enable the KVCM tier. |
| `MM_REMOTE_CACHE_MAX_OBJECT_BYTES` | `268435456` (256 MiB) | Maximum total serialized object size, including all tensors and metadata. |
| `MM_REMOTE_CACHE_MAX_INFLIGHT_BYTES` | `1073741824` (1 GiB) | Per-process admission budget for retained snapshots and transfer staging. |
| `MM_REMOTE_CACHE_MAX_PENDING` | `8` | Per-process maximum admitted reads and writes, including metadata probes. |
| `MM_REMOTE_CACHE_READ_TIMEOUT_MS` | `200` | Per-input wait budget for metadata lookup and queued data loading. Tensor validation and device restoration follow a successful load. |

Each option also has a lowercase CLI form, for example
`--mm_remote_cache_enable=true`. Explicit CLI values override environment
variables. `MM_CACHE_CPU_MAX_BYTES` must be greater than zero when the remote
tier is enabled. An individual object must fit both the object limit and the
in-flight budget: admission reserves three times its serialized size for
source, packed storage and temporary copies. These budgets are additional to
the resident GPU/CPU cache budgets and apply independently in every process;
they are not a cap on total process memory or KVCM storage.

Reuse the prefill KVCM `RECO_*` configuration, including backend SDK settings.
The engine consumes the parsed `KVCacheConfig`, so its CLI overrides also
apply. The adapter derives a separate `kve_` instance/group; see
[client configuration](kvcm_emb_python_client.md#existing-online-configuration).
Install a `kvcm_py_client` wheel with KVMeta object support, enable
`kvcm.kv_meta.enabled=true` on KVCM and provision the derived instance group.
Missing dependencies or invalid configuration fail startup when explicitly
enabled. With the feature disabled, it does not create a client or access KVCM.

Workers sharing identical model weights and output semantics must use the same
derived instance to share results. Isolate different model/weight versions in
different instances, for example through `RECO_INSTANCE_ID_SALT`. The existing
input key alone does not encode a model revision. Metadata endpoint discovery
uses the existing startup-time VIPServer resolution.

## Object representation and lifetime

The current Python SDK accepts one tensor buffer per key. RTP packs the entire
result into one contiguous CPU `uint8` buffer: a versioned header describes
the tensor tree, and aligned payload segments hold the original tensor bytes.
A SHA-256 checksum covers the metadata and payload. There is no pickle or
executable metadata. This compatibility path adds CPU packing/checksum work;
it is not a zero-copy or native multi-IOV implementation.

Reads first call the existing KVMeta `Get` gRPC to learn the exact object size,
then use the SDK to load the one object. RTP carries a small wire-compatible
protobuf subset in `rtp_llm/multimodal/kvcm/lookup.proto`, because the current
Python object client does not expose metadata lookup. There is no HTTP lookup.

Application wait timeout does not imply that the SDK has stopped transferring.
Timed-out transfers retain their buffers and byte reservation until the SDK
call actually returns. Late reads are discarded without CUDA restoration.
Shutdown drains active transfers before closing the client.

Local eviction never calls remote `Remove`: another ViT may still be reading
the shared object. KVMeta V1 does not supply an automatic TTL/LRU or a server
read lease. This integration therefore does not perform online remote GC.
Use an isolated instance and an appropriate remote quota; when writes fail
because storage is full, computation continues using the local caches. Any
external destructive cleanup must wait for all readers/writers to drain.
Same-key concurrent writers rely on KVMeta's existing immutable-object
behavior; ambiguous failed writes are not blindly retried or deleted.

The controller exposes hit/miss, bypass/error and in-flight counters through
`MMRemoteEmbeddingCache.stats()`. Repeated error warnings are rate limited.

## Whole-object client API

For callers that need explicit ownership rather than automatic cache tiers:

```python
from rtp_llm.multimodal.kvcm import RtpKvMetaObjectClient

# One worker-lifetime client, shared by requests.
client = RtpKvMetaObjectClient.from_kv_cache_config(kv_cache_config)
key = mm_input.cache_key()
client.save_object(key, (embedding, position_ids, extra_input))
result = client.load_object(key)  # None if metadata reports a miss.
```

These methods are synchronous and propagate errors. Synchronize GPU producer
streams before calling `save_object`. `load_object(timeout_ms=...)` limits its
metadata query; SDK settings govern the subsequent blocking transfer. The
bounded engine controller supplies the request-side wait timeout described
above. Close the client on worker shutdown, after active users have finished.

## Validation

`mm_kvcm_tensor_object_test` covers mixed tensor types, shape and structure
restoration, malformed payloads, CPU/CUDA placement, the one-key facade API,
and the Get wire contract over loopback gRPC.
`mm_remote_embedding_cache_test` covers cross-worker cache sharing using an
in-memory storage stand-in, pool reuse, timeout lifetime, normal-compute
fallback and GreenNet rejection. These do not replace a real KVCM backend test.

The manual `mm_kvcm_emb_client_integration_test` starts an isolated KVMeta
service with a file backend and now also saves a full result through one
client and restores it through another. In an RTP test environment with the
KVCM wheel installed and the KVCM server built, run:

```bash
RTP_KVCM_RUN_INTEGRATION=1 \
RTP_KVCM_SOURCE_ROOT=/path/to/tair-kvcache \
python -m unittest -v \
  rtp_llm.multimodal.test.mm_kvcm_emb_client_integration_test
```

The server checkout and wheel must both support KVMeta objects. A skipped
manual test is not evidence that the target storage backend or a deployed ViT
service has been validated.
