# Optional KVCM/PACE cache in model inference

The original PR #1436 adds an exact tensor client facade. This additional opt-in
integration calls that facade from the real `MMProcessEngine` used by multimodal
inference. It retains the PR's API v2 check, timeout overrides and `try_load`
contract. On the current baseline it preserves the existing cache-aware
scheduler, GPU/CPU pools, asynchronous serving and CPU-eviction remote tier.
The tensor-object format and read-only metadata lookup are shared with that tier.

## Configuration

```bash
export MM_REMOTE_CACHE_ENABLE=true
export MM_REMOTE_CACHE_NAMESPACE=<immutable-weight-and-processor-code-revision>
export MM_CACHE_GPU_MAX_BYTES=0
export MM_CACHE_CPU_MAX_BYTES=4900000
export MM_REMOTE_CACHE_MAX_OBJECT_BYTES=268435456
export MM_REMOTE_CACHE_MAX_INFLIGHT_BYTES=1073741824
export MM_REMOTE_CACHE_MAX_PENDING=8
export MM_REMOTE_CACHE_READ_TIMEOUT_MS=200
export MM_REMOTE_CACHE_WRITE_TIMEOUT_MS=1000
```

The option is off by default. Existing parsed `RECO_*` / `KVCacheConfig` settings
select KVCM and derive the `kve_` instance as described in
[`kvcm_emb_python_client.md`](kvcm_emb_python_client.md). The namespace is required
when enabled: change it whenever weights or preprocessing code change. Model
type, compute dtype, checkpoint processor/config file hashes, input limits and
all media preprocessing parameters also participate in the key.

This initial path caches one IMAGE work item with an immutable inline
`data:image/...;base64,` URL. The key hashes strictly decoded image bytes, not the
URL spelling, alongside the model and preprocessing identity. Empty or malformed
inline data, mutable HTTP/file URLs, tensor inputs and grouped media use the
existing encoder. A later HTTP/file implementation must establish a content
identity; hashing a mutable URL is insufficient.

The Qwen3-VL and DeepSeek-VL2 C++ GenerateInput adapters receives `torch::empty(1)` as the native
QueryConverter's URL transport placeholder. At `mm_embedding_cpp` only, this
known CPU float32 shape `(1,)` is normalized to an empty tensor for inline IMAGE
inputs, because these models' image preprocessors consume URL bytes exclusively.
No placeholder value is read or hashed. Other models, types, tensor shapes,
dtypes/devices and direct tensor entry points retain strict cache rejection.

## Inference path and lifetime

On a local LRU miss, the engine queries the exact object size, loads one complete
object through the native client, validates its checksum/schema and restores its
tensor devices. A hit bypasses preprocessing and the ViT scheduler. A miss or
cache error runs the existing encoder. Successful fresh encoder results enter the existing local GPU/CPU cache.
Only a completed CPU entry that is actually evicted and detached from its pool
can submit an asynchronous remote write. The object contains the whole
`(embedding, position_ids, extra)` tree, including Qwen3-VL deepstack tensors;
DeepSeek-VL2 uses `(embedding, None)`. Shapes, dtypes, empty tensors and
list/tuple/dict structure are retained without pickle.

The remote tier bounds pending tasks and retained bytes. Its two-worker executor
uploads detached CPU eviction snapshots; it never writes through directly from
an encoder completion or deletes a shared object on a local eviction. Byte
admission accounts for the snapshot, packed buffer and possible contiguous copy.
A read timeout discards the result while retaining its buffer/accounting until
actual IO completes. This adds CPU staging and checksums and is not a GPU-direct
performance claim.

`MM_CACHE_ITEM_NUM` only controls the legacy model-internal cache. The scheduler's
GPU, CPU and hash-key caches have separate byte budgets; setting the item count
to zero does not disable them. The remote eviction policy requires a positive
CPU budget. Use real model requests whose embeddings fit individually but exceed
the total local budget together to trigger the production eviction path.

On read failure the destination is discarded. On any cache operation failure the
client is retired and recreated for a later operation; native transfer timeouts
must not leave subsequent requests using a poisoned wrapper. Cache write errors
do not fail the model request. Stopping an engine closes its client; it does not
delete shared objects. No PACE implementation change is required.

The current KVCM native loader has no post-load generation fence. Its delayed
retirement grace must cover the configured read window before concurrent removal
can be considered safe. This integration does not provide read leases or validate
physical reclamation, concurrent deletion, GPU direct transfer, network-failure
recovery or CUDA graph capture coordination in the full model service.

## Proving remote reuse with a complete model request

Use real multimodal chat requests with immutable inline images and deterministic
decode parameters. Preserve KVCM/PACE between these steps:

1. Load the full model and submit image A. Require successful output,
   `load_miss`, `encoder_forward`, and `result_used` with `source=encoder`.
2. Submit other real images to exceed the local CPU budget. Require
   `cpu_eviction_admitted` for A and then `store_success` only after native
   PutFinish succeeds. These events retain A's original trace and can occur
   during a later request's observation window.
3. Restart RTP to remove local state and submit A again. Require the same
   complete output and tensor/payload checksums, `load_hit`,
   `result_used` with `source=remote`, no ViT execution and a new process identity.
4. Bind the metadata sessions and PACE allocations to the same key and byte size.
   A 200 response, repeated text, Get success or a local cache hit alone is not
   evidence of remote reuse. A previously loaded remote entry can itself be
   evicted again, so a successful no-op save is not a new allocation.

The `MM_EMB_REMOTE_JSON` log records `timestamp_ns`, PID/process start ticks,
trace ID, key, instance, event counts and source. Successful writes/loads also
record serialized SHA256, exact byte size and tensor dtype/shape/device metadata.
Admission records the image byte hash/size; bypass records a reason without URL
or image content. Placeholder normalization is an explicit adapter event.
It never records tensor data. Its trace ID is an engine invocation ID passed to
KVCM, not the frontend HTTP request ID; a serial request window can correlate them.
`store_success` means the native save API succeeded; an already existing exact-size
key can skip a write. Establish a cold miss and a new PACE allocation as well as
the independent consumer's successful load before claiming a new payload write.

Focused unit modules are `mm_kvcm_tensor_object_test`,
`mm_remote_embedding_cache_test` and `mm_remote_embedding_engine_test`; run them
alongside the original facade/config tests on the development machine. Full-model
results must be reported against the precise checkout diff, runtime libraries,
wheel, model manifest and service configuration actually used.

Optional Qwen and DeepSeek tensor diagnostics remain off by default. For delayed
CPU evictions, `eviction_pack_slices` compares the staging bytes and serialized
slice SHA with the original observed GPU tensor metadata. A cache entry retains
only that bounded metadata, never a tensor or a file used as inference input.
An independent reader restores from native cache bytes and does not read the
writer's diagnostic ledger. Unsupported concurrency/model layouts and missing
origins are recorded as unproven.
