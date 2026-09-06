# KVCM storage for separated multimodal embeddings

RTP-LLM can use the isolated KVCM KVMeta exact-key object API as the data
plane between a separated ViT worker and the LLM process. This path is
independent of `cpp/cache/connector/remote_connector`; enabling it does not
change KV-cache storage or the default multimodal transport.

## Enablement

The implementation is deliberately opt-in at both build time and runtime:

```text
build flag: --define=use_kvcm_emb_storage=true
runtime:    MM_TRANSPORT_MODE=kvcm
```

The build flag requires a KVCM client RPM that exports
`KvMetaObjectClient` and the `kv_meta_{client,object_client,transfer_client}.h`
headers. Until that RPM is published and pinned in `deps/http.bzl`, normal RTP
builds continue to select the no-dependency stub.

Both ViT and LLM processes must use the same configuration:

```text
MM_KVCM_ADDRESSES=host-a:port,host-b:port
MM_KVCM_INSTANCE_ID=<instance>
MM_KVCM_INSTANCE_GROUP=<group>
MM_KVCM_USER_DATA=<registration payload, if required>
MM_KVCM_TRANSFER_CLIENT_CONFIG=<KVCM client JSON>
```

Optional safety limits are `MM_KVCM_CALL_TIMEOUT_MS`,
`MM_KVCM_WRITE_TIMEOUT_SECONDS`, `MM_KVCM_OBJECT_GC_TIMEOUT_MS`,
`MM_KVCM_MAX_OBJECT_BYTES`, and `MM_KVCM_MAX_RECEIPT_BYTES`. The existing
`MM_RDMA_RELEASE_TIMEOUT_MS` controls the shared release RPC deadline for all
external multimodal transports.

The transfer JSON must use the isolated marker `block_size=1` and
`location_spec_infos={"value": 1}` with the same instance/group identity.
`MM_KVCM_WRITE_TIMEOUT_SECONDS * 1000` must be strictly greater than its
`put_timeout_ms + 3 * MM_KVCM_CALL_TIMEOUT_MS`; the three metadata windows
cover the `PutStart` hand-off, the masked-hit compatibility `Get`, and
`PutFinish`. KVCM rejects an impossible budget before registering the instance.
Its `sdk_config.queue_size` must be at least 64 so one service-sized object batch
can be admitted without blocking on the exact-object worker queue; this
requirement does not apply to the existing fixed-block `TransferClient` path.

## Data and failure semantics

- Each contiguous tensor chunk is one KVMeta object with its exact byte size.
  Different objects in the same result may have different sizes.
- Before handing CUDA pointers to KVCM, the ViT backend synchronizes producer
  work on each participating device because KVCM SDKs do not inherit PyTorch
  stream dependencies.
- A tensor larger than `MM_KVCM_MAX_OBJECT_BYTES` is split only along its
  first dimension. A single row larger than the limit is rejected before any
  metadata or storage operation.
- Before storage, the writer verifies the same manifest constraints enforced
  by the LLM reader: 1--16 positive dimensions, supported dtype, per-image
  position-row alignment, and one non-empty flat extra-input tensor per image.
- The ViT writer generates a fresh UUID key for every object. The LLM validates
  all keys, roles, shapes, dtypes, byte counts, limits, and split metadata
  before allocating tensors or invoking KVCM.
- KVCM calls are chunked to at most 64 keys and 4 GiB per batch, matching the
  KVMeta service admission limits. One receipt also accepts at most 16384
  logical values and 1024 physical objects. The physical-object bound matches
  the shared control client's pending-release capacity, so one valid receipt
  fits an otherwise empty async-release queue; both limits also bound
  reconstruction metadata and tensor-view materialization. The LLM stops admitting further
  load batches when the end-to-end multimodal request budget expires. An SDK
  operation already using a caller-owned tensor is still drained before its
  storage returns, so a non-cancellable backend cannot write freed memory. A
  partial multi-batch write is
  rolled back; this is safe because the writer uses fresh UUID keys for every
  receipt. If a rollback removal is temporarily unavailable, the ViT worker
  keeps the keys in its cleanup queue and retries instead of leaving permanent
  KVMeta metadata. No inline or RDMA fallback is attempted when `kvcm` is
  selected.
- Successful reads release the keys asynchronously through the existing
  control RPC. The ViT worker accepts releases only for keys it owns and also
  runs a deadline-triggered cleanup loop for lost receipts/releases. Failed
  removals are retried until they succeed or the worker shuts down. Shutdown
  first closes admission and drains in-flight transfer/release/GC operations,
  then performs one final best-effort cleanup of all remaining keys.
- `MM_KVCM_OBJECT_GC_TIMEOUT_MS` starts when the ViT result is committed. It
  must cover receipt delivery, LLM allocation, and every 64-key/4-GiB KVCM
  load batch. Its default is 180 seconds, one minute beyond the default
  multimodal request budget. Increase it together with any larger request
  timeout; configuring it too low can reclaim an object while the LLM is still
  reading it.

KVMeta V1 removes metadata on release. A storage backend may have its own
physical reclamation behavior; in particular, the current NFS backend does not
unlink object files, so deployments using NFS still need backend-level orphan
cleanup.

Use an instance group reserved exclusively for KVMeta objects; admitting a
regular fixed-block KVCM instance into that group violates the isolated schema
assumption and causes the object service to fail closed. The current RTP
adapter does not provide a registered memory span for Mooncake, so its KVCM
transfer JSON must select a backend that does not require that registration.
Finally, the ViT retry queue is process-local: a worker crash after commit can
leave objects without a release. Production deployments need an operational
orphan policy (for example namespace rotation plus `Trim`) in addition to the
normal receipt/release path.

## Tests

The default `mm_output_transport_test` uses an in-memory writer and covers
receipt validation, exact-size chunking, rollback, release, GC, shutdown races,
and input bounds without starting KVCM. It remains on the ordinary RTP test
path.

The live cross-repository contract test is a separate Bazel target tagged
`manual`; wildcard/default test runs do not select it. It starts a KVCM service
with an isolated local metadata database and temporary file backend, sends
real CPU tensors through the production RTP Python output backend and v6d
native object adapter, reconstructs every role from the RTP receipt, and checks
explicit release plus deadline GC:

```bash
export RTP_KVCM_RUN_INTEGRATION=1
export RTP_KVCM_SOURCE_ROOT=/path/to/KVCacheManager/github-opensource
export V6D_SOURCE_ROOT=/path/to/vineyard2
export PYTHONPATH="${RTP_KVCM_SOURCE_ROOT}/bazel-bin:${V6D_SOURCE_ROOT}/src:${PYTHONPATH}"

bazel test //rtp_llm/multimodal/test:mm_kvcm_cross_repo_integration_test \
  --test_env=RTP_KVCM_RUN_INTEGRATION \
  --test_env=RTP_KVCM_SOURCE_ROOT \
  --test_env=V6D_SOURCE_ROOT \
  --test_env=PYTHONPATH
```

This contract test deliberately substitutes v6d's production adapter for the
RTP C++ writer. The compiled `MMKvcmWriter`/`MMKvcmReader` batching and manifest
boundary remains covered by `MMKvcmTransportTest`; an RTP full-process test
still requires the KVCM-enabled RTP build image and its normal Bazel execution
environment.
