# Decode EOS remote KV publication

Status: implementation and validation in progress against
`alibaba/rtp-llm:feature/dsv41-dev-main-202161013`. Historical results from another
revision do not qualify this candidate. See [performance validation](decode_remote_cache_performance.md)
for the measurement protocol.

## Behavior and supported layouts

After successful Decode completion, retain native GPU KV pages and publish a
complete checkpoint through the existing Prefill remote-cache pipeline. A fresh
Prefill process can restore that checkpoint and recompute its unpublished tail.
There is no early CPU KV snapshot or second GPU KV history pool.

The supported route uses synchronous, single-token autoregressive Decode, one
beam, reuse-cache enabled, and a single native PACE backend with dedicated GPU
registrations and GDR. MTP/speculative decoding, asynchronous stream reservation
and mixed SDK backends are rejected when this feature is enabled.

P uses TP equal to its target CP count and DP1. D uses TP1 and an independent
positive DP count. Both roles use the same `prefill_cp_size` and
`prefill_cp_kv_cache_sharded` setting. Sharded and replicated layouts use the same
implementation; a shape accepted by the code is not proof of real-model
qualification on every physical CP/DP topology.

Remote cache identity includes target CP, physical shard count and base block
size. A layout change causes a cold miss. D retains its normal local hot-cache
reuse and receives prompt KV through the existing P-to-D transfer; this feature
does not implement remote-to-D reconstruction from the P-layout wire format.

## Configuration and feature-off behavior

Enable on both P and D:

```sh
export RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH=1
export RTP_LLM_REMOTE_CACHE_ENABLE_GDR_ZERO_COPY=1
```

Use the existing `pace` / read backend 3 for DRAM, or `pace_ssd` / read backend 9
for explicit SSD. The enabled SSD route automatically uses GPU packing and
bounded SDK calls. No additional packing or CP-specific feature switch exists.
State/SWA geometry must match the configured base block size and physical CP
shards; unsupported geometry fails initialization or checkpoint validation.
A nonempty instance salt is required.

| Setting | Default | Meaning |
| --- | --- | --- |
| `RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH` | `0` | Opt in to the supported P/D publication route. |
| `RTP_LLM_REMOTE_CACHE_DECODE_PENDING_MIB` | `min(native_pool_bytes / 8, 4 GiB)` | Optional override, 1-16384 MiB, per D process. |
| `RTP_LLM_REMOTE_CACHE_DECODE_PENDING_JOBS` | `4` | Optional override, 1-64 retained publications per D process. |

When the publication switch is unset or `0`, Factory and ClientWrapper delegate
to the upstream SDK path with its original call granularity and registration
APIs. Decode-only backend/layout/budget validation, admission accounting,
source-page holds, sampler and SSD packing arenas are bypassed. Existing Prefill
remote caching and GDR keep their upstream behavior.

The transport must use matching SDK, Consumer, Provider and Manager versions.
RDMA/GDR and absence of TCP or CPU payload fallback require runtime evidence;
registration success or an environment flag alone does not establish them.
Configure Consumer external-memory registration and its actual local client
count using the transport's existing settings. Do not derive a remote Provider's
client count from P CP or D DP.

CUDA builds pin the `2026_10_25_15_21` client packages in `deps/http.bzl`, with
matching headers and libraries. The older public API alone does not establish
PACE GPU registration support; an old-header/new-library overlay is not a
qualified build. Packaging can rewrite SONAME/RPATH: verify that transformation
reproducibly, then record the actual loaded library hash separately from the raw
RPM and raw library hashes.

## Checkpoints, ownership and shutdown

1. `GenerateStream` records the first successful `GenerateDone` and submits
   before final block remap. `StreamCacheResource` supplies an idempotent
   release-time fallback. Errors, cancellation and uncached warmup do not publish.
2. Computed KV ends at `seqLength() - 1`; the final sampled token has not yet
   been evaluated. Select a checkpoint on the common physical-CP/state/SWA
   boundary, checking all required resident FULL and terminal state/SWA pages.
   Missing pages cause backoff to an earlier complete checkpoint, never relabeling
   final mutable state as an earlier checkpoint.
3. Sharded layouts use the existing CP slot mapper and native state slices.
   Replicated layouts write the same full pages to each target P rank's spec.
4. Admission charges all unique physical source pages for the candidate before
   the StartWrite mask is available. A successful admission performs one EOS
   device fence, then `StorageBackend::prepareWrite()` pins all source/peer pages
   before request ownership can end. There is no per-token device fence.
5. Reuse the existing executor, StartWrite mask, tag batches, actual returned
   URIs and FinishWrite acknowledgement. Normal local hot-cache insertion continues.
   Pins are released only when transfer completion is known.
6. Unknown DMA completion preserves permanent FROZEN references until pool
   destruction and retains its admission charge until backend shutdown. SDK
   shutdown does not unfreeze pool pages. A missing commit acknowledgement cannot
   be called a confirmed publication or trigger an unsafe abort of a possibly
   committed transaction.

On the enabled route, local pre-I/O rejection and an explicit matching SDK
BUFFER_SAFE result are distinct from an unproven SDK error/exception. Unknown
completion quarantines native pages, and a failed RPC or legacy response is not
completion proof. The off route retains its original response handling. After
an unproven write, keep the Manager WRITING lease instead of immediately aborting
and recycling its destination. SDK drain and remote lease lifetime need their
own transport validation; waiting for a timeout does not prove DMA has stopped.

The per-process byte default sums participating unique physical pool capacities
once, takes one eighth and caps at 4 GiB. A 4 GiB pool permits 512 MiB; a 48 GiB
pool permits 4 GiB. CP wire copies do not multiply native capacity. Different jobs
conservatively charge shared pages again. These are limits on delayed page
reclamation, not a preallocated extra pool or a measured optimum. Exhaustion
skips optional publication while preserving the response and local cache.
Admission currently holds the full candidate checkpoint even if StartWrite later
masks most of it; masked pages are not released early.

P supplies its computed-KV exclusive end in the presence-bearing internal
`GenerateRequestPB.prefill_kv_handoff` message, after its successful forward and
transfer. The endpoint excludes the sampled first token. D validates the request
identity and its successful all-rank load span before retaining that assertion
across block remaps. Feature-off or legacy messages leave the boundary unknown;
D input length alone cannot establish source attribution.
If checkpoint selection or state-page backoff leaves no complete checkpoint
beyond that explicit P boundary, skip publication before retaining pages,
charging admission or executing the EOS device fence. An unknown boundary keeps
the compatibility path.

The implementation retains upstream scheduler, CPU-TP broadcast and shutdown
coordination. It adds explicit shutdown of the asynchronous cache-store writer
and detaches surviving Python owners. Acceptance must inspect every native rank
and owned helper exit, rather than the service root's exit code alone.

## SSD transfer bounds

A process/device shares a registered 64 MiB GPU arena, with a 512 MiB
process-wide arena cap and a 64 MiB operation limit. Reads use at most four
objects and 256 KiB of live payload per window. One larger object can use a
window alone within the operation limit. Each concurrent object has a disjoint
GPU span; every sibling finishes before scatter or span reuse. Worker threads
select the originating CUDA device. Callers without a worker initializer retain
the single-object path; writes retain single-object completion ordering.

Gather/scatter is D2D and completes before external RDMA uses or releases the
span. Failure, exception or uncertain completion joins all workers and quarantines
the operation's range until the delegate drains. Independent ranges can continue;
quarantine consumes capacity. There is no CPU KV fallback and no claim of
GPUDirect Storage on the remote SSD host. Actual SDK concurrency and DRAM/SSD URI
compatibility still require validation against the deployed dependency versions.

## Eight monitoring categories

Reuse existing metrics where their meaning matches. Request IDs, keys, URIs and
individual objects belong in logs/timeline, never metric labels.

| Category | Required interpretation and implementation status |
| --- | --- |
| D new logical tokens | The accompanying Manager reports `manager.completed_decode_publish_tokens_qps` for whole-session completed publication; RTP reports `rtp_llm_decode_cache_logical_tokens_qps{stage=restored,used;tier=DEVICE,HOST,DISK,REMOTE}`. See the boundaries below. Final native/model qualification is pending. |
| KV payload bytes | `rtp_llm_remote_completed_payload_bytes_qps{mode=read,write}` counts IOV bytes of successful SDK chunks, including scale/state and actual CP copies. Unknown completion is separate. |
| P restore latency | `rtp_llm_remote_restore_latency_bucket_qps` and `latency_sum_us_qps`, with separate checkpoint, GPU-completion and CPU-observation modes. The complete path's native and all-CP timeline qualification is pending. |
| EOS to publication | `rtp_llm_decode_publish_eos_latency_bucket_qps` and `eos_latency_sum_us_qps`, from first real GenerateDone through confirmed FinishWrite, including delayed planning, fence and queue. |
| Held page budget | `rtp_llm_decode_publish_pending_charge_bytes`, `pending_bytes_limit`; includes frozen pages and conservative cross-job charging. |
| Pending jobs | Same `rtp_llm_decode_publish_` prefix: `pending_jobs`, `queued_jobs`, `running_jobs`, `quarantined_jobs`, `pending_jobs_limit`. Total includes quarantine. |
| Oldest pending age | `rtp_llm_decode_publish_oldest_pending_age_us`, from original admission; retries do not reset it. |
| Operation results | Publication outcomes and stage-aware lookup/read/use outcomes below. Final source-attributed model qualification remains pending. |

An enabled D backend with a reporter samples state once per second, including
idle/stuck work. P and feature-off backends create no publication sampler.
Shutdown joins it before releasing its owner. Queued includes preparation before
executor submission; a late callback cannot move quarantine back to running.

`rtp_llm_decode_publish_outcome_qps{outcome=...}` records one terminal result per
admission: `committed`, `already_present`, `completion_unknown` or `failed`.
`rtp_llm_decode_publish_budget_skip_qps` records the first budget refusal per
logical EOS, separately. A later retry may succeed without repeating that
notification. Do not add budget notifications to terminal results for a request
denominator. Direct low-level callers without an EOS event produce no invented
EOS latency sample.

EOS buckets have fixed `le_us` values from 100 microseconds to 60 seconds and
`+Inf`. Cumulative bucket rates divided by `+Inf` provide approximate P50/P95;
a gauge average does not. Preserve deployment/configuration populations.

`rtp_llm_remote_cache_outcome_qps{stage=lookup,outcome=...}` distinguishes
`failed`, `not_found`, `found_unusable`, `usable_found` per logical remote lookup.
Local-only hits are outside that denominator. Stage `read` reports `failed` or
`read_complete` at the backend all-rank transfer boundary; this is not allocator
adoption, tail replay, or actual request use. SDK-internal retries are not new
logical lookups.

Payload observers report each rank's actual SDK calls, with fixed `tp_rank`,
`dp_rank` and `mode` tags. Feature-on nonroot P backends use a payload-only
reporter; general engine/cache and builtin metrics keep their root reporting
scope. The backend drains and detaches its observer before the process owner
stops the shared metrics factory. Sum per-rank byte rates across the selected
P/D population; do not multiply the root value by CP. Feature-off creates no
payload reporter or metrics factory through this path.

Payload counts exclude D2D and NIC retransmissions. Successful earlier chunks
remain counted when a later chunk fails. An attempted failed/throwing SDK call
increments `rtp_llm_remote_payload_completion_unknown_qps`; its partial bytes
remain unknown. Pre-SDK geometry rejection is not unknown I/O.

For SSD, the observer runs at each actual inner SDK call. Completed objects
remain counted if a later object or D2D scatter fails; the outer wrapper does
not count them again. Arena/geometry/device-initialization rejection before the
inner SDK call produces no SDK payload event. Payload-volume uncertainty is
separate from whether the SDK proves that caller buffers are safe to release.

Restore mode `checkpoint_gpu_ready` starts before cache allocation, including
necessary lookup and page preparation, and ends at successful GPU load
settlement. Emit it only after P adopts a real remote restore.
`read_to_checkpoint_gpu_ready` retains the narrower backend-read subphase.

`fused_context_gpu_ready` requires an explicit producer dependency through the
existing post-model CP hidden collective (or TP1 execution) and completion of
the consumer's existing output stream fence. Its timestamp is taken at that
fence, before CPU worker dispatch. Emit only after the context update succeeds.
This is an observed upper bound on readiness, including sampling/output-copy
wait and any merged new-prompt forward; it is not isolated historical-tail cost
or generic TTFT. No additional CUDA fence or collective is introduced.
Unproven paths emit only `root_context_update_observed`; they must not populate
the GPU-ready distribution. Keep these modes separate. Producer side-stream
joins and all-rank ordering still require final runtime/timeline validation.

Logical D token counts must exclude the already-computed P prefix, CP replicas
and duplicate retries; each real repeated use counts separately. Restoration is
not actual use. Restoration/publication token ratios are not hit rates: one
publication can serve multiple requests. Request hit rates use lookup requests
as their denominator.

The origin-aware Manager owns the publication count: after a marked FinishWrite
claims its allocation exactly once, count the union of D-computed intervals only
when every newly allocated key succeeds and the overall result is `EC_OK`.
A partial success mask, failed CAS or aggregate error contributes zero completed
publication tokens. This does not change existing per-location commit semantics.
If the Manager knows that the whole commit succeeded but D loses its ACK, the
Manager's authoritative count can increase once while D's operation outcome
remains `completion_unknown`. No extra Manager lookup per write is required.

RTP reports `restored` after all required rank transfers and physical-origin
bindings succeed. It reports `used` after P successfully completes its first
context update with the existing all-rank GPU completion proof. A cancelled,
failed, forced-token or CPU-observation-only update cannot count actual use.
Hot DEVICE/HOST/DISK reuse carries its physical source independently of remote
restore timing, so a later independent request can increase `used` without a new
`restored` sample. Overlapping groups and replicas count once, clipped at the
adopted prefix; mixed-tier overlap is attributed to its coldest source.

Sources follow live physical allocation incarnations and per-tag regions,
including successful local tier transfers. Free/reallocation, writable fallback
recomputation and asynchronous overwrite invalidate the affected assertion.
Legacy, inconsistent or stale origins remain usable as generic cache payload
without being labeled Decode-generated. They never acquire attribution from a
canonical-key side map.
Generic nonidentity beam/fork copies lose attribution unless another validated
load establishes it. The same subsequent forward can modify their terminal
state/SWA, so its fence cannot justify restoring the old source assertion.
The supported single-beam, independent-request path retains same-allocation
and successful tier-transfer provenance.

An unmodified Manager ignores FinishWrite.locations, including submitted origin
markers. Accurate D-origin counters require the accompanying origin-aware
Manager contract: validate marked locations against the original instance,
session and allocation version, then replace the complete spec URIs, origin and
SERVING state together in the existing per-location metadata operation. Preserve
legacy unmarked behavior and existing multi-key partial-failure semantics.
The D-side acknowledgement boundary and the Manager's authoritative whole-commit
boundary are distinct as described above.

The bounded `v1` origin format and canonical URI handling have passed
cross-repository CPU differential checks. Private SDK DRAM/SSD payload probes
establish only their recorded revisions and scope. Persistent source metadata
with the real PACE Manager backend, physical-page provenance and end-to-end
token accounting remain delivery gates.

Group dashboards into publication/reuse effect, latency/transfer efficiency,
and backlog/errors. Reuse existing TTFT and overall throughput metrics.
For RTP QPS metrics, sum rates across the intended role/rank population; obtain
window totals by integrating those rates rather than applying a second counter
rate. Payload bytes per second divided by `2^20` gives MiB/s. Keep restore modes
separate when computing bucket quantiles. Sum pending bytes/jobs across D
processes and take the maximum oldest age; divide microseconds by `1e6` for
seconds. The Manager publication QPS export consumes only its request-local
delta, while its registry/Prometheus counter remains cumulative. Do not feed a
shared cumulative value into QPS aggregation or treat a per-query gauge average
as token throughput. Exporter and dashboard qualification on the final real
dependency remain pending.

## Delivery gates

The same committed source, complete wheel, Python models and fingerprinted
SDK/Manager/Consumer/Provider dependencies must pass:

- Feature-off compatibility against upstream; targeted native regression,
  resource admission, retries, quarantine, cancellation and shutdown checks.
- Real Decode-generated KV publication, independent fresh-P recovery, local-hot
  and bypass comparisons for DRAM/SSD by sharded/replicated CP. Compare actual
  continuation tokens through natural EOS and required GPU bytes/state.
- Runtime evidence for RDMA/GDR, absence of KV CPU snapshots and payload D2H/H2D,
  and real SSD IO with cache state documented.
- Eight metric semantics, including proven D provenance, complete P restore
  timing, CP/retry deduplication, actual use and idle backlog age.
- Paired performance with matched effective batch, pool budget and cache state,
  plus D EOS/fence/held-page/queue/commit timeline. Report gains and remaining
  SSD-versus-DRAM gaps without treating historical results as current passes.
- Normal bounded exit of every owned rank/helper, clean resource audit, required
  build/CI and upstream review against the exact Alibaba target branch.

Native mocks, standalone CPU checks and historical four-path runs establish
only their stated scope. They cannot substitute for these final-candidate gates.
