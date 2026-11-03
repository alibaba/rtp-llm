# Decode remote-cache performance validation

The delivery target is `alibaba/rtp-llm:feature/dsv41-dev-main-202161013`.
Refresh that branch before the final source freeze. Build, correctness,
monitoring and performance acceptance must identify the same committed source,
complete wheel, Python models and matching transport dependencies. Previous
wheels, native libraries or Python overlays do not qualify a new candidate.

## Feature-off compatibility

With `RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH` unset or `0`, the factory calls
the original SDK initialization APIs directly, including the original external
memory registration API when supplied. It does not inspect SSD backend JSON,
add Decode-specific backend restrictions, or allocate its SSD GPU packing arena.
This does not disable upstream Prefill caching or upstream GDR.

With the main switch set to `1`, the supported SSD path automatically selects
GPU packing from the `pace_ssd` backend. No additional packing enable switch is
introduced. Missing GPU registrations and unsupported mixed-backend layouts are
rejected before allocating a packing arena. CPU payload snapshots are not a
fallback.

P-only performance experiments that exercise this adapter explicitly enable
the main switch on P in both comparison arms. The private driver prevents
measurement-phase publication. Thus the comparison changes only the intended
read policy, rather than inadvertently changing the P transport path by using
different D-publication flags. Preparation writes remain separately reported.

Feature-boundary tests compare disabled factory behavior against the SDK's own
overloads, preserve the upstream non-dedicated-pool fallback when disabled,
and check enabled SSD validation. These checks passed in an intermediate
matching-SDK native run. Repeat the related tests on the final candidate;
native results do not establish feature-off model performance.

## Bounded read candidate

The supported SSD GPU factory supplies the originating device initializer.
Reads use windows of at most four objects and 256KiB of simultaneously live
payload. A single larger state/SWA object remains allowed by itself, up to the
existing 64MiB logical-operation limit. The shared registered GPU arena and
512MiB process accounting ceiling are unchanged; each operation charges its
largest live window. There is no read-window environment variable. The existing
single-object path remains for callers without a worker-device initializer.

Every concurrent object has a disjoint GPU span. All siblings are joined before
scatter, including after an SDK error, exception, device-initialization failure
or worker-launch exception. Uncertain operations retain the whole window until
actual transport quiescence is established; SDK shutdown alone is not proof of
DMA completion. No CPU payload fallback is introduced.

Writes use sequential windows of at most 64 objects and 2MiB. Each URI owns a
disjoint packed GPU range; all synchronous gathers finish before one SDK Save
for the window. The next window reuses that lease only after confirmed
completion. An oversized object stays alone within the existing 64MiB limit.
All window descriptors and trace slices are allocated before GPU copies or SDK
submission. Unknown completion, exceptions and malformed returned URI counts
retain the entire lease. Explicit SDK buffer-safe failures preserve the existing
release contract. Payload reporting sums confirmed successful windows, including
a successful prefix before a later failure. No SDK or Manager changes are needed.

Native tests cover count/concurrency, device selection, disjoint data and
trace slices, byte admission, oversized state objects, device-init failure and
exception drain/quarantine. An intermediate matching-SDK snapshot passed these
native checks. Actual same-client SDK concurrency, error completion and buffer
lifetime still require independent runtime validation. Later telemetry and
safety changes must pass the final source's native and runtime gates.

## Two-turn, four-session measurement protocol

The current wait5-20261029 acceptance uses one A/B comparison per storage medium, not 21
alternating pairs. Each arm has four concurrent independent sessions and two
turns, for 16 requests across A and B. Both arms use the same committed source,
complete wheel, matching SDK, model, physical layout, pool budgets and reader
policy. Only Decode publication differs: disabled on D in A, enabled on D in B.
Keep the Prefill reader adapter configuration identical in both arms and keep
upstream Prefill local/remote caching enabled. Feature-off compatibility for the
whole factory is a separate correctness gate.

Generate each first-turn response with min_new_tokens=max_new_tokens=32768 and
explicit EOS suppression. This is a length-controlled load test, not natural
EOS or answer-quality validation. Four fixed, different token prefixes prevent
cross-session prefix sharing; use the same first-turn input manifest in A and B.
Record the actual computed KV endpoint and aligned committed endpoint rather
than assuming that the last sampled token has computed KV. The prepared workload
has 7680 input tokens, a 128-token follow-up suffix and at most 32 second-turn
output tokens; reserve at least their full combined context (65536 is the
proposed limit, subject to actual model and cache capacity checks).

After all four first-turn streams reach normal EOF, wait five seconds from the
latest EOF using the same client monotonic clock in both arms, then issue the
four second-turn requests together. Record every request's actual dispatch gap;
request preparation can add a scheduling tail beyond the five-second deadline.
The timer does not inspect publication readiness. Continue each session with its complete original input
and actual output token IDs plus its fixed follow-up suffix. Do not retokenize
the generated history. Five seconds is not evidence that publication completed.
Do not add an unreported commit wait only to B: record
EOF-to-commit and EOF-to-next-request separately, including publications still
pending when the follow-up arrives. Preserve the ordinary Prefill prefix cache
and report local reuse, remote reuse and proven coverage of D-computed ranges
separately. A and B must use independent cache groups so A cannot restore B's
published Decode history. Earlier immediate-follow-up cohorts remain results of
the old protocol; do not relabel them as wait5 or pool the two protocols.

The frozen wait5 driver SHA256 is
`31f5f0664d42ce0d940bd8a3699fd0f7e881967f797749b7b3e00e78c8408a6e`.
Request/token/transport logic matches the previous driver except for two
monotonic timestamp fields. It records protocol identity and per-session actual
gaps and accepts only gap_seconds=5. No commit-readiness gate is added.

Client TTFT starts when each streaming request is sent and ends at its first
valid generated token, excluding empty control frames. Keep the real request
concurrency and observed physical batches distinct. Publish all four TTFTs per
arm, their median and range, input/output hashes, complete token IDs, errors and
cache state. Different generated histories make the next-turn input unequal;
mark that comparison inconclusive instead of attributing the difference solely
to the optimization. One comparison is not a P95 or statistical-significance
claim, and no adaptive repetition is used to obtain a favorable result.

Prewarm before timing and collect short diagnostic traces separately from the
formal run. Include the production reporter and actual inner SDK payload
observer consistently. Measure publication and restore stages, SDK time,
thread creation, device/stream synchronization and GPU gather/scatter. For SSD,
record the actual backend, data FD flags, Provider reads and warm-cache state;
page-cache or DRAM hits are not physical SSD bandwidth. Do not flush shared
caches, alter other users' services, compile on the timing host or change the SDK
between the arms. Historical P-only results remain separate from real
D-publication session gains.

## D publication timeline scope

P-only restoration measurements do not establish end-to-end D-publication
performance. The current common baseline fences the device with
`cudaDeviceSynchronize()` after EOS page admission in `planCompletedDecode`.
Measure that fence, whole-history admission, held GPU pages, queue delay and
EOS-to-commit separately from the actual D-only ranges written after
StartWrite masking. Do not replace the fence until an existing producer event
or execution-completion boundary demonstrably orders all FULL/scale/final
state/SWA writes, and coordinate that lifetime change with the functional owner.

## Historical candidates

The previous CP4 sharded, remote-SSD, 32K by actual physical batch4 experiment
used seven alternating pairs per candidate. Vector batching reduced the median
from 2146.15ms to 2093.78ms (2.44%). Bounded concurrency reduced its independently
measured median from 2215.28ms to 2049.33ms (7.49%). Each candidate used the same
wheel and model load in both arms; their samples are not pooled.

The concurrent candidate remains about 5.2 times slower than its 394.36ms
recomputation reference. All seven pairs improved, with two improvements below
3%; it does not meet the historical strong criterion that every pair exceeds
3%, and it does not establish session net gains. Overlap between the two raw
sample ranges is reported, but is not independently a reason to reject a paired
effect.
All four source histories' restored GPU bytes matched; restoration copy
observations were D2D, with no observed KV D2H/H2D. Those observations and the
384 native regressions apply only to the historical candidate artifacts.

Distinguish evidence of a positive paired effect from evidence that the effect
stably exceeds 3%. The historical fixed 21-pair proposal is superseded by the two-turn
protocol above; these historical samples are not pooled with the new candidate. Estimates remain conditional on the recorded
model/Provider run and SSD cache state. P-only read gains do not prove session
net gains: include D publication cost and confirmed actual D-KV reuse.
