# Global Batched Prefill Live E2E

## Result

**PASS for global B=8 planning, cache-affinity placement, and eight-request
Prefill engine steps.**

The decisive load experiment released 48 concurrent `pre-chat-test` requests
through Chat and FlexLB. Each had 2,001 client-visible prompt tokens, all
started in a one-millisecond range, and all returned HTTP 200. FlexLB formed
six adjacent global B=8 planning events and assigned exactly eight requests to
each of the six live logical Prefill engines. All 48 selected candidates were
modeled and had zero actual cache hit.

A separate B=8 mixed workload then released four warm-prefix and four cold
requests together. The four hot requests predicted and actually hit 2,176
tokens on the two cache-owning logical engines; all four cold requests had zero
hit and were placed on the remaining four engines. This is route-correlated
cache-affinity evidence, not merely the `CACHE_AFFINITY` reason string.

The full 48-request Chat PD trace has an observed Prefill `bs=8` on every
logical engine. With the 1,997 tokens seen by the engine per request, that is
15,976 prompt tokens in an eight-request engine step, i.e. approximately a
16k-token fill.

All times below are `Asia/Shanghai` (`+08:00`) on 2026-09-14.

## Scope and route oracle

| Component | Deployment / identity |
| --- | --- |
| FlexLB | `flexlb-test-wlcb`, debug instance `ds-bc30de2e-1-8b9ccc69c-2tg25` |
| Prefill | `vllm-test-2-0`: `172.27.112.157`, `172.27.112.158`, and `172.27.65.218` |
| Decode | `vllm-test-2-decode-0`, Pod `ds-5446d4c4-1-5db899cbd6-csm2d`, `172.27.133.112` |
| Chat | `chat-test-da45` |
| Entry | DashScope-compatible `pre-chat-test` endpoint |

For request-placement claims, the oracle is the joined request ID across client
HTTP result, FlexLB `pv.log`, Chat PD logs, and the selected Prefill/Decode
engine logs. Engine utilization is reported separately and is never used as
proof that FlexLB selected an endpoint.

## Deployed artifact and configuration

The debug deployment originally contained an old FlexLB archive. The local
package was built and uploaded:

```text
./mvnw package -pl flexlb-api -am -DskipTests
```

The six-module packaging reactor succeeded. The uploaded
`flexlb-api/target/ai-whale.tgz` has SHA-256
`762c613b5ab0ee2be418c8fd3373a294713a79ce5a907ab42f8855162b72d002` and was
verified to contain `CostBasedBatchedPrefillStrategy.class`, its
`BatchCandidates` class, `PrefillStrategy.class`, and the global-window
coordinator classes.

The local open-source archive lacks the internal DashScope discovery provider.
To preserve the existing test deployment topology, the original package's
`dashscope-discovery`, `vipserver`, and `vipserver-client` jars were copied into
the local package's classpath; the local application classes remained in use.
The original archive is recoverably backed up in the debug container at:

```text
/home/admin/ai-whale/target/ai-whale.pre-e2e-20260914-0949.tgz
```

Its SHA-256 is
`42e7d61866211bf2c259f2f559125fc0b54b9ed8cf22e09bb12d8fdd4ae908c7`.
Manual start succeeded and both `/health` and `/hook/process_ok` returned 200.
The runtime continues to run in debug mode; no restore was performed.

Startup confirmed the `UNICONFIG` source and effective queue settings:

```text
scheduler=QUEUE, ordering=PRIORITY, decision=SINGLE,
dispatcher=NON_BATCH, prefillCandidateChoice=BEST_ONLY
```

Here `decision=SINGLE` is the endpoint-local delivery mode. It is not the
global planning-window mode. The later B=8 decision records are the live proof
that the global fixed-window batch planner was active after the user set
`maxRequests=8`.

## Why the `BATCH_*` record is decisive

`CostBasedBatchedPrefillStrategy.selectBatch` has a B=1 fast path: it delegates
to `selectOne`, which retains the ordinary `BEST_ONLY/...` reason. Only the
multi-request branch constructs `PlanningRequest`s and writes
`BATCH_BEST_ONLY` or `BATCH_BEST_ONLY/CACHE_AFFINITY`.

The control warm-up request
`268609cb-138f-90ba-ba4f-a60f60357f4a` verifies that behavior in the deployed
service: it returned HTTP 200, selected `172.27.112.157:8080@0`, and recorded
`BEST_ONLY/NO_CACHE_LEAD`.

Each B>1 PV record still has `decisionGroup.committedSize=1` and
`reason=single_request`. That is expected. `WorkerBatcher` owns the later
endpoint-local `SINGLE` delivery group after global planning; it is not the
cardinality of `CostBasedBatchedPrefillStrategy.selectBatch`.

## Earlier cache-affinity B=2 check

A warm-prefix request was followed by one hot and one cold concurrent request.
Both returned HTTP 200 at the public API and received the same FlexLB planning
timestamp, `1789351951309`.

| Case | Request ID | FlexLB placement | Cache / route evidence |
| --- | --- | --- | --- |
| Hot prefix | `337888ff-1973-9c46-b789-3ee8b0e16dd7` | `172.27.112.157:8080@0` | 1,088 effective/routing match tokens; Chat and the selected Prefill Pod `ds-38bd08d3-1-7dbd99d8b4-bl5pp` confirmed receipt; Decode completed in 216 ms. |
| Cold prefix | `fcb8c6c1-b3ad-9f12-afda-8f0f2540068b` | `172.27.112.158:8080@0` | zero cache-match tokens; Chat and Prefill Pod `ds-38bd08d3-1-7dbd99d8b4-r7swm` confirmed receipt; Decode completed in 278 ms. |

Both records contain `BATCH_BEST_ONLY/CACHE_AFFINITY`. The hot request stayed
on the cache-owning endpoint while the cold request was placed on the other
Prefill IP. This is route and cache-affinity evidence, not a latency benchmark.

## Decision-record load check

Before changing `maxRequests`, a 24-request burst of 2,440-token prompts formed
12 adjacent two-request planning events, each with a distinct pair timestamp.
That was consistent with the former `maxRequests=2` configuration.

After the user set `maxRequests=8`, the formal run used a Python thread barrier
to release exactly eight requests together. The client start range was one
millisecond (`10:29:02.084`), and every request returned HTTP 200 with 2,001
prompt tokens.

| Request IDs in one B=8 plan | `decisionTimeMs` | Reason | Selected logical endpoints |
| --- | ---: | --- | --- |
| `903a718f-b0af-9906-a7ef-5ab135d78764`, `cfd4f1d0-7171-9430-88dc-5e5da918ea89`, `089a35ec-10ec-9bd5-9676-c007b7b5673a`, `6499f45b-2aed-98ff-8893-93c70dabf5ea`, `1c89568d-ee60-9b7f-a921-30b16ce7ba45`, `66c00d42-688a-9fbf-a93a-ee893c419f2f`, `2d56bcbe-c8fe-9431-bd4d-d9bef800e727`, `da25d03b-4bd7-9be3-90a0-ebaad5c31dfb` | `1789352942486` | all `BATCH_BEST_ONLY/CACHE_AFFINITY` | `.158@0` ×2, `.158@1` ×2, `.157@0` ×1, `.157@1` ×1, `.65.218@0` ×1, `.65.218@1` ×1 |

Every selected candidate had a 2,001 ms base projected TTFT and zero effective
cache hit. The planner therefore spread the eight equally sized cold requests
over all six healthy logical endpoints, assigning the two remaining requests to
the two `172.27.112.158` endpoints. This is the expected cost-based behavior;
the decision record is exactly one B=8 global plan even though its local delivery
groups remain single-request.

## 48-way cold-load global planning and engine fill

The primary experiment used the same barrier pattern at `10:43:25.314` with 48
unique cold prompts (2,001 input tokens and 8 output tokens each). All 48
returned HTTP 200 in 969--1,512 ms. The external OpenAI-compatible response ID
has a `chatcmpl-` prefix; Chat/FlexLB use the same UUID without that prefix for
route correlation.

| Check | Observed result |
| --- | --- |
| B>1 decision branch | 48/48 records use `BATCH_BEST_ONLY/CACHE_AFFINITY`; all have `predictionState=MODELED`. |
| Global-window cardinality | Six contiguous clusters of eight decision records: `1789353805970`; `5983--5984`; `6003`; `6019--6020`; `6046--6047`; `6089--6090`. A PV record has no explicit plan ID, so clusters crossing an adjacent millisecond are inferred from contiguity, the synchronized release, and the configured B=8 cap. |
| Final placement | `172.27.112.157:8080@0/@1`, `172.27.112.158:8080@0/@1`, and `172.27.65.218:8080@0/@1` each received exactly 8 requests. |
| Cache | 48/48 predicted and actual cache hits were zero, as intended for the cold inputs. |
| Global rather than per-request greedy behavior | 28 selected candidates equalled the visible per-request minimum projected TTFT; 20 did not. The latter are the joint planner reserving/consuming virtual capacity to produce the balanced 8-per-engine final plan, rather than independently taking each visible minimum. |

The four Chat ingress Pods yielded 12, 9, 9, and 18 matching PD receipts,
respectively: all 48 request IDs were accounted for. Their engine-generated
`prefill_engine_trace` samples include `bs=8` for each logical engine:

| Logical Prefill engine | Maximum observed `prefill_engine_trace.bs` |
| --- | ---: |
| `172.27.112.157@0` | 8 |
| `172.27.112.157@1` | 8 |
| `172.27.112.158@0` | 8 |
| `172.27.112.158@1` | 8 |
| `172.27.65.218@0` | 8 |
| `172.27.65.218@1` | 8 |

This is the requested engine-side step-fill observation. The trace's load
telemetry reports `available=false`, and the direct vLLM logs do not emit one
aggregate scheduled-token-capacity value, so `15,976 / 16k` is token arithmetic
from the observed batch size and engine prompt length, not a GPU-utilization
metric.

## Cache-affinity mixed B=8 decision

The warm-up prefix request `0b054043-730d-9090-9970-4848d1804f7c` (2,490
tokens) was first placed on `172.27.112.157:8080@1`. After the cache event had
propagated, four hot variants preserving that full prefix and four unrelated
cold prompts were released in the same millisecond at `10:50:51.439`. All eight
requests returned HTTP 200. Seven routing records have timestamp
`1789354251923` and the eighth `1789354251924`, the same adjacent-millisecond
B=8 logging pattern as the cold run.

| Class | Request IDs | Placement | Projected / actual cache result |
| --- | --- | --- | --- |
| Hot (4) | `c225d1fe-78fe-9972-994b-0271a3bd924c`, `f333e566-221e-990e-81f5-78c332f0fce3`, `2a89b231-4c2d-9623-b84c-ba7f5c6cadbd`, `683a8cb7-3800-9cb9-aa39-ed1e7411c47f` | `.157@0` ×2, `.157@1` ×2 | 2,176 effective-hit tokens predicted and 2,176 actual hit tokens for all four; selected predicted TTFT 972 ms. |
| Cold (4) | `1341906f-d87e-9cc6-92a9-4842e4f1e951`, `9dd0a33c-c101-9a94-995b-c1b00c2e1d43`, `0896b6a8-1b49-960d-9c88-33d829a5e2a1`, `de0b19e6-79d5-92d5-a74f-18a5d44760c9` | `.65.218@0/@1`, `.158@0/@1` | zero predicted and actual hit tokens; selected predicted TTFT 2,492 ms. |

Chat PD receipts cover all eight: the four hot receipts report
`prompt_cached_token_num=2176` on the selected `.157` engines, and the four
cold receipts report zero on the four non-cache engines; all completed with
code 200 and reached the expected Decode Pod. The test exercises the positive
cache-affinity branch under B>1 global planning. It does not intentionally
create a cache candidate which violates the configured fixed TTFT allowance;
that rejection boundary remains covered by the focused
`cacheAffinityCannotSpendAnotherRequestsTtftAllowance` test below.

## Initial B=8 engine-side observation

The test's global input is approximately 16k tokens (`8 × 2,001`). The engine
receives 1,997 prompt tokens per request after the serving wrapper. Direct
Prefill vLLM logs and Chat engine traces show:

| Selected Prefill Pod | Requests selected from B=8 | Direct engine observation |
| --- | ---: | --- |
| `ds-38bd08d3-1-7dbd99d8b4-r7swm` (`172.27.112.158`) | 4 | Two engine log files each started two requests about 100 ms apart. A Chat `prefill_engine_trace` sample reported `bs=2`. |
| `ds-38bd08d3-1-7dbd99d8b4-bl5pp` (`172.27.112.157`) | 2 | One request started on each local engine; two sampled `prefill_engine_trace` records reported `bs=1`. |
| `ds-38bd08d3-1-7dbd99d8b4-84cmx` (`172.27.65.218`) | 2 | Both requests entered running state at `10:29:02.499`; sampled `prefill_engine_trace` records reported `bs=1`. |

Sampled Decode traces for the same request IDs reported batch sizes from 1 to 6
(maximum `bs=6`). All sampled results have status 200. The direct Prefill logs
do not emit one aggregate `16k scheduled tokens` line for this global window,
so no exact per-step token-budget utilization percentage is claimed.

This early eight-request calibration is retained to show why one global B=8
event alone does not guarantee a full step on every engine. The later 48-way
run supplies six B=8 planning events quickly enough for every logical engine
to reach an observed Prefill `bs=8`, without constraining the candidate set or
changing live routing scope.

## Focused local verification

The focused reactor test command was:

```text
./mvnw -pl flexlb-sync -am \
  -Dtest=CostBasedBatchedPrefillStrategyTest,GlobalBatchedSchedulingTest \
  -Dsurefire.failIfNoSpecifiedTests=false test
```

Result: **15 tests run, 0 failures, 0 errors; BUILD SUCCESS** at 10:17:03.
The `-am` dependency closure is material: a prior single-module attempt mixed
stale Maven snapshot classes with this checkout and is not counted as
validation. The focused cases cover virtual worker capacity, completion after
greedy blocking, exchange improvement, cache selection within the fixed TTFT
allowance, the per-request allowance guard, capacity skipping, global-window
timing, priority ordering, configuration boundaries, cancellation, and joint
cached-worker placement.

## Limits

- This is a placement and engine-observation test, not a claim of model quality,
  end-to-end latency improvement, cache eviction behavior, or full topology
  audit.
- `pv.log` does not expose a dedicated global-plan ID/cardinality field. The
  exact B=8 conclusion relies on the synchronized release, adjacent decision
  timestamp clusters of eight, and the B>1-only selection reason.
- The live candidate snapshot is truncated in PV output. Selected endpoints and
  the three current Prefill Pods were independently verified, but no claim is
  made that the printed candidate list is a complete topology dump.
- The safe live tests do not manufacture an `UNMODELED_PENDING_LRU` or
  `BATCH_NO_AVAILABLE_CANDIDATE` condition by corrupting prediction/worker
  state. Those protective branches therefore remain outside this live traffic
  run.
