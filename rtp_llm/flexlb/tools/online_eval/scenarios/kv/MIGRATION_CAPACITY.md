# KV capacity migration

This family migrates four retained legacy cases to explicit stages, with four
profiles each (16 instances). Source baseline: `9d8576c44bf6dda4191f5d9070c3cc929b7fc5a0`.
The source cases are `flexlb_ft/cases/kv/kv_<variant>.py`; constants and request
helpers are in `flexlb_ft/support/kv.py` and `support/requests.py`.
No legacy case is invoked or deleted. Independent static acceptance covers
`432b85f3577c0c65646652ac01aee15ba6e436c3`; integrated suite verification and Java
execution remain pending. Local external-IO fixtures are not Java evidence.

## Integration boundary

The dedicated `actions/kv_capacity.py` exports `HANDLERS`. The core integrator
must merge these handlers into the default catalog; this branch deliberately
does not edit the shared catalog, compiler, backend, or `actions/kv.py`.
Tests explicitly merge the handlers and exercise the real compiler and runtime.
All dispatcher choices use compiled `effective_axes`, including the error wrapper
and BATCH deferred versus NON_BATCH immediate stream consumption.

## Preserved cases and predicates

| Variant / legacy suffix | Environment | Construction and acceptance |
| --- | --- | --- |
| `decode_pool_exhaustion_terminal` | 2P/1D, D pool 3 | 2560/2 probe with two explicit keys; LACK_MEM plus decode-side text, BATCH additionally EnqueueBatch rejected, elapsed <3s. D permanent counter grows exactly one within 10s, retry counter remains flat; every P permanent counter stays flat and held blocks return to its own baseline within 10s. A 1024/2 headroom request succeeds, D counters stabilize, Master and engine tables clean separately (30s each), fresh generic recovery succeeds. |
| `decode_capacity_park` | 2P/4D, default pools | Exhaust each D's actual available+active tokens, sync 1.5s; snapshot pre-probe scheduler/P-batch/D-reservation watermarks. 2048/10 Schedule reaches its 5s client deadline and never starts a stream. After .5s, no engine lifecycle contains this request. Explicit Master Cancel, .5s pause and 10s poll restore each total to <= its pre-probe watermark. Clear all pressure, sync 2s, then both fresh and generic recovery succeed. No global-zero replacement of the ownership watermark. |
| `pool_saturation_evict_reject_recover` | 1P/2D, P pool 27, D pool 32 | Ten serial 2048/2 requests, eight disjoint keys each, all succeed; every post-request sample conserves held+referenced+available=capacity, eviction delta >=1, final key count <=27. Slow P to 3000ms, sync 1.5s, fire three eight-key occupants .4s apart, pause .3s. Preserve the ignored 2s/.05s peak wait followed by a separate leading sample. Probe typed LACK_MEM/insufficient KV cache (+BATCH wrapper) in <3s; two follow-up requests may succeed but any failure must carry LACK_MEM in <3s; total failures 1..3. Leading plus 3s/.1s samples show peak held >=24, available floor <=3, conservation always; permanent reject delta >=1. Drain all three occupants successfully, recover >=8 available blocks within 10s, fresh request succeeds, separate Master/engine clean checks and generic recovery, restore perf. No explicit cache eviction API is used. |
| `capacity_conflict_overflow` | 2P/2D, D pool 180, default P pool | Prime 40-key family at 40960/2; resolve actual holder, wait both key sets quiet 3.5s within 8s. Slow both P to 5000ms, sync 1.5s; seed 147456/2 with the same 40-key prefix plus 104 tail keys must land on holder and appear pending within 6s. Restore cool P to 100ms, pause .3s, measure 2048/2 baseline. Five 40960/2 same-prefix requests with .12s gaps, drain seed+wave, then measure same-prefix probe. Preserve that source order even though its comment calls the later probe a live-ledger measurement. P6 requires all five admissions and successful probe, hard hot share <1; P5 share bands strict/normal/loose=0/.05/.1; P7 probe/baseline ratio bands=2/3/5. BATCH uses Schedule-to-transport-end duration, NON_BATCH uses Schedule-to-first-output TTFT. Restore both P to 100ms. |

Explicit key lists preserve family equality, disjointness and seed-prefix
relations. Absolute key bases are fixed in each fresh isolated environment;
wire request IDs remain allocated by the backend. Ordinary requests retain
15s stream bounds; fired cohorts retain 30s stream bounds, with bounded stage
deadlines large enough for their sequential drains. Counter samples are saved
with owner names, timestamps and requested raw values. Watermark artifacts also
retain the raw Master response.

## Explicit execution differences

- Missing/malformed owner counters, lifecycle maps or Master watermark data are
  ERROR, not the legacy helper's default zero. HTTP failures propagate. Sampling
  takes a final endpoint observation when its bounded interval expires.
- Request success requires actual business completion and verified consumer
  termination. Expected UNKNOWN/INTERNAL stream failures are accepted only with
  explicit per-stage RPC policy and completed transport/consumer evidence; a
  stage timeout or unverified consumer is never an expected capacity failure.
  The parked probe allows only actual Schedule DEADLINE_EXCEEDED, with no stream.
- The legacy conflict case ignores drain outcomes. The new runner still enforces
  its transport/error and cleanup contract during that drain; this can report
  ERROR where the old case ignored a consumer failure. This is an explicit
  execution strengthening, not a claim of identical failure classification.
- Timings use backend Schedule start and actual first-output/transport-terminal
  timestamps. This preserves the measured phases, but excludes tiny outer helper
  setup/return overhead in the legacy wall-clock probe. No server latency or
  invented timing zero substitutes for missing client evidence.
- Fault/perf actions retain registered compensating cleanup and validate actual
  engine ownership/ACK. Cleanup errors remain visible even if business checks
  pass. The old best-effort finally blocks could suppress these failures.
- Stage failures block dependent work; independent cleanup still runs. Legacy
  cases sometimes continued subsequent observations after a failed predicate.

## Local verification

`test_scenario_kv_capacity.py` executes all 16 programs using external IO fixtures
and real stage handlers. Counterexamples reject the wrong D counter family,
request delivery despite a parked deadline, pool conservation failure, and a
wave that never spills from the hot holder. Further tests reject missing owner
counters, unverified expected-error consumers, and invalid request policy values.
The parked fixture starts with nonzero watermarks (7/2/3) to guard resource
ownership semantics. Integration and Java evidence must cite a fixed commit.

Initial local results: eight capacity tests PASS (2.504s). The full scenario
suite on this deliberately unintegrated 9d8576 base ran 276 tests (40.040s):
seven errors are the existing KV tests scanning this new YAML before catalog
registration; two FIFO fixture failures are the already identified admission
fixture issue fixed separately by `258c10b1effa8f777e481076322ed58f753689cb`.
Raw output is `/tmp/agent4-kv-capacity-scenario-tests.log`. These results do not
constitute a passing integrated suite: register handlers and retain the FIFO
fixture fix in the core integration, then rerun once on its fixed commit.


## Independent static acceptance

Agent3 independently reviewed fixed commit
`432b85f3577c0c65646652ac01aee15ba6e436c3` against all four legacy programs,
the new action handlers, compiled plans and fixture boundaries. The review
accepted 4 variants / 16 instances / 224 checks with no remaining confirmed
static blocker. Its isolated source is `/tmp/agent3-kv-capacity-432`; its eight
capacity tests passed in 2.519s with exit code 0. The review explicitly retained
the execution differences above and did not claim identical error classification.
The reviewed test setup merges HANDLERS explicitly. Default catalog integration,
whole-suite validation and actual Java evidence remain separate outstanding gates.
