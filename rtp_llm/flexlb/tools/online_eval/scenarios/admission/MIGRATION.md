# Admission migration checkpoint

`admission_queue` is implemented with full-program local fixtures: queue_depth, slo_deadline and master_capacity, each in batch-window and single-batch. `engine_admission_gate` now has four programs/eight profile instances and focused local tests, with full-program external-I/O fixtures, but still needs independent review. `priority_admission` adds the current no-preemption incomer contract with one batch-window instance and an execution fixture. Batcher/placement and prefill token budgeting (seven legacy cases) remain pending. Existing cases are retained. This is a local implementation checkpoint, not Java execution evidence or full migration acceptance.

The profiles follow the current case declarations, not the older mapping table: both SLO and master-capacity now include single-batch; queue-depth requires the BATCH dispatcher. Queue-depth uses the original default 2P/4D; SLO and master-capacity use dedicated 2P/2D environments. Config overrides preserve each selected profile's decision/dispatcher axes.

| Legacy case | YAML variant | Retained predicates / stage.check |
|---|---|---|
| admission_queue_depth_reject | queue_depth | occupants.all_occupied observes each P waiting+running>=1 before probe; depth_error.criterion preserves case-sensitive queue depth text matching; fast_reject.criterion latency<3s; recovered.criterion one successful fresh request; master_clean.inflight; engine_clean.engine_inflight with explicit10s timeout |
| admission_slo_queue_deadline | slo_deadline | priority ordering, queue timeout1500ms, kv_pressure6291456 tokens on both P; deadline_error_family.criterion preserves old literal error family; waited.criterion>=1s and bounded_deadline.criterion<=8s; clear and1s poll; recovered.criterion; master_clean.inflight20s |
| admission_master_capacity_reject | master_capacity | priority ordering, queue timeout60000ms, outstanding cap2; four concurrent requests; at_least_one_reject and at_most_two_rejects; at_least_two_served; no_serving_error; typed_code reads actual response code8502; typed_detail requires TooManyRequests and QUEUE_FULL; reject_fast<3s; fresh recovery; master_clean.inflight30s |

Actions expose bounded traffic, occupancy evidence, awaiting and individual predicates. They do not call legacy case implementations. Waves reuse the core RequestBatch RPC and consumer-completion implementation, retain raw Schedule response code/success/text, and register cleanup before submission. Submission workers have independent done signals; cleanup cancels actual calls, waits the submission futures, and invokes core consumer cleanup. Missing evidence or unexpected execution errors remain ERROR, not findings. Seed business rejections are retained, and no seed success-rate threshold is imposed.

Explicit stricter observations: master_clean also requires current full ready topology; engine_clean checks every configured prefill and valid leak flags. These additions are not described as old-only equivalence. The SLO error-family check intentionally preserves literal text matching; it is not a typed error-code claim. Only capacity typed_code reads the numeric Schedule response.

Complete compile-to-execute fixtures exercise all six programs and distinguish wrong numeric capacity codes, SLO errors that arrive too early, and consumer cleanup errors from a green run. External Java I/O is explicitly replaced in these fixtures; no real Java success is inferred.

Pending before acceptance: independent legacy-contract review, integration through the core-owned catalog, and scheduled real Java validation. No remote load was started for this checkpoint.


## Engine gate programs awaiting independent acceptance

| Legacy case | Variant | Retained predicates / stage.check |
|---|---|---|
| engine_prefill_concurrency_gate_park | prefill_concurrency | four separated Schedule calls, all_admitted; park_seen max(prefill_waiting_batches, waiting)>=1 within10s; all_completed=4; park_empty both counters0 within10s; master_clean30s; recovered |
| engine_decode_hard_gate_unbounded_park | decode_hard_gate | 280 serial Schedule calls,64 output tokens,50ms pacing per25; decode routing cap5000; all_admitted=280;18s peak observation; conditional_park only requires waiting>=1 when running_max>=128; completed_95pct>=266; park_empty15s; master_clean60s; recovered |
| admission_engine_waiting_batch_cap_reject | prefill_waiting_cap | two occupants under3000ms/max_waiting_batches1; cap_seen>=1 before probe; backpressure_error matches both old tokens and fast_reject<3s; cap reset0 before fourth; occupants_complete=2 and pressure_recovery=1; park_empty10s; master_clean30s; recovered |
| admission_engine_kv_lack_mem_fast_reject | kv_pool_capacity |17-block P pool; two requests each eight disjoint block keys; pool_full>=16 before probe; lack_mem_error matches all three old strings and fast_reject<3s; occupants_complete=2; pool_recovered available>=8; fresh_succeeded eight-block request; explicit P/D engine_clean; master_clean30s; recovered |

`admission_fire` waits only for each Schedule submission, retains immediate versus deferred consumption explicitly, and applies declared inter-fire spacing. Deferred mode performs no Fetch until `admission_wait`. Core-owned consumer completion checks remain mandatory at wait/cleanup. KV keys come from a separately reserved request-ID seed to make the per-request key sets disjoint; the actual key list remains in request-shape evidence. The seed is not an additional inference request.

Decode's old conditional park contract can pass below the gate, with gate_filled=false explicitly recorded; this does not demonstrate that overflow occurred. Waiting-at-fourth is retained as diagnostic evidence and is not silently upgraded into a new old-contract gate. The core's treatment of unexpected RPC/execution errors is retained; real Java validation is still pending.


## Priority admission

The old name `admission_priority_incomer_reject` now maps to `priority_admission/permit_released_without_preemption`. Its current contract admits the incomer:1P/1D, PRIORITY ordering, queue timeout60000ms, one delivered-not-accepted permit and no preemption block. Victim priority30/output200 is first admitted (`victim_admitted`); `victim_running` requires decode running>=1 within10s. Incomer priority70/output2 must receive actual Schedule code200 (`incomer_code`) and success (`incomer_admitted`) within<3s (`accepted_fast`, Schedule timestamps only). Both streams must complete (`victim_unmolested`, `incomer_completed`), followed by fresh recovery, master-clean30s and P/D engine-clean15s. Recovery priority remains unset. This preserves the new-B permit-release contract; the historical `_reject` suffix does not select a rejection program.
