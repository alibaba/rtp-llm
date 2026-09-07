# Admission migration checkpoint

Only `admission_queue` is implemented here so far: queue_depth, slo_deadline and master_capacity, each in batch-window and single-batch. The other four families and twelve legacy cases remain pending. Existing cases are retained. This is a local implementation checkpoint, not Java execution evidence or full migration acceptance.

The profiles follow the current case declarations, not the older mapping table: both SLO and master-capacity now include single-batch; queue-depth requires the BATCH dispatcher. Queue-depth uses the original default 2P/4D; SLO and master-capacity use dedicated 2P/2D environments. Config overrides preserve each selected profile's decision/dispatcher axes.

| Legacy case | YAML variant | Retained predicates / stage.check |
|---|---|---|
| admission_queue_depth_reject | queue_depth | occupants.all_occupied observes each P waiting+running>=1 before probe; depth_error.criterion requires error text containing queue depth; fast_reject.criterion latency<3s; recovered.criterion one successful fresh request; master_clean.inflight; engine_clean.engine_inflight |
| admission_slo_queue_deadline | slo_deadline | priority ordering, queue timeout1500ms, kv_pressure6291456 tokens on both P; deadline_error_family.criterion preserves old literal error family; waited.criterion>=1s and bounded_deadline.criterion<=8s; clear and1s poll; recovered.criterion; master_clean.inflight20s |
| admission_master_capacity_reject | master_capacity | priority ordering, queue timeout60000ms, outstanding cap2; four concurrent requests; at_least_one_reject and at_most_two_rejects; at_least_two_served; no_serving_error; typed_code reads actual response code8502; typed_detail requires TooManyRequests and QUEUE_FULL; reject_fast<3s; fresh recovery; master_clean.inflight30s |

Actions expose bounded traffic, occupancy evidence, awaiting and individual predicates. They do not call legacy case implementations. Waves reuse the core RequestBatch RPC and consumer-completion implementation, retain raw Schedule response code/success/text, and register cleanup before submission. Submission workers have independent done signals; cleanup cancels actual calls, waits the submission futures, and invokes core consumer cleanup. Missing evidence or unexpected execution errors remain ERROR, not findings. Seed business rejections are retained, and no seed success-rate threshold is imposed.

Explicit stricter observations: master_clean also requires current full ready topology; engine_clean checks every configured prefill and valid leak flags. These additions are not described as old-only equivalence. The SLO error-family check intentionally preserves literal text matching; it is not a typed error-code claim. Only capacity typed_code reads the numeric Schedule response.

Pending before acceptance: full compile-to-execute scenario fixtures, independent legacy-contract review, integration through the core-owned catalog, and scheduled real Java validation. No remote load was started for this checkpoint.
