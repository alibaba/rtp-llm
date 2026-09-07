# Pending-drain SINGLE/BATCH migration

`elastic_pending_drain::single_batch_terminal::single-batch` is a separate
13-stage, 13-check candidate. The reviewed batch-window `legacy_terminal` and
unmapped stronger `zero_errors` variants stay batch-window only. The new variant
maps only the legacy pending-drain single-batch contract; it does not enable
single-nonbatch/window-nonbatch or extend any other elastic family.

## Configuration and mechanism

The reference is the profile-aware legacy `_pending_drain_spec` at
`a22f0678a2beb479c3da7ff9fa09df9c354f3d19`. Local tests compare the entire resolved
Master config, performance dictionary, 2P/2D topology, discovery mode and both KV
pool sizes for each declared profile. Only the decision differs across profiles:

| Property | batch-window | single-batch |
| --- | --- | --- |
| Scheduler / ordering / dispatcher | QUEUE / PRIORITY / BATCH | QUEUE / PRIORITY / BATCH |
| Decision | FIXED_WINDOW | SINGLE |
| Decision collection / prediction / request limits | 10ms / 550ms / 32 | Absent |
| Prefill batch inflight cap | 2 | 2 |
| Queue timeout override | Omitted | Omitted |
| Pre-event Prefill speed | Both 8000ms, sync1.5s | Both 8000ms, sync1.5s |

The Java sources are unchanged. `WorkerBatcher.java:256` selects SINGLE;
`processQueue` at1535 chooses the decision method, and1581 submits `List.of(head)`
to the same admission/delivery boundary. `DeliveryCreditPolicy.java:22,45`
selects **batch** publication capacity from the BATCH dispatcher, using the batch
inflight limit; SINGLE does not convert that owner into per-request/non-batch
publication credits. `BatchDeliveryStrategy.java:34` owns EnqueueBatch admission
and transport. `WorkerBatcher.stopAndDrain` at545/615 detaches queued requests
and projects their failure without a decision-specific branch. These are code
paths, not a claim that an observed client terminal released Engine slots or KV.
Paths above are under `flexlb-sync/src/main/java/org/flexlb/balance/scheduler/`.

The serial wave remains1024 input /2 output /3 unique keys, Schedule30s,
concurrent Fetch consumers, and300ms spacing after accepted requests; it stops at
four victim routes or14 attempts. Under SINGLE each dispatched decision has one
member, so two batch credits can leave additional victim routes in the Master
queue. The existing construction checks remain mandatory: at least3 victim
routes, `routed - engine(waiting+running) >= 1`, no earlier victim client terminal,
and no victim completed-counter increment. This remains an **aggregate** pending
estimate; exact pending RID identities are not inferred from those counters.

The new `batch_path.batch_fetch` guard requires every admitted stream to use
FetchResponse, the client path chosen by Schedule's `enqueued_by_master` flag.
It rejects a GenerateStreamCall fixture accidentally substituted for BATCH.
It does **not** claim every request already reached EnqueueBatch: queued,
undispatched requests intentionally have Fetch consumers waiting for dispatch.

## Preserved verdicts and evidence limits

The server graceful drain cap stays60s with95s HTTP client allowance. Every
victim-routed request must have a visible completed or explicit-error terminal
within40s after removal. The50s accounting window starts after collection and
checks scheduler, Prefill batch and Decode load separately. Restore100ms,
recovery20 at concurrency10 with>=19 successes, and victim disappearance from
mock/discovery/Master retain the batch-window program's thresholds and timing.
No zero-error requirement is substituted for the legacy allowed visible errors.

The independent SINGLE/BATCH fixture runs the actual RecordedRequests consumer
implementation against simulated Schedule and Fetch RPCs. Its initial engine
counters start at zero; each accepted request either occupies one singleton batch
ledger (cap2) or enters a separate queue. It produces two modeled victim ledgers
and two extra queued victim routes before removal. Those exact fixture identities
are test inputs, not an assertion that real aggregate observations identify them.
The modeled removal retires the queued work and finishes active/survivor work
within16s; this synthetic timing is not Java latency evidence.

Negative fixtures bypass the modeled batch cap (so the pending estimate must
fail), inject an earlier completed count, return the non-batch stream path, and
exercise recovery19/20 versus18/20. The same shared40s/50s boundary tests remain in
the pending suite. None of these modeled results execute Java's SINGLE decision.
Independent static review and a fixed-SHA real single-batch run remain required;
legacy single-batch remains available until acceptance.
