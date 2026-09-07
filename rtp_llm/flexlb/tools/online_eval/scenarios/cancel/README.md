# Cancellation migration checkpoint

This is an incomplete first checkpoint: 11 of 19 old cancel cases are mapped.
`cancel_lifecycle` currently has 18 explicit variants and 38 profile instances.
The remaining preemption case and `cancel_fence_settlement` are still being built;
compilation is not remote correctness evidence.

Baseline is core `7120c1ff19446f694cbd99c5c9c5545bbd11e0c2`, with the status owner
predicate corrections `de9182d807` and `8f3d8064fd`. The cancellation adapter reuses
the bounded core request driver through the existing prepared-cohort abstraction.
No old case function is called, and no global catalog is mutated on import.

The first programs preserve these distinctions:

- `cancel_basic`: first output, explicit Master Cancel, frontend worker Cancel only
  for NON_BATCH, stream termination, recovery. BATCH additionally requires engine
  receipt within five seconds of Master Cancel issuance and the original ten-second
  Master drain. NON_BATCH engine receipt remains observation.
- `cancel_idempotent`: separate first and second RPCs, first engine receipt and
  forwarding counter, then a post-recovery counter proving no second forwarding
  under BATCH. The old closing drain was calculated but absent from `passed`; it
  remains observation here.
- `cancel_after_terminal`: completed request before Cancel, successful RPC(s), no
  engine cancellation rewrite and recovery. BATCH retains the closing drain.
  The old extra two-second wait result remains observational.
- `cancel_unknown_rid`: never-dispatched prepared ID, typed found=false or explicit
  gRPC NOT_FOUND, and unchanged ledger fingerprint. It does not send an engine RPC.

Every RPC receipt records its actual target, owner, request ID, start/end times and
response fields. Engine receipt timing uses the recorded issuance anchor; a late
receipt cannot pass merely because it eventually exists. Missing route, schema or
exit evidence is not empty success. Stream consumers must supply a done signal,
terminal fields and independently verified exit. Allowed cancellation transport
statuses are explicit; untyped errors are not accepted as cancellation evidence.

Ten focused local tests cover manual stream opening, unknown-ID RPC routing, missing
routes, late and incomplete receipts, verified exits and the profile-specific
programs. No remote run has been made for this checkpoint.

Second checkpoint adds sibling isolation, phase timing, anomaly, deadline exemption,
Schedule RPC drop and both autonomous stream-break cases. Explicit client-only
observation windows do not perform HTTP calls that could move the cancellation
phase. Schedule.future cancellation is identified as Python grpc.FutureCancelledError
and verified against the owned call, separate from gRPC wire CANCELLED.

The sibling group submits three independently shaped requests concurrently, caps
aggregate concurrency at 16 and rejects repeated cohort handles before dispatch.
The deadline case keeps its 45-second completion bound and clears enqueue_delay
after the recovery request, matching the old fault lifetime. BATCH-only five-second
receipt bounds and 95-second closing drain windows remain separate contracts.
