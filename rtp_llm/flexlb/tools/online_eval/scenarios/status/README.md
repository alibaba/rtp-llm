# Status protocol migration

This directory is an in-progress migration of `batch_ack_and_execution` (four
legacy cases) and `status_protocol` (21 legacy cases). The existing Python suite
remains registered. Compiling a definition does not establish remote correctness
or complete the 25-contract migration.

`batch_ack_and_execution.yaml` names every experiment stage explicitly. ACK
partial failure retains member accounting, the transient retry policy, permanent
error isolation, dispatch-count bounds and cleanup. Execution partial failure has
separate normal and slow-Prefill arms, typed execution terminals, cleanup,
idempotency and recovery. Multi-error checks each injected code separately. ACK
drop keeps bounded residue, non-growth after eight seconds, and eventual drain;
only the drain contract is a finding. A sampling or stage error is never a finding.

The adapter exports reusable operations, without calling legacy case functions:

- `status_prepare` freezes request IDs before fault installation.
- `status_dispatch` submits the finite cohort with bounded concurrency, using the
  core RPC driver. Deferred mode performs no Fetch until the ordinary `wait`.
- `status_control` applies a named supported injection and registers owner-scoped
  cleanup before its first HTTP mutation. Unknown fields, including accidentally
  ignored camelCase `batchId`/`errorCode`, are rejected.
- `status_perf` uses an explicit performance setting and explicit restore value.
- `status_sample` stores raw sources, capture times and environment epoch in a
  frozen snapshot. Missing fields or failed sources raise an execution error.
- `status_check` checks one declared metric boundary; `status_outcomes` checks
  complete per-request outcomes and distinguishes Schedule rejection from typed
  execution failure. The request count is not a claim about one physical batch.

Decode `/inflight_status` exposes `total_load` and layered admission counters,
not the old `inflight_requests` field. The new `decode_total_load` metric and
fingerprint use these live fields and reject missing schema. A zero test may
combine independent owner counts; their raw fields remain separate in evidence.
The fingerprint includes the Decode layers rather than treating an absent old
key as zero. Prefill batches and Prefill members remain different metrics.

The four ACK variants currently compile, and eight targeted fake tests cover
prepared-ID binding and deferred Fetch, separate failure phases, missing-owner
schema, partial injection cleanup, epoch refusal and validation. Remote execution
and the remaining status variants are still pending.
