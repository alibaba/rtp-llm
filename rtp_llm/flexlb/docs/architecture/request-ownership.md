# Request ownership and event paths

The directory identifies an exact request generation. `RequestSlot` owns the
complete per-request decision. Endpoint ledgers, timers and frontend publication
retain their existing transaction and execution responsibilities.

| Component | State and responsibility | Entry points |
| --- | --- | --- |
| RequestRegistry | Canonical ID-to-slot mapping, registration mutex, global admission shutdown barrier | register, exact ID/item resolution, conditional tombstone removal, snapshots, shutdown |
| RequestSlot | Request state, admission first cause and retained facts, delivery/engine ownership, preemption, deadline validity, terminal and response selection | cancelRequest, onAdmissionCompleted/Failed, commitRoute, claimDelivery, onDeliveryStarted/Result, observeEngineFact, onPrefill/DecodeRetired, onPreemption*, expireInactiveRequest, onDecisionExpired, onShutdown |
| AdmissionMutation | One-shot completion of one exact admission | close, terminate; callbacks capture their slot and release the global admission count |
| DeliveryClaim | One exact delivery attempt, callback consumption guarded by its slot | begin, publishRoute, complete |
| PreemptionRegistration | One exact preemption attempt and its observed protocol phase | applyPhase, release, settleTerminal; callbacks go directly to its slot |
| RequestTerminalCleanup | Executes a detached terminal action, isolates cleanup failures, commits tombstone, submits selected response | finishTerminal, submitTerminal |
| RequestCompletionPublisher | Frontend execution and publication shutdown accounting | reserve publication, submit response, adapt external Future operations |
| ExpirationTimer | Timer scheduling and exact timer capabilities; periodic retention scan | attach/cancel deadlines; deliver expiry to the owning slot; remove exact tombstones via directory |
| Endpoint | Queue and resource ledgers, exact handoff/settlement transactions | unchanged exact item/reservation operations |
| RequestResponses | Defensive response representation construction | success/error construction, server status copies |

Registry retains ID/item entry points where callers do not already possess a
capability. They resolve identity and dispatch one event; they do not interpret
cancellation first cause, transport outcome or terminal replay. Delivery and
preemption callbacks no longer return to the registry or resolve a bare ID.
The slot has no directory reference: its only global admission dependency is the
completion callback for the already-counted mutation.

## Execution paths

- Registration: Registry registers a configured slot and its original Future,
  then installs expiration. DIRECT and QUEUE keep their existing selection paths.
- Admission: the global gate admits an operation; the slot issues its exact
  mutation. The slot binds the item under its monitor. Endpoint queue publication
  runs outside that monitor while the admission mutation pins the binding.
  Failure rolls back that exact binding. Mutation completion resumes retained
  request facts inside Slot, then releases the global admission count.
- Delivery: resolve exact item, claim the slot and Endpoint handoff, send through
  the existing delivery strategy, report through DeliveryClaim. Slot interprets
  definite failure, uncertainty and prior Engine acceptance. BATCH sends its RPC;
  NON_BATCH publishes the route after starting the same lifecycle.
- Worker facts: the projector preserves synchronous exception isolation; Registry
  locates a slot; Slot validates item/reservation/Endpoint identity and reduces the
  fact. Endpoint retirement uses the same exact evidence rules.
- Cancellation: Slot checks the expected batch, first cause, admission and delivery
  ownership. It either retains the cancellation or claims terminal ownership.
  Admission completion, Worker facts and inactivity all resume through Slot rules.
- Preemption: the cross-request coordinator still owns Engine Cancel sequencing.
  Its exact PreemptionRegistration reports phases to the owning Slot; Endpoint
  reconciliation transactions retain resource settlement authority.
- Terminal: Slot selects immutable TerminalOutcome and claims TERMINALIZING,
  detaching cleanup/publication capabilities. RequestTerminalCleanup executes
  selected leaves outside the slot monitor, then Slot commits TOMBSTONE. Publisher
  selects the frontend result through Slot and completes the Future outside locks.
- Shutdown: close registration/admission and wait for in-flight mutations, ask
  each slot for its shutdown action, execute actions, then close Timer and Publisher.

## Concurrency contracts

1. Slot state uses the existing slot monitor. Global registration/admission gates
   must not be held while executing Endpoint queue publication or user callbacks.
2. Endpoint settlement is not ordinary deferred cleanup. Existing atomic handoff
   and priority-reconciliation transactions remain at their established decision
   boundaries and lock order.
3. First cancellation cause, terminal ownership, Endpoint settlement and frontend
   publication are separate decisions. A published ACK does not end resource
   lifetime; expiry can finish the request without replacing that response.
4. Publication may re-enter Slot to arbitrate against a terminal selected after
   ACK confirmation. Publisher does not interpret lifecycle fields itself.
5. TERMINALIZING excludes competing request actions before cleanup; TOMBSTONE is
   committed after cleanup. Cleanup failure remains isolated and logged.
6. Directory removal invalidates the slot under its monitor. Old delivery,
   preemption, timer and Future capabilities cannot resolve a replacement request.

TerminalOutcome replaces arbitrary externally supplied transition functions.
The resource flags in TerminalAction are internal decisions built by Slot;
RequestTerminalCleanup only consumes them and never chooses an outcome.

## Validation

Existing lifecycle, admission, publication-race, delivery-lock, Endpoint and
preemption assertions are retained. Test interception follows DeliveryClaim and
Slot timer events instead of removed Registry callback methods. A new regression
expires and removes a request, reuses its ID, and delivers old delivery/preemption
callbacks; the replacement must remain queued and its Future incomplete.
Performance measurements use tools/run_queue_performance.sh with 750P/750D,
BATCH/FIXED_WINDOW and NON_BATCH/SINGLE, 3000/10000 QPS, 10s warmup and 10s measure.
