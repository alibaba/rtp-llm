# Request ownership and event paths

The directory identifies an exact request generation. `RequestSlot` owns the
complete per-request decision. Endpoint ledgers, timers and frontend publication
retain their existing transaction and execution responsibilities.

| Component | State and responsibility | Entry points |
| --- | --- | --- |
| RequestRegistry | Canonical ID-to-slot mapping, registration mutex, global admission shutdown barrier | register, exact ID/item resolution, conditional tombstone removal, snapshots, shutdown |
| RequestSlot | Request state, admission first cause and retained facts, delivery/engine ownership, preemption, deadline validity, terminal and response selection | cancelRequest, finishAdmission/terminateAdmission, commitRoute, claimBatchDelivery/claimRouteDelivery, setDeliveryPrediction, publishRoute, completeDelivery (through DeliveryClaim), processPrefillStatus/processDecodeStatus, recordPrefillRetirement/recordDecodeRetirement, updatePreemption/releasePreemption/completePreemption, expire, prepareShutdown |
| RequestSlot.AdmissionHandle | One-shot completion of one exact admission | close, terminate; callbacks capture their slot and release the global admission count |
| RequestSlot.DeliveryClaim | Exact asynchronous delivery identity; only Slot can construct it and consume its result | complete |
| PreemptionRegistration | One exact preemption attempt and its observed protocol phase | applyPhase, release, settleTerminal; callbacks go directly to its slot |
| RequestTerminalCleanup | Executes a detached terminal action, isolates cleanup failures, commits tombstone, submits selected response | finishTerminal, submitTerminal |
| RequestCompletionPublisher | Frontend execution and publication shutdown accounting | reserve publication capacity, submit selected completion, execute selected completion synchronously |
| ExpirationTimer | Timer scheduling and exact timer capabilities; periodic retention scan | attach/cancel deadlines; deliver expiry to the owning slot; remove exact tombstones via directory |
| Endpoint | Queue and resource ledgers, exact handoff/settlement transactions | unchanged exact item/reservation operations |
| Response | Defensive response representation construction | success/error construction, server status copies |

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
- BATCH delivery: Registry resolves the exact item; Slot claims the existing batch
  transaction's Endpoint handoff. The strategy recalculates work for the final
  claimed members, supplies predictions through Registry, then submits the RPC.
  `DeliveryClaim.complete` returns directly to the original Slot.
- NON_BATCH delivery: Slot claims the existing CommittedAdmissionOwner (shared by
  queued and direct routes). Registry.publishRoute delegates to that exact Slot;
  one operation updates prediction, reconciles Engine acceptance and acknowledges
  the route. Publication executes after releasing the monitor.
- Worker facts: the projector preserves synchronous exception isolation; Registry
  locates a slot; Slot validates item/reservation/Endpoint identity and reduces the
  fact. Endpoint retirement uses the same exact evidence rules.
- Cancellation: `Slot.cancelRequest` validates identity, records the first cause and
  decides whether local termination is possible under one monitor acquisition.
  Selected terminal cleanup executes outside the monitor. Admission completion
  settles retained evidence and cancellation in the same decision; Worker status
  and inactivity apply their own resource-settlement rules.
- Preemption: the cross-request coordinator still owns Engine Cancel sequencing.
  Its exact PreemptionRegistration reports phases to the owning Slot; Endpoint
  reconciliation transactions retain resource settlement authority.
- Terminal: Slot selects immutable TerminalOutcome and claims TERMINALIZING,
  detaching cleanup/publication capabilities. RequestTerminalCleanup executes
  selected leaves outside the slot monitor, then Slot commits TOMBSTONE. Slot selects the frontend result before submission; Publisher completes the Future outside locks.
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
4. Slot arbitrates publication against a terminal selected after ACK confirmation,
   before submitting a concrete SelectedPublication. Publisher never re-enters
   request decisions; it only executes the selected completion and accounts for it.
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

## Concrete method boundaries

- `RequestFuture.complete/completeExceptionally/cancel` enter the corresponding
  Slot event first. Slot performs local terminal cleanup and response selection,
  then calls `Publisher.publishNow` to preserve synchronous Future semantics.
- Slot.selectResponse/selectFailure/selectCancellation explicitly consumes an exact
  permit and arbitrates the frontend result. PublicationPermit only owns capacity,
  one-shot consumption and close accounting; it never calls back into Slot.
  A later request terminal cannot replace a response that already won.
- `Publisher.submit` queues an already-selected completion. `publishNow` executes
  it synchronously. Neither accepts decision functions or invokes Slot reducers.
- `Slot.processPrefillStatus/processDecodeStatus` and the three preemption operations each
  invoke a named reducer under the Slot monitor. Their shared helpers execute
  already-determined effects rather than accepting a state-mutating Function.
- `RequestSlot.AdmissionHandle` captures its exact Slot and invokes named admission events;
  callers cannot provide arbitrary completion/termination consumers.
- Slot-only reducers are private; exact event methods used by sibling scheduler
  classes are package-visible. Public token operations remain available to
  delivery and preemption callers. Stateless helpers are static.

## Deadline and fact settlement

Timer delivers one exact deadline to `Slot.expire`. Slot validates and consumes
it and makes the complete decision under its monitor. Timer never interprets an
expiry result to choose another request operation. Inactivity wake-ups are rearmed
after processing; installing the next task still checks Slot eligibility.

Ordinary delivery confirmation and terminal evidence directly produce a
`RequestEffect` containing an already-selected delivery publication or terminal
action. `PendingReplay`, `PreemptionReduction`, and post-lock materialization are
removed. The effect executor performs no business decisions: it runs the selected
action and then signals an exact preemption observation if required.

Only blocked evidence is retained by admission or preemption. Once constraints
allow settlement, the same terminal/delivery decision methods are used. Endpoint
reconciliation remains atomic before a decision that relies on its outcome.
Slot, rather than PreemptionRegistration, selects the stronger terminal evidence.
Cancellation paths share the resource-disposition decision, while retaining the
distinction between local cancellation and inactivity accounting expiry.

These operations are not sequential request stages. Claiming delivery precedes
transport submission; its completion arrives through that exact DeliveryClaim.
Worker facts, cancellation, deadlines and preemption can interleave with delivery.

Endpoint queue publication suppliers and individual resource cleanup leaves
remain. Delivery preparation and handoff use existing concrete transactions;
no Function or Runnable carries ordinary terminal/delivery decisions through a
replay layer. This does not introduce a universal event switch
or change the separate request-terminal and frontend-publication winners.

## Cancellation boundaries

- `cancelRequest` is the complete business operation; Registry only locates and
  dispatches to the exact Slot. Its snapshot result is not a cancellation-winner flag.
- Private `recordCancellationLocked` records only the first cause, closes admission
  and sets CANCEL_REQUESTED. It neither releases resources nor selects a response.
- Private `tryTerminateCancellationLocked` selects a terminal action only if no
  admission remains open and ownership permits local cancellation. Otherwise the
  request waits for resource evidence; repeated cancellation cannot replace its cause.
- Inactivity expiry uses `beginExpiredRequestLocked` for accounting cleanup. It does
  not switch ordinary cancellation behavior with a boolean argument.
- `settleAdmissionLocked` handles retained evidence and any pending cancellation
  together. There is no separate cancellation replay after releasing the monitor.
- Worker termination selects its resource cleanup from Endpoint evidence and its
  outcome from the preserved first cause. It has no cancellation-specific wrapper chain.
- `commitTerminalStateLocked` commits all terminal phases, including CANCELLED,
  after cleanup. `Future.cancel` remains a separate synchronous Future adapter.

## Delivery interfaces and ownership

Registry exposes concrete `prepareBatchDelivery`, `prepareBatchMember` and
`prepareRouteMember` operations. Slot validates the exact item and invokes the
existing transaction under its monitor. There is no `prepareIfOwned(Supplier)`
or delivery-handoff BooleanSupplier, and no new preparation capability.
The shared eligibility check reads Slot fields directly; it does not allocate a
snapshot or repeat the same checks through a second eligibility method.

BatchDeliveryStrategy.BatchTransaction supplies the batch ID and exact handoff.
Both direct and queued route delivery use PrefillAdmissionResources.CommittedAdmissionOwner.
There is no artificial transaction interface or duplicate route adapter.

`RequestSlot.DeliveryClaim` exposes only `complete(result)`. Its constructor is
private to Slot. Each generation can issue at most one claim, so the existing
exact item, kind and batch identity suffices to validate it; Slot does not add a
second current-claim field. Terminal cleanup still drops the Slot's item reference.
The asynchronous sender may retain its original claim, but it cannot reach a reused ID.

Slot's `completeDelivery` consumes a result once and handles success, definite
rejection and uncertainty. Only actual BATCH success records an RPC ACK timestamp.
Exact Decode acceptance overrides transport failure; ambiguous transport retains
resources until Engine evidence or expiry settles them.

The private `acknowledgeDeliveryLocked` is shared by delivery success, Engine
acceptance and released preemption. It retains confirmation only when blocked;
otherwise it commits ACKNOWLEDGED, detaches the scheduling deadline and produces
one DeliveryPublication. The old type-specific confirmation wrappers, state-only
ACK methods and nested DeliveryConfirmation payload are removed.

NON_BATCH prediction and acknowledgement share one monitor acquisition. Detached
deadline cancellation, metrics and response execution remain outside the monitor.
ACKNOWLEDGED and winning the frontend response are still separate: Slot explicitly
selects the response after deadline cancellation and reporting, preserving the
existing opportunity for cancellation or termination to win before publication.
