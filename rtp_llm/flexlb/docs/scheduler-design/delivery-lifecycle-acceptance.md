# Delivery lifecycle acceptance criteria

This contract is the gate for the delivery stop and preemption refactor. The
refactor must preserve these behaviors before performance results are compared.

## Functional contracts

| ID | Contract | Test coverage |
| --- | --- | --- |
| DL-01 | A stop for a claim that has not started sending returns `NOT_SENT` and does not call the engine. | `DeliveryStopTest.acceptance_unsentStopDoesNotIssueRemoteCancel` |
| DL-02 | A remote `REQUEST_CLEANED` reply does not complete the local delivery signal. Local settlement remains a separate fact. | `DeliveryStopTest.acceptance_remoteProofDoesNotReleaseLocalSettlement` |
| DL-03 | Repeating stop for the same exact claim returns the same operation; an equal-valued but different claim is stale. | `DeliveryStopTest.acceptance_exactClaimHasOneOperation` |
| DL-04 | A late cleanup reply is applied before dispatcher drain completes, and does not create a second cleanup operation. | `DeliveryStopTest.lateCleanupProofSurvivesRetryExhaustionAndDrainWaitsForItsApplication` |
| DL-05 | A preemption reply is a victim progress result; capacity is committed only after all victim resolution facts are complete. | `DecodeAdmission` preemption and `RequestTerminalSettlementTest` preemption cases |
| DL-06 | Cancellation, physical settlement, and a late reply may race without releasing a replacement claim. | `DeliveryStopTest.staleStopRetiresWithoutWaitingForAnIrrelevantReplyOrReleasingTheReplacement` |
| DL-07 | Shutdown drains accepted stop work and callback application before closing request executors. | `DeliveryStopTest.physicalSettlementWakesStopWithoutWaitingForThePreemptionRpcDeadline` |
| DL-08 | A preemption caller deadline may complete its caller, but a late remote receipt is still applied once. | `DeliveryStopTest.acceptance_preemptionTimeoutDoesNotDiscardLateRemoteReceipt` |
| DL-09 | A preemption rejected after ordinary stop ownership is established returns `UNKNOWN` and cannot reuse another registration's reply. | `DeliveryStopTest.acceptance_rejectedPreemptionDoesNotReuseAnotherRegistrationReply` |

## Structural contracts

The implementation must satisfy these invariants:

1. Delivery facts enter one Scheduler business ingress.
2. A Context transition produces at most one follow-up action set.
3. Dispatcher never calls request close or releases Decode/Prefill capacity.
4. Context never starts an executor, performs RPC, or waits on a Future.
5. One delivery identity has at most one active stop operation.
6. No callback invokes Scheduler while holding the Context or operation monitor.
7. The existing RequestWorkQueue remains the serialization point for close and stop work.

## Concurrency matrix

The following interleavings must be covered before the refactor is accepted:

- cancel versus send-start;
- cancel versus physical settlement;
- remote cleanup reply versus replacement claim;
- preemption timeout versus late `REQUEST_CLEANED`;
- multiple victims with mixed `NOT_FOUND`, `UNKNOWN`, and cleaned replies;
- shutdown while a stop operation is parked on remote IO.

## Performance gate

Run the same 750P and 750D workload and JVM options before and after the
refactor. Acceptance requires all of the following against the clean baseline:

- no QPS regression greater than 3%;
- no p99 latency regression greater than 5%;
- no increase in allocation rate per request greater than 5%;
- no increase in GC pause time;
- no `RESOURCE_EXHAUSTED` caused by the refactor;
- zero active stop operations and zero delivery inflight after shutdown.

The performance gate is independent of unit-test success. A passing UT run
does not approve a performance regression.
