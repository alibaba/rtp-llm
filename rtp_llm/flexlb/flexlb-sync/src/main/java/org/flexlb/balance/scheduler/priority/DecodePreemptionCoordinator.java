package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.DecodeEndpoint;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BooleanSupplier;

import org.springframework.stereotype.Component;

/**
 * Executes one Engine-Cancel preemption transaction.
 *
 * <p>The scheduler supplies a pure plan and consumes one result.  This class
 * owns the two-phase protocol, token fencing and exactly-once child settlement:
 * Cancel ACCEPTED first, then an independent wait for typed original-Prefill
 * WorkerStatus CANCELED+8429.</p>
 */
@Component
public final class DecodePreemptionCoordinator {

    public enum ResultCode {
        COMMITTED,
        REPLAN_NOT_FOUND,
        CONFLICT,
        CONTROL_FAILED
    }

    public record ExecutionResult(ResultCode code,
                                  PreemptionAttempt attempt,
                                  String detail) {
        static ExecutionResult of(ResultCode code, PreemptionAttempt attempt, String detail) {
            return new ExecutionResult(code, attempt, detail);
        }
    }

    public record Request(DecodeEndpoint endpoint,
                          long snapshotVersion,
                          boolean requireVersionMatch,
                          long incomingRequestId,
                          long incomingKvTokens,
                          long incomingExpectedKvTokens,
                          int incomingPriority,
                          List<DecodeRequestSnapshot> victims,
                          long cancelAckTimeoutMs,
                          long cancelCompletionTimeoutMs,
                          BooleanSupplier admissionOpen,
                          String detail) {
        public Request {
            if (endpoint == null || victims == null || victims.isEmpty()) {
                throw new IllegalArgumentException("endpoint and victims are required");
            }
            if (victims.stream().anyMatch(victim -> !victim.phase().requiresEngineCancel())) {
                throw new IllegalArgumentException("coordinator accepts only Engine-Cancel victims");
            }
            if (admissionOpen == null) {
                throw new IllegalArgumentException("admission gate is required");
            }
        }
    }

    private final EngineCancelChannel cancelChannel;
    private final AtomicLong tokenSequence = new AtomicLong(1);

    public DecodePreemptionCoordinator(EngineCancelChannel cancelChannel) {
        this.cancelChannel = cancelChannel;
    }

    public CompletableFuture<ExecutionResult> execute(Request request,
                                                       InflightRegistrar registrar) {
        long token = nextToken();
        List<PreemptionAttempt.Victim> victims = new ArrayList<>(request.victims().size());
        for (DecodeRequestSnapshot victim : request.victims()) {
            EngineCancelChannel.CancelTarget target =
                    registrar.resolveCancelTarget(victim.requestId());
            if (target == null || !target.isRoutable()) {
                return CompletableFuture.completedFuture(ExecutionResult.of(
                        ResultCode.CONTROL_FAILED, null,
                        "cancel_owner_missing:" + victim.requestId()));
            }
            victims.add(new PreemptionAttempt.Victim(victim.requestId(), victim.priority(),
                    victim.kvTokens(), victim.phase(), target));
        }

        PreemptionAttempt attempt = new PreemptionAttempt(token,
                request.incomingRequestId(), request.snapshotVersion(),
                victims);

        List<Long> claimedInflight = new ArrayList<>(victims.size());
        for (PreemptionAttempt.Victim victim : victims) {
            if (!registrar.claimForPreemption(victim.requestId(), token, request.detail())) {
                for (Long claimed : claimedInflight) {
                    registrar.releasePreemptionClaim(claimed, token);
                }
                attempt.markAborted(false);
                return CompletableFuture.completedFuture(ExecutionResult.of(
                        ResultCode.CONFLICT, attempt, "victim_inflight_gone"));
            }
            claimedInflight.add(victim.requestId());
        }

        DecodeEndpoint.PreemptionBeginResult begin = request.endpoint().beginPriorityPreemption(
                token, victims.stream().map(PreemptionAttempt.Victim::requestId).toList(),
                request.incomingRequestId(), request.incomingKvTokens(),
                request.incomingExpectedKvTokens(), request.incomingPriority(),
                request.snapshotVersion(), request.requireVersionMatch());
        if (begin != DecodeEndpoint.PreemptionBeginResult.SUCCESS) {
            for (Long claimed : claimedInflight) {
                registrar.releasePreemptionClaim(claimed, token);
            }
            attempt.markAborted(false);
            ResultCode code = begin == DecodeEndpoint.PreemptionBeginResult.VERSION_MISMATCH
                    || begin == DecodeEndpoint.PreemptionBeginResult.VICTIM_GONE
                    || begin == DecodeEndpoint.PreemptionBeginResult.VICTIM_ALREADY_CLAIMED
                    ? ResultCode.CONFLICT : ResultCode.CONTROL_FAILED;
            return CompletableFuture.completedFuture(
                    ExecutionResult.of(code, attempt, "begin_" + begin.name().toLowerCase()));
        }
        attempt.claimAll();

        if (!request.endpoint().markPriorityCancelInFlight(token)) {
            request.endpoint().abortPriorityPreemption(token);
            for (Long claimed : claimedInflight) {
                registrar.releasePreemptionClaim(claimed, token);
            }
            attempt.markAborted(true);
            return CompletableFuture.completedFuture(ExecutionResult.of(
                    ResultCode.CONTROL_FAILED, attempt, "endpoint_cancel_linearization_failed"));
        }
        for (PreemptionAttempt.Victim victim : victims) {
            if (!registrar.markPreemptionCancelInFlight(victim.requestId(), token)) {
                request.endpoint().abortPriorityPreemption(token);
                for (Long claimed : claimedInflight) {
                    registrar.releasePreemptionClaim(claimed, token);
                }
                attempt.markAborted(true);
                return CompletableFuture.completedFuture(ExecutionResult.of(
                        ResultCode.CONTROL_FAILED, attempt,
                        "inflight_cancel_linearization_failed:" + victim.requestId()));
            }
        }
        if (!attempt.markCancelInFlight()) {
            request.endpoint().abortPriorityPreemption(token);
            for (Long claimed : claimedInflight) {
                registrar.releasePreemptionClaim(claimed, token);
            }
            attempt.markAborted(true);
            return CompletableFuture.completedFuture(ExecutionResult.of(
                    ResultCode.CONTROL_FAILED, attempt, "attempt_cancel_linearization_failed"));
        }

        // All state is CANCEL_IN_FLIGHT before the first RPC is invoked.
        List<CompletableFuture<EngineCancelChannel.CancelOutcome>> acknowledgements =
                new ArrayList<>(victims.size());
        for (PreemptionAttempt.Victim victim : victims) {
            acknowledgements.add(cancelChannel.cancel(victim.target(), victim.requestId(),
                    request.cancelAckTimeoutMs()).exceptionally(ignored ->
                            EngineCancelChannel.CancelOutcome.failed()));
        }

        return CompletableFuture.allOf(
                        acknowledgements.toArray(new CompletableFuture[0]))
                .thenCompose(ignored -> handleAcknowledgements(
                        request, registrar, attempt, acknowledgements));
    }

    private CompletableFuture<ExecutionResult> handleAcknowledgements(
            Request request,
            InflightRegistrar registrar,
            PreemptionAttempt attempt,
            List<CompletableFuture<EngineCancelChannel.CancelOutcome>> acknowledgements) {
        // A transport-unknown ACK is not a negative acknowledgement: the
        // Prefill may have installed the intent before the reply was lost.
        // Such a child therefore waits for the stronger typed CANCELED proof
        // exactly like an ACCEPTED child.
        List<PreemptionAttempt.Victim> completionCandidates = new ArrayList<>();
        // Parallel to completionCandidates: a successful ACK has already
        // locked the Cancel first-cause. A completion timeout may reject the
        // incoming request, but must not downgrade that victim to UNKNOWN and
        // replay a cached ordinary Decode terminal ahead of late typed 8429.
        List<Boolean> acceptedAcknowledgements = new ArrayList<>();
        boolean hasNotFound = false;
        int tombstonedSettlements = 0;
        boolean tombstoneSettlementFailed = false;

        List<PreemptionAttempt.Victim> victims = attempt.victims();
        for (int i = 0; i < victims.size(); i++) {
            PreemptionAttempt.Victim victim = victims.get(i);
            EngineCancelChannel.CancelOutcome outcome = acknowledgements.get(i).join();
            switch (outcome.ack()) {
                case ACCEPTED -> {
                    boolean transitioned = attempt.recordAccepted(victim.requestId())
                            && request.endpoint().markPriorityCancelAccepted(
                                    attempt.token(), victim.requestId())
                            && registrar.markPreemptionCancelAccepted(
                                    victim.requestId(), attempt.token());
                    if (transitioned) {
                        completionCandidates.add(victim);
                        acceptedAcknowledgements.add(true);
                    } else {
                        markUnknown(request, registrar, attempt, victim.requestId());
                        completionCandidates.add(victim);
                        acceptedAcknowledgements.add(false);
                    }
                }
                case NOT_FOUND -> {
                    hasNotFound = true;
                    attempt.recordNotFound(victim.requestId());
                    request.endpoint().markPriorityCancelNotFound(
                            attempt.token(), victim.requestId());
                    registrar.markPreemptionNotFound(victim.requestId(), attempt.token());
                }
                case TOMBSTONED -> {
                    // Unlike NOT_FOUND, TOMBSTONED atomically proves absence
                    // and prevents every racing late Enqueue. It is therefore
                    // a terminal victim proof and contributes freed capacity
                    // to this same transaction.
                    if (settleTombstoned(
                            request, registrar, attempt, victim)) {
                        tombstonedSettlements++;
                    } else {
                        tombstoneSettlementFailed = true;
                    }
                }
                case FAILED, UNSUPPORTED -> {
                    markUnknown(request, registrar, attempt, victim.requestId());
                    completionCandidates.add(victim);
                    acceptedAcknowledgements.add(false);
                }
            }
        }

        if (completionCandidates.isEmpty() && tombstonedSettlements == 0) {
            request.endpoint().abortPriorityPreemption(attempt.token());
            boolean cleanSingleNotFound = hasNotFound && victims.size() == 1;
            attempt.markAborted(!cleanSingleNotFound);
            return CompletableFuture.completedFuture(ExecutionResult.of(
                    cleanSingleNotFound
                            ? ResultCode.REPLAN_NOT_FOUND : ResultCode.CONTROL_FAILED,
                    attempt, cleanSingleNotFound
                            ? "cancel_not_found" : "cancel_ack_invalid"));
        }

        attempt.beginCanceledWait();
        // The completion budget begins only after the ACK phase has ended; a
        // 40ms ACK followed by a 100ms cleanup therefore gets the full cleanup
        // window rather than sharing one 50ms deadline.
        long completionDeadlineNanos = System.nanoTime()
                + TimeUnit.MILLISECONDS.toNanos(
                        Math.max(1, request.cancelCompletionTimeoutMs()));
        List<CompletableFuture<Boolean>> boundedSettlements =
                new ArrayList<>(completionCandidates.size());
        for (PreemptionAttempt.Victim victim : completionCandidates) {
            CompletableFuture<InflightRegistrar.PriorityCanceledObservation> signal =
                    registrar.priorityCanceledSignal(victim.requestId(), attempt.token());
            CompletableFuture<Boolean> settlement = signal.thenApply(observation ->
                    settleTypedCanceled(request, registrar, attempt, victim, observation));
            // Keep the unbounded settlement continuation alive after the
            // admission attempt times out.  A late typed CANCELED must still
            // release the victim exactly once even though the incoming has
            // already been rejected and its provisional reservation released.
            boundedSettlements.add(withDeadline(settlement, completionDeadlineNanos)
                    .exceptionally(ignored -> false));
        }

        final boolean ackNotFound = hasNotFound;
        final int alreadySettled = tombstonedSettlements;
        final boolean tombstoneFailed = tombstoneSettlementFailed;
        return CompletableFuture.allOf(boundedSettlements.toArray(new CompletableFuture[0]))
                .thenApply(ignored -> {
                    boolean allCanceled = !tombstoneFailed
                            && alreadySettled + boundedSettlements.size() == victims.size()
                            && boundedSettlements.stream().allMatch(CompletableFuture::join);
                    if (allCanceled && !ackNotFound && request.admissionOpen().getAsBoolean()
                            && request.endpoint().commitPriorityPreemption(attempt.token())
                            && attempt.markCommitted()) {
                        return ExecutionResult.of(ResultCode.COMMITTED, attempt, "committed");
                    }
                    for (int i = 0; i < boundedSettlements.size(); i++) {
                        if (!boundedSettlements.get(i).join()
                                && !acceptedAcknowledgements.get(i)) {
                            markUnknown(request, registrar, attempt,
                                    completionCandidates.get(i).requestId());
                        }
                    }
                    request.endpoint().abortPriorityPreemption(attempt.token());
                    // REPLAN_NOT_FOUND is safe only for the single-victim,
                    // zero-side-effect branch above. Once any sibling entered
                    // the completion path, a NOT_FOUND sibling makes the
                    // transaction partial: the canceled sibling cannot be
                    // resurrected by a fresh scheduling attempt.
                    boolean controlFailed = !allCanceled || ackNotFound;
                    attempt.markAborted(controlFailed);
                    return ExecutionResult.of(
                            ResultCode.CONTROL_FAILED,
                            attempt,
                            ackNotFound
                                    ? "cancel_partial_not_found" : "cancel_completion_unknown");
                });
    }

    private static boolean settleTypedCanceled(
            Request request,
            InflightRegistrar registrar,
            PreemptionAttempt attempt,
            PreemptionAttempt.Victim victim,
            InflightRegistrar.PriorityCanceledObservation observation) {
        if (observation == null || observation.requestId() != victim.requestId()
                || observation.errorCode() != 8429) {
            return false;
        }
        // Endpoint accounting is the resource-owning CAS.  The attempt ledger
        // and scheduler lifecycle are token-fenced followers; the signal
        // future completes once, so this chain cannot double-decrement.
        boolean endpointSettled = request.endpoint().settlePriorityCanceled(
                attempt.token(), victim.requestId());
        boolean attemptSettled = endpointSettled
                && attempt.recordCanceled(victim.requestId());
        boolean inflightSettled = attemptSettled
                && registrar.finishPreemptedById(
                        victim.requestId(), attempt.token(), request.detail());
        return endpointSettled && attemptSettled && inflightSettled;
    }

    private static boolean settleTombstoned(
            Request request,
            InflightRegistrar registrar,
            PreemptionAttempt attempt,
            PreemptionAttempt.Victim victim) {
        // Endpoint accounting remains the resource-owning CAS. The remaining
        // two transitions are exact-token followers, just like typed CANCELED,
        // but no WorkerStatus future is required for this stronger proof.
        boolean endpointSettled = request.endpoint().settlePriorityTombstoned(
                attempt.token(), victim.requestId());
        boolean inflightSettled = endpointSettled
                && registrar.finishTombstonedById(
                        victim.requestId(), attempt.token(), request.detail());
        boolean attemptSettled = inflightSettled
                && attempt.recordTombstoned(victim.requestId());
        return endpointSettled && inflightSettled && attemptSettled;
    }

    private static void markUnknown(Request request,
                                    InflightRegistrar registrar,
                                    PreemptionAttempt attempt,
                                    long requestId) {
        attempt.recordUnknown(requestId);
        request.endpoint().markPriorityCancelUnknown(attempt.token(), requestId);
        registrar.markPreemptionUnknown(requestId, attempt.token());
    }

    private static <T> CompletableFuture<T> withDeadline(
            CompletableFuture<T> source, long deadlineNanos) {
        long remainingNanos = Math.max(1, deadlineNanos - System.nanoTime());
        CompletableFuture<T> timeout = new CompletableFuture<>();
        CompletableFuture.delayedExecutor(remainingNanos, TimeUnit.NANOSECONDS)
                .execute(() -> timeout.completeExceptionally(
                        new TimeoutException("priority CANCELED deadline exceeded")));
        return source.applyToEither(timeout, value -> value);
    }

    private long nextToken() {
        long token = tokenSequence.getAndIncrement();
        if (token <= 0) {
            throw new IllegalStateException("preemption attempt token exhausted");
        }
        return token;
    }
}
