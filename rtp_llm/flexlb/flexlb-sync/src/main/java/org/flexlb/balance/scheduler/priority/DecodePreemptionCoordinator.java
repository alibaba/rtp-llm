package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint.EndpointRetiredException;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
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
                          long incomingDeadlineMs,
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
        try {
            for (DecodeRequestSnapshot victim : request.victims()) {
                EngineCancelChannel.CancelTarget target =
                        registrar.resolveCancelTarget(victim.requestId());
                if (target == null || !target.isRoutable()
                        || !target.isGenerationBound()) {
                    return CompletableFuture.completedFuture(ExecutionResult.of(
                            ResultCode.CONTROL_FAILED, null,
                            "cancel_owner_missing:" + victim.requestId()));
                }
                victims.add(new PreemptionAttempt.Victim(victim.requestId(), victim.priority(),
                        victim.kvTokens(), victim.phase(), target));
            }
        } catch (RuntimeException targetFailure) {
            return CompletableFuture.completedFuture(ExecutionResult.of(
                    ResultCode.CONTROL_FAILED, null, "cancel_owner_resolution_failed"));
        }

        PreemptionAttempt attempt = new PreemptionAttempt(token,
                request.incomingRequestId(), request.snapshotVersion(),
                victims);

        try {
            for (PreemptionAttempt.Victim victim : victims) {
                if (!registrar.claimForPreemption(
                        victim.requestId(), token, request.detail())) {
                    releaseRegistrarClaims(registrar, victims, token);
                    attempt.markAborted(false);
                    return CompletableFuture.completedFuture(ExecutionResult.of(
                            ResultCode.CONFLICT, attempt, "victim_inflight_gone"));
                }
            }
        } catch (RuntimeException claimFailure) {
            releaseRegistrarClaims(registrar, victims, token);
            attempt.markAborted(true);
            return CompletableFuture.completedFuture(ExecutionResult.of(
                    ResultCode.CONTROL_FAILED, attempt, "victim_claim_failed"));
        }

        DecodeEndpoint.PreemptionBeginResult begin;
        try {
            begin = request.endpoint().beginPriorityPreemption(
                    token, victims.stream().map(PreemptionAttempt.Victim::requestId).toList(),
                    request.incomingRequestId(), request.incomingKvTokens(),
                    request.incomingExpectedKvTokens(), request.incomingPriority(),
                    request.incomingDeadlineMs(), request.snapshotVersion(),
                    request.requireVersionMatch());
        } catch (RuntimeException beginFailure) {
            try {
                request.endpoint().releaseIfHeld(request.incomingRequestId());
            } catch (RuntimeException ignored) {
                // Continue clearing registrar ownership.
            }
            releaseRegistrarClaims(registrar, victims, token);
            attempt.markAborted(true);
            return CompletableFuture.completedFuture(ExecutionResult.of(
                    ResultCode.CONTROL_FAILED, attempt, "begin_failed"));
        }
        if (begin != DecodeEndpoint.PreemptionBeginResult.SUCCESS) {
            releaseRegistrarClaims(registrar, victims, token);
            attempt.markAborted(false);
            ResultCode code = begin == DecodeEndpoint.PreemptionBeginResult.VERSION_MISMATCH
                    || begin == DecodeEndpoint.PreemptionBeginResult.VICTIM_GONE
                    || begin == DecodeEndpoint.PreemptionBeginResult.VICTIM_ALREADY_CLAIMED
                    ? ResultCode.CONFLICT : ResultCode.CONTROL_FAILED;
            return CompletableFuture.completedFuture(
                    ExecutionResult.of(code, attempt, "begin_" + begin.name().toLowerCase()));
        }
        if (!attempt.claimAll()) {
            return abortBeforeRpc(request, registrar, attempt, victims,
                    "attempt_claim_linearization_failed");
        }

        try {
            if (!request.endpoint().markPriorityCancelInFlight(token)) {
                return abortBeforeRpc(request, registrar, attempt, victims,
                        "endpoint_cancel_linearization_failed");
            }
            for (PreemptionAttempt.Victim victim : victims) {
                if (!registrar.markPreemptionCancelInFlight(victim.requestId(), token)) {
                    return abortBeforeRpc(request, registrar, attempt, victims,
                            "inflight_cancel_linearization_failed:" + victim.requestId());
                }
            }
            if (!attempt.markCancelInFlight()) {
                return abortBeforeRpc(request, registrar, attempt, victims,
                        "attempt_cancel_linearization_failed");
            }
        } catch (RuntimeException linearizationFailure) {
            return abortBeforeRpc(request, registrar, attempt, victims,
                    "cancel_linearization_failed");
        }

        // All state is CANCEL_IN_FLIGHT before the first RPC is invoked.
        List<CompletableFuture<EngineCancelChannel.CancelOutcome>> acknowledgements;
        try {
            acknowledgements = initiateCancelRequests(
                    request.endpoint(), victims, request.cancelAckTimeoutMs());
        } catch (EndpointRetiredException retired) {
            return abortBeforeRpc(request, registrar, attempt, victims,
                    "cancel_generation_retired");
        }

        CompletableFuture<ExecutionResult> execution = CompletableFuture.allOf(
                        acknowledgements.toArray(new CompletableFuture[0]))
                .thenCompose(ignored -> handleAcknowledgements(
                        request, registrar, attempt, acknowledgements));
        return execution.handle((result, failure) -> {
            if (failure == null && result != null) {
                return result;
            }
            abortAfterRpcFailure(request, registrar, attempt);
            return ExecutionResult.of(
                    ResultCode.CONTROL_FAILED, attempt, "cancel_transaction_failed");
        });
    }

    /**
     * Invoke every Cancel while the exact Decode owner and all original
     * Prefill generations hold a shared dispatch lease. If any generation is
     * already retired, the supplier is never entered and no victim RPC is
     * invoked.
     */
    private List<CompletableFuture<EngineCancelChannel.CancelOutcome>>
            initiateCancelRequests(DecodeEndpoint decodeGeneration,
                                   List<PreemptionAttempt.Victim> victims,
                                   long timeoutMs) {
        PrefillEndpoint owner = victims.getFirst().target().prefillGeneration();
        List<WorkerEndpoint> generations = new ArrayList<>(victims.size() + 1);
        generations.add(decodeGeneration);
        for (PreemptionAttempt.Victim victim : victims) {
            generations.add(victim.target().prefillGeneration());
        }
        return owner.initiateGenerationDispatch(generations, () -> {
            List<CompletableFuture<EngineCancelChannel.CancelOutcome>> acknowledgements =
                    new ArrayList<>(victims.size());
            for (PreemptionAttempt.Victim victim : victims) {
                acknowledgements.add(requestCancel(victim, timeoutMs));
            }
            return acknowledgements;
        });
    }

    private CompletableFuture<EngineCancelChannel.CancelOutcome> requestCancel(
            PreemptionAttempt.Victim victim, long timeoutMs) {
        try {
            CompletableFuture<EngineCancelChannel.CancelOutcome> acknowledgement =
                    cancelChannel.cancel(victim.target(), victim.requestId(), timeoutMs);
            if (acknowledgement == null) {
                return CompletableFuture.completedFuture(
                        EngineCancelChannel.CancelOutcome.failed());
            }
            return acknowledgement.handle((outcome, failure) ->
                    failure == null && outcome != null
                            ? outcome : EngineCancelChannel.CancelOutcome.failed());
        } catch (RuntimeException synchronousFailure) {
            return CompletableFuture.completedFuture(
                    EngineCancelChannel.CancelOutcome.failed());
        }
    }

    private static CompletableFuture<ExecutionResult> abortBeforeRpc(
            Request request,
            InflightRegistrar registrar,
            PreemptionAttempt attempt,
            List<PreemptionAttempt.Victim> victims,
            String detail) {
        try {
            request.endpoint().abortPriorityPreemption(attempt.token());
        } catch (RuntimeException ignored) {
            // Registrar claims are an independent ownership domain.
        }
        releaseRegistrarClaims(registrar, victims, attempt.token());
        attempt.markAborted(true);
        return CompletableFuture.completedFuture(ExecutionResult.of(
                ResultCode.CONTROL_FAILED, attempt, detail));
    }

    private static void releaseRegistrarClaims(
            InflightRegistrar registrar,
            List<PreemptionAttempt.Victim> victims,
            long token) {
        for (PreemptionAttempt.Victim victim : victims) {
            try {
                registrar.releasePreemptionClaim(victim.requestId(), token);
            } catch (RuntimeException ignored) {
                // Continue releasing independent victim claims.
            }
        }
    }

    private static void abortAfterRpcFailure(
            Request request,
            InflightRegistrar registrar,
            PreemptionAttempt attempt) {
        for (PreemptionAttempt.Victim victim : attempt.victims()) {
            try {
                PreemptionAttempt.VictimState state =
                        attempt.victimState(victim.requestId());
                if (state == PreemptionAttempt.VictimState.CANCEL_IN_FLIGHT
                        || state == PreemptionAttempt.VictimState.CANCEL_REQUESTED) {
                    markUnknown(request, registrar, attempt, victim.requestId());
                }
            } catch (RuntimeException ignored) {
                // Preserve the remaining victims' reconciliation fences.
            }
        }
        try {
            request.endpoint().abortPriorityPreemption(attempt.token());
        } catch (RuntimeException ignored) {
            // The result is still control-failed; preserve victim fences.
        }
        attempt.markAborted(true);
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
                    // The request was absent and is now fenced against a
                    // racing late Enqueue. For preemption planning this is a
                    // stronger form of NOT_FOUND and requires a replan.
                    hasNotFound = true;
                    attempt.recordNotFound(victim.requestId());
                    request.endpoint().markPriorityCancelNotFound(
                            attempt.token(), victim.requestId());
                    registrar.markPreemptionNotFound(victim.requestId(), attempt.token());
                }
                case FAILED, UNSUPPORTED -> {
                    markUnknown(request, registrar, attempt, victim.requestId());
                    completionCandidates.add(victim);
                    acceptedAcknowledgements.add(false);
                }
            }
        }

        if (completionCandidates.isEmpty()) {
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
        return CompletableFuture.allOf(boundedSettlements.toArray(new CompletableFuture[0]))
                .thenApply(ignored -> {
                    boolean allCanceled = boundedSettlements.stream()
                            .allMatch(future -> future.join());
                    if (allCanceled && !ackNotFound && request.admissionOpen().getAsBoolean()
                            && request.endpoint().commitPriorityPreemption(attempt.token())
                            && attempt.markCommitted()) {
                        return ExecutionResult.of(ResultCode.COMMITTED, attempt, "committed");
                    }
                    for (int i = 0; i < boundedSettlements.size(); i++) {
                        if (!boundedSettlements.get(i).join()) {
                            long requestId = completionCandidates.get(i).requestId();
                            if (acceptedAcknowledgements.get(i)) {
                                markAcceptedCompletionTimedOut(
                                        registrar, attempt.token(), requestId);
                            } else {
                                markUnknown(request, registrar, attempt, requestId);
                            }
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

    private static void markAcceptedCompletionTimedOut(
            InflightRegistrar registrar, long attemptToken, long requestId) {
        try {
            registrar.markPreemptionCompletionTimedOut(requestId, attemptToken);
        } catch (RuntimeException ignored) {
            // Preserve the accepted first-cause and its accounting fence. A
            // telemetry/control callback failure must never downgrade it to
            // transport UNKNOWN or release the victim locally.
        }
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

    private static void markUnknown(Request request,
                                    InflightRegistrar registrar,
                                    PreemptionAttempt attempt,
                                    long requestId) {
        if (!attempt.recordUnknown(requestId)) {
            return;
        }
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
