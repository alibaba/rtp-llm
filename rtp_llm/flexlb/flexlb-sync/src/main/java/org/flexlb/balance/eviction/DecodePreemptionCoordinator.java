package org.flexlb.balance.eviction;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.balance.preemption.PreemptionClaim;
import org.flexlb.balance.preemption.PreemptionLifecyclePort;
import org.flexlb.balance.preemption.VictimTerminal;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
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
 * owns the two-phase protocol, token fencing and exactly-once child settlement.
 * Engine acknowledgement is only control evidence; the canonical victim
 * terminal transaction may complete before or after that acknowledgement.</p>
 */
@Component
public final class DecodePreemptionCoordinator {

    public enum ResultCode {
        COMMITTED,
        NOT_FOUND_UNRESOLVED,
        CONFLICT,
        CONTROL_FAILED
    }

    public record ExecutionResult(ResultCode code, String detail) {
        static ExecutionResult of(ResultCode code, String detail) {
            return new ExecutionResult(code, detail);
        }
    }

    public record Request(DecodeEndpoint endpoint,
                          long incomingRequestId,
                          long incomingKvTokens,
                          long incomingExpectedKvTokens,
                          int incomingPriority,
                          DecodeEndpoint.AdmissionCapacity capacity,
                          List<DecodeRequestSnapshot> victims,
                          long cancelAckTimeoutMs,
                          long cancelCompletionTimeoutMs,
                          BooleanSupplier admissionOpen,
                          String detail) {
        public Request {
            if (endpoint == null || victims == null || victims.isEmpty()) {
                throw new IllegalArgumentException("endpoint and victims are required");
            }
            if (incomingRequestId <= 0L) {
                throw new IllegalArgumentException(
                        "incoming request id must be positive");
            }
            if (capacity == null) {
                throw new IllegalArgumentException("capacity policy is required");
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
    private final PreemptionLifecyclePort lifecycle;
    private final AtomicLong tokenSequence = new AtomicLong(1);

    public DecodePreemptionCoordinator(
            EngineCancelChannel cancelChannel,
            PreemptionLifecyclePort lifecycle) {
        this.cancelChannel = Objects.requireNonNull(
                cancelChannel, "cancelChannel");
        this.lifecycle = Objects.requireNonNull(
                lifecycle, "lifecycle");
    }

    public CompletableFuture<ExecutionResult> execute(Request request) {
        Objects.requireNonNull(request, "request");
        long token = nextToken();
        List<PreemptionAttempt.Victim> victims = new ArrayList<>(request.victims().size());
        List<DecodeEndpoint.ReservationHandle> victimReservations =
                new ArrayList<>(request.victims().size());
        long endpointGenerationId =
                request.endpoint().getStatus().getGenerationId();
        for (DecodeRequestSnapshot victim : request.victims()) {
            CancelTarget target =
                    lifecycle.resolveCancelTarget(
                            victim.requestId(), victim.reservationToken());
            if (target == null || !target.isRoutable()) {
                return CompletableFuture.completedFuture(ExecutionResult.of(
                        ResultCode.CONTROL_FAILED,
                        "cancel_owner_missing:" + victim.requestId()));
            }
            victims.add(new PreemptionAttempt.Victim(victim.requestId(), victim.priority(),
                    victim.kvTokens(), victim.phase(), victim.reservationToken(), target));
            victimReservations.add(new DecodeEndpoint.ReservationHandle(
                    endpointGenerationId,
                    victim.requestId(),
                    victim.reservationToken()));
        }

        PreemptionAttempt attempt = new PreemptionAttempt(token, victims);
        AttemptCapability capability = new AttemptCapability(
                request, lifecycle, attempt);
        try {
            for (PreemptionAttempt.Victim victim : victims) {
                PreemptionClaim claim =
                        lifecycle.claimForPreemption(
                                victim.requestId(), victim.reservationToken(),
                                token, request.detail());
                if (claim == null) {
                    return CompletableFuture.completedFuture(capability.abort(
                            ResultCode.CONFLICT, "victim_inflight_gone"));
                }
                ClaimedVictim owned = capability.add(victim, claim);
                if (claim.requestId() != victim.requestId()
                        || claim.attemptToken() != token) {
                    return CompletableFuture.completedFuture(capability.abort(
                            ResultCode.CONTROL_FAILED,
                            "lifecycle_returned_mismatched_claim:" + owned.requestId()));
                }
            }

            DecodeEndpoint.PreemptionBeginResult begin =
                    request.endpoint().beginPriorityPreemption(
                            token, victimReservations,
                            request.incomingRequestId(), request.incomingKvTokens(),
                            request.incomingExpectedKvTokens(), request.incomingPriority(),
                            request.capacity());
            if (begin != DecodeEndpoint.PreemptionBeginResult.SUCCESS) {
                ResultCode code = begin
                        == DecodeEndpoint.PreemptionBeginResult.ENDPOINT_RETIRED
                        ? ResultCode.CONTROL_FAILED : ResultCode.CONFLICT;
                return CompletableFuture.completedFuture(capability.abort(
                        code, "begin_" + begin.name().toLowerCase()));
            }
            capability.endpointBegun();
            if (!attempt.claimAll()) {
                return CompletableFuture.completedFuture(capability.abort(
                        ResultCode.CONTROL_FAILED,
                        "attempt_claim_linearization_failed"));
            }

            if (!request.endpoint().markPriorityCancelInFlight(token)) {
                return CompletableFuture.completedFuture(capability.abort(
                        ResultCode.CONTROL_FAILED,
                        "endpoint_cancel_linearization_failed"));
            }
            for (ClaimedVictim owned : capability.claims()) {
                if (!lifecycle.markPreemptionCancelInFlight(owned.claim())) {
                    return CompletableFuture.completedFuture(capability.abort(
                            ResultCode.CONTROL_FAILED,
                            "inflight_cancel_linearization_failed:"
                                    + owned.requestId()));
                }
            }
            if (!attempt.markCancelInFlight()) {
                return CompletableFuture.completedFuture(capability.abort(
                        ResultCode.CONTROL_FAILED,
                        "attempt_cancel_linearization_failed"));
            }

            // Capture every terminal capability before the first outbound
            // side effect. The exact claim remains the only lookup key.
            List<TerminalSettlement> terminalSettlements =
                    new ArrayList<>(victims.size());
            for (ClaimedVictim owned : capability.claims()) {
                CompletableFuture<Boolean> completion = owned.claim().terminal()
                        .handle((terminal, failure) -> failure == null
                                && capability.recordTerminal(owned, terminal));
                terminalSettlements.add(
                        new TerminalSettlement(owned, completion));
            }

            List<CompletableFuture<EngineCancelChannel.CancelOutcome>>
                    acknowledgements = new ArrayList<>(victims.size());
            for (ClaimedVictim owned : capability.claims()) {
                if (capability.outboundStarted(owned)) {
                    acknowledgements.add(cancel(
                            owned.victim(), request.cancelAckTimeoutMs()));
                } else {
                    acknowledgements.add(CompletableFuture.completedFuture(
                            EngineCancelChannel.CancelOutcome.failed()));
                }
            }

            CompletableFuture<ExecutionResult> protocol =
                    CompletableFuture.allOf(
                            acknowledgements.toArray(new CompletableFuture[0]))
                    .thenCompose(ignored -> handleAcknowledgements(
                            capability, acknowledgements, terminalSettlements));
            return protocol.handle((result, failure) -> {
                if (failure != null) {
                    return capability.abort(
                            ResultCode.CONTROL_FAILED,
                            "coordinator_continuation_failed:"
                                    + failureDetail(failure));
                }
                return result == null
                        ? capability.abort(
                                ResultCode.CONTROL_FAILED,
                                "coordinator_returned_null_result")
                        : result;
            });
        } catch (RuntimeException | Error failure) {
            return CompletableFuture.completedFuture(capability.abort(
                    ResultCode.CONTROL_FAILED,
                    "coordinator_setup_failed:" + failureDetail(failure)));
        }
    }

    private CompletableFuture<ExecutionResult> handleAcknowledgements(
            AttemptCapability capability,
            List<CompletableFuture<EngineCancelChannel.CancelOutcome>> acknowledgements,
            List<TerminalSettlement> terminalSettlements) {
        Request request = capability.request();
        PreemptionLifecyclePort lifecycle = capability.lifecycle();
        PreemptionAttempt attempt = capability.attempt();
        // A transport-unknown ACK is not a negative acknowledgement: the
        // Prefill may have installed the intent before the reply was lost.
        // Such a child therefore waits for the canonical victim terminal
        // transaction exactly like an ACCEPTED child.
        List<PendingTerminal> pendingTerminals = new ArrayList<>();
        boolean hasNotFound = false;

        List<PreemptionAttempt.Victim> victims = attempt.victims();
        for (int i = 0; i < victims.size(); i++) {
            PreemptionAttempt.Victim victim = victims.get(i);
            TerminalSettlement terminal = terminalSettlements.get(i);
            if (attempt.isTerminal(victim.requestId())) {
                continue;
            }
            EngineCancelChannel.CancelOutcome outcome = acknowledgements.get(i).join();
            switch (outcome.ack()) {
                case ACCEPTED -> {
                    boolean transitioned = attempt.recordAccepted(victim.requestId())
                            && request.endpoint().markPriorityCancelAccepted(
                                    attempt.token(), victim.requestId())
                            && lifecycle.markPreemptionCancelAccepted(
                                    terminal.owned().claim());
                    if (transitioned) {
                        capability.transferred(terminal.owned());
                        pendingTerminals.add(
                                new PendingTerminal(terminal, true));
                    } else {
                        capability.transferUnknown(terminal.owned());
                        pendingTerminals.add(
                                new PendingTerminal(terminal, false));
                    }
                }
                case NOT_FOUND -> {
                    attempt.recordNotFound(victim.requestId());
                    request.endpoint().markPriorityCancelNotFound(
                            attempt.token(), victim.requestId());
                    lifecycle.markPreemptionNotFound(
                            terminal.owned().claim());
                    capability.transferred(terminal.owned());
                    if (!attempt.isTerminal(victim.requestId())) {
                        hasNotFound = true;
                    }
                }
                case TOMBSTONED -> {
                    // Unlike NOT_FOUND, TOMBSTONED atomically proves absence
                    // and prevents every racing late Enqueue. It is therefore
                    // a terminal victim proof and contributes freed capacity
                    // to this same transaction.
                    settleTombstoned(capability, terminal);
                }
                case FAILED, UNSUPPORTED -> {
                    capability.transferUnknown(terminal.owned());
                    pendingTerminals.add(
                            new PendingTerminal(terminal, false));
                }
            }
        }

        attempt.beginTerminalWait();
        if (pendingTerminals.isEmpty()) {
            return CompletableFuture.completedFuture(
                    capability.finish(hasNotFound));
        }
        // The completion budget begins only after the ACK phase has ended; a
        // 40ms ACK followed by a 100ms cleanup therefore gets the full cleanup
        // window rather than sharing one 50ms deadline.
        long completionDeadlineNanos = System.nanoTime()
                + TimeUnit.MILLISECONDS.toNanos(
                        Math.max(1, request.cancelCompletionTimeoutMs()));
        List<CompletableFuture<Boolean>> boundedSettlements =
                new ArrayList<>(pendingTerminals.size());
        for (PendingTerminal pending : pendingTerminals) {
            // The unbounded continuation was installed before the first RPC.
            // Timing out this admission wait cannot cancel or lose a later
            // canonical victim terminal.
            boundedSettlements.add(withDeadline(
                    pending.settlement().completion(), completionDeadlineNanos)
                    .exceptionally(ignored -> false));
        }

        final boolean ackNotFound = hasNotFound;
        return CompletableFuture.allOf(boundedSettlements.toArray(new CompletableFuture[0]))
                .thenApply(ignored -> {
                    for (int i = 0; i < boundedSettlements.size(); i++) {
                        if (!boundedSettlements.get(i).join()
                                && !pendingTerminals.get(i).acceptedAcknowledgement()) {
                            capability.transferUnknown(
                                    pendingTerminals.get(i).settlement()
                                            .owned());
                        }
                    }
                    return capability.finish(ackNotFound);
                });
    }

    private static boolean settleTombstoned(
            AttemptCapability capability,
            TerminalSettlement terminal) {
        Request request = capability.request();
        PreemptionLifecyclePort lifecycle = capability.lifecycle();
        PreemptionAttempt attempt = capability.attempt();
        ClaimedVictim owned = terminal.owned();
        PreemptionAttempt.Victim victim = owned.victim();
        // Endpoint accounting remains the resource-owning CAS. The remaining
        // transitions are exact-token followers, but no WorkerStatus future
        // is required for this stronger proof.
        boolean endpointSettled = request.endpoint().settlePriorityTombstoned(
                attempt.token(), reservation(request.endpoint(), victim));
        boolean inflightSettled = endpointSettled
                && lifecycle.finishTombstoned(
                        owned.claim(), request.detail());
        boolean attemptSettled = inflightSettled
                && attempt.recordTerminal(victim.requestId());
        if (attemptSettled) {
            capability.terminal(owned);
        }
        return endpointSettled && inflightSettled && attemptSettled;
    }

    private static DecodeEndpoint.ReservationHandle reservation(
            DecodeEndpoint endpoint,
            PreemptionAttempt.Victim victim) {
        return new DecodeEndpoint.ReservationHandle(
                endpoint.getStatus().getGenerationId(),
                victim.requestId(),
                victim.reservationToken());
    }

    private static <T> CompletableFuture<T> withDeadline(
            CompletableFuture<T> source, long deadlineNanos) {
        long remainingNanos = Math.max(1, deadlineNanos - System.nanoTime());
        CompletableFuture<T> timeout = new CompletableFuture<>();
        CompletableFuture.delayedExecutor(remainingNanos, TimeUnit.NANOSECONDS)
                .execute(() -> timeout.completeExceptionally(
                        new TimeoutException("victim terminal deadline exceeded")));
        return source.applyToEither(timeout, value -> value);
    }

    private CompletableFuture<EngineCancelChannel.CancelOutcome> cancel(
            PreemptionAttempt.Victim victim,
            long timeoutMs) {
        try {
            CompletableFuture<EngineCancelChannel.CancelOutcome> stage =
                    cancelChannel.cancel(
                            victim.target(), victim.requestId(), timeoutMs);
            if (stage == null) {
                return CompletableFuture.completedFuture(
                        EngineCancelChannel.CancelOutcome.failed());
            }
            return stage.handle((outcome, failure) -> failure == null
                            && outcome != null
                    ? outcome : EngineCancelChannel.CancelOutcome.failed());
        } catch (RuntimeException | Error failure) {
            // The capability marked this victim OUTBOUND before invocation,
            // so close conservatively transfers its exact claims to UNKNOWN.
            return CompletableFuture.completedFuture(
                    EngineCancelChannel.CancelOutcome.failed());
        }
    }

    private static String failureDetail(Throwable failure) {
        Throwable current = failure;
        while (current.getCause() != null && current.getCause() != current) {
            current = current.getCause();
        }
        String message = current.getMessage();
        return current.getClass().getSimpleName()
                + (message == null || message.isBlank() ? "" : ":" + message);
    }

    private record TerminalSettlement(
            ClaimedVictim owned,
            CompletableFuture<Boolean> completion) {
    }

    private record PendingTerminal(
            TerminalSettlement settlement,
            boolean acceptedAcknowledgement) {
    }

    private enum ClaimDisposition {
        RELEASABLE,
        OUTBOUND,
        TRANSFERRED,
        TERMINAL
    }

    /** Exact opaque slot claim paired with its immutable endpoint victim. */
    private static final class ClaimedVictim {
        private final PreemptionAttempt.Victim victim;
        private final PreemptionClaim claim;
        private volatile ClaimDisposition disposition =
                ClaimDisposition.RELEASABLE;

        private ClaimedVictim(
                PreemptionAttempt.Victim victim,
                PreemptionClaim claim) {
            this.victim = victim;
            this.claim = claim;
        }

        private PreemptionAttempt.Victim victim() {
            return victim;
        }

        private PreemptionClaim claim() {
            return claim;
        }

        private long requestId() {
            return victim.requestId();
        }
    }

    /**
     * The one owner of endpoint admission plus every exact RequestSlot claim.
     * A non-committed close is total: uncertain outbound claims transfer to
     * reconciliation, the incoming endpoint reservation aborts, and only
     * claims which never crossed an outbound boundary are released.
     */
    private static final class AttemptCapability implements AutoCloseable {
        private final Request request;
        private final PreemptionLifecyclePort lifecycle;
        private final PreemptionAttempt attempt;
        private final List<ClaimedVictim> claims = new ArrayList<>();
        private boolean endpointBegun;
        private boolean committed;
        private boolean closed;
        private String cleanupFailure;

        private AttemptCapability(
                Request request,
                PreemptionLifecyclePort lifecycle,
                PreemptionAttempt attempt) {
            this.request = request;
            this.lifecycle = lifecycle;
            this.attempt = attempt;
        }

        private Request request() {
            return request;
        }

        private PreemptionLifecyclePort lifecycle() {
            return lifecycle;
        }

        private PreemptionAttempt attempt() {
            return attempt;
        }

        private ClaimedVictim add(
                PreemptionAttempt.Victim victim,
                PreemptionClaim claim) {
            ClaimedVictim owned = new ClaimedVictim(victim, claim);
            claims.add(owned);
            return owned;
        }

        private List<ClaimedVictim> claims() {
            return List.copyOf(claims);
        }

        private void endpointBegun() {
            endpointBegun = true;
        }

        private boolean outboundStarted(ClaimedVictim owned) {
            if (owned.disposition == ClaimDisposition.TERMINAL) {
                return false;
            }
            if (owned.disposition != ClaimDisposition.RELEASABLE) {
                throw new IllegalStateException(
                        "Cancel outbound ownership changed request_id="
                                + owned.requestId());
            }
            owned.disposition = ClaimDisposition.OUTBOUND;
            return true;
        }

        private boolean recordTerminal(
                ClaimedVictim owned,
                VictimTerminal terminal) {
            if (terminal == null
                    || terminal.requestId() != owned.requestId()) {
                return false;
            }
            boolean recorded = attempt.recordTerminal(owned.requestId());
            if (recorded) {
                owned.disposition = ClaimDisposition.TERMINAL;
            }
            return recorded;
        }

        private void transferred(ClaimedVictim owned) {
            if (attempt.isTerminal(owned.requestId())) {
                owned.disposition = ClaimDisposition.TERMINAL;
            } else {
                owned.disposition = ClaimDisposition.TRANSFERRED;
            }
        }

        private void terminal(ClaimedVictim owned) {
            owned.disposition = ClaimDisposition.TERMINAL;
        }

        private void transferUnknown(ClaimedVictim owned) {
            if (attempt.isTerminal(owned.requestId())) {
                owned.disposition = ClaimDisposition.TERMINAL;
                return;
            }
            attempt.recordUnknown(owned.requestId());
            request.endpoint().markPriorityCancelUnknown(
                    attempt.token(), owned.requestId());
            lifecycle.markPreemptionUnknown(owned.claim());
            transferred(owned);
        }

        private ExecutionResult finish(boolean hasNotFound) {
            if (attempt.allVictimsTerminal()
                    && request.admissionOpen().getAsBoolean()
                    && request.endpoint().commitPriorityPreemption(
                            attempt.token())
                    && attempt.markCommitted()) {
                committed = true;
                closed = true;
                return ExecutionResult.of(
                        ResultCode.COMMITTED, "committed");
            }
            boolean cleanSingleNotFound = hasNotFound
                    && attempt.victims().size() == 1
                    && !attempt.isTerminal(
                            attempt.victims().get(0).requestId());
            return abort(
                    cleanSingleNotFound
                            ? ResultCode.NOT_FOUND_UNRESOLVED
                            : ResultCode.CONTROL_FAILED,
                    cleanSingleNotFound
                            ? "cancel_not_found"
                            : "cancel_terminal_unknown");
        }

        private ExecutionResult abort(ResultCode code, String detail) {
            close();
            attempt.markAborted();
            String resultDetail = cleanupFailure == null
                    ? detail : detail + ";cleanup_failed=" + cleanupFailure;
            return ExecutionResult.of(code, resultDetail);
        }

        @Override
        public void close() {
            if (closed || committed) {
                return;
            }
            closed = true;

            // OUTBOUND means the call may have reached Prefill even if its
            // Java invocation or continuation failed. Transfer before endpoint
            // abort so neither owner can be mistaken for locally releasable.
            for (ClaimedVictim owned : claims) {
                if (owned.disposition == ClaimDisposition.OUTBOUND) {
                    try {
                        transferUnknown(owned);
                    } catch (RuntimeException | Error failure) {
                        recordCleanupFailure(
                                "transfer_unknown:" + owned.requestId(),
                                failure);
                    }
                }
            }
            if (endpointBegun) {
                try {
                    request.endpoint().abortPriorityPreemption(
                            attempt.token());
                } catch (RuntimeException | Error failure) {
                    recordCleanupFailure("endpoint_abort", failure);
                }
            }
            for (ClaimedVictim owned : claims) {
                if (owned.disposition == ClaimDisposition.RELEASABLE) {
                    try {
                        lifecycle.releasePreemptionClaim(owned.claim());
                    } catch (RuntimeException | Error failure) {
                        recordCleanupFailure(
                                "release_claim:" + owned.requestId(),
                                failure);
                    }
                }
            }
        }

        private void recordCleanupFailure(String operation, Throwable failure) {
            if (cleanupFailure == null) {
                cleanupFailure = operation + ":" + failureDetail(failure);
            }
        }
    }

    private long nextToken() {
        long token = tokenSequence.getAndIncrement();
        if (token <= 0) {
            throw new IllegalStateException("preemption attempt token exhausted");
        }
        return token;
    }
}
