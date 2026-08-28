package org.flexlb.balance.eviction;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.balance.preemption.PreemptionClaim;
import org.flexlb.balance.preemption.VictimTerminal;
import org.flexlb.balance.scheduler.RequestRegistry;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
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

    /** Exact lifecycle updates emitted by one Engine-Cancel transaction. */
    public sealed interface PreemptionUpdate {

        enum Step implements PreemptionUpdate {
            RELEASE,
            CANCEL_STARTED,
            CANCEL_ACCEPTED,
            CANCEL_NOT_FOUND,
            CANCEL_UNKNOWN
        }

        record Tombstoned(String detail) implements PreemptionUpdate {
        }
    }

    public enum Outcome {
        COMMITTED,
        NOT_FOUND_UNRESOLVED,
        CONFLICT,
        CONTROL_FAILED
    }

    public record PreemptionResult(Outcome outcome, String detail) {
        public PreemptionResult {
            Objects.requireNonNull(outcome, "outcome");
            Objects.requireNonNull(detail, "detail");
        }

        static PreemptionResult of(Outcome outcome, String detail) {
            return new PreemptionResult(outcome, detail);
        }
    }

    public record PreemptionCommand(
            DecodeEndpoint endpoint,
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
        public PreemptionCommand {
            if (endpoint == null || victims == null || victims.isEmpty()) {
                throw new IllegalArgumentException("endpoint and victims are required");
            }
            victims = List.copyOf(victims);
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
    private final RequestRegistry requests;
    private final AtomicLong tokenSequence = new AtomicLong(1);

    public DecodePreemptionCoordinator(
            EngineCancelChannel cancelChannel,
            RequestRegistry requests) {
        this.cancelChannel = Objects.requireNonNull(
                cancelChannel, "cancelChannel");
        this.requests = Objects.requireNonNull(requests, "requests");
    }

    public CompletableFuture<PreemptionResult> preempt(
            PreemptionCommand command) {
        long token = nextToken();
        List<PreemptionAttempt.Victim> victims =
                new ArrayList<>(command.victims().size());
        List<DecodeEndpoint.ReservationHandle> victimReservations =
                new ArrayList<>(command.victims().size());
        long endpointGenerationId =
                command.endpoint().getStatus().getGenerationId();
        for (DecodeRequestSnapshot victim : command.victims()) {
            Optional<CancelTarget> target = requests.findCancelTarget(
                    victim.requestId(), victim.reservationToken());
            if (target.isEmpty()) {
                return CompletableFuture.completedFuture(PreemptionResult.of(
                        Outcome.CONTROL_FAILED,
                        "cancel_owner_missing:" + victim.requestId()));
            }
            victims.add(new PreemptionAttempt.Victim(victim.requestId(), victim.priority(),
                    victim.kvTokens(), victim.phase(), victim.reservationToken(),
                    target.get()));
            victimReservations.add(new DecodeEndpoint.ReservationHandle(
                    endpointGenerationId,
                    victim.requestId(),
                    victim.reservationToken()));
        }

        PreemptionAttempt attempt = new PreemptionAttempt(token, victims);
        AttemptCapability capability = new AttemptCapability(
                command, requests, attempt);
        try {
            for (PreemptionAttempt.Victim victim : victims) {
                Optional<PreemptionClaim> claimAttempt = requests.tryClaim(
                        victim.requestId(), victim.reservationToken(),
                        token, command.detail());
                if (claimAttempt.isEmpty()) {
                    return CompletableFuture.completedFuture(capability.abort(
                            Outcome.CONFLICT, "victim_inflight_gone"));
                }
                PreemptionClaim claim = claimAttempt.get();
                ClaimedVictim owned = capability.add(victim, claim);
                if (claim.requestId() != victim.requestId()
                        || claim.attemptToken() != token) {
                    return CompletableFuture.completedFuture(capability.abort(
                            Outcome.CONTROL_FAILED,
                            "lifecycle_returned_mismatched_claim:" + owned.requestId()));
                }
            }

            DecodeEndpoint.PreemptionBeginResult begin =
                    command.endpoint().beginPriorityPreemption(
                            token, victimReservations,
                            command.incomingRequestId(), command.incomingKvTokens(),
                            command.incomingExpectedKvTokens(),
                            command.incomingPriority(), command.capacity());
            if (begin != DecodeEndpoint.PreemptionBeginResult.SUCCESS) {
                Outcome outcome = begin
                        == DecodeEndpoint.PreemptionBeginResult.ENDPOINT_RETIRED
                        ? Outcome.CONTROL_FAILED : Outcome.CONFLICT;
                return CompletableFuture.completedFuture(capability.abort(
                        outcome, "begin_" + begin.name().toLowerCase()));
            }
            capability.endpointBegun();
            if (!attempt.claimAll()) {
                return CompletableFuture.completedFuture(capability.abort(
                        Outcome.CONTROL_FAILED,
                        "attempt_claim_linearization_failed"));
            }

            if (!command.endpoint().markPriorityCancelInFlight(token)) {
                return CompletableFuture.completedFuture(capability.abort(
                        Outcome.CONTROL_FAILED,
                        "endpoint_cancel_linearization_failed"));
            }
            for (ClaimedVictim owned : capability.claims()) {
                if (!requests.tryApplyUpdate(
                        owned.claim(), PreemptionUpdate.Step.CANCEL_STARTED)) {
                    return CompletableFuture.completedFuture(capability.abort(
                            Outcome.CONTROL_FAILED,
                            "inflight_cancel_linearization_failed:"
                                    + owned.requestId()));
                }
            }
            if (!attempt.markCancelInFlight()) {
                return CompletableFuture.completedFuture(capability.abort(
                        Outcome.CONTROL_FAILED,
                        "attempt_cancel_linearization_failed"));
            }

            // Capture every terminal capability before the first outbound
            // side effect. The exact claim remains the only lookup key.
            List<TerminalSettlement> terminalSettlements =
                    new ArrayList<>(victims.size());
            for (ClaimedVictim owned : capability.claims()) {
                CompletableFuture<Boolean> completion = owned.claim()
                        .terminalObservation()
                        .handle((terminal, failure) -> failure == null
                                && capability.recordTerminal(owned, terminal))
                        .toCompletableFuture();
                terminalSettlements.add(
                        new TerminalSettlement(owned, completion));
            }

            List<CompletableFuture<EngineCancelChannel.CancelOutcome>>
                    acknowledgements = new ArrayList<>(victims.size());
            for (ClaimedVictim owned : capability.claims()) {
                if (capability.outboundStarted(owned)) {
                    acknowledgements.add(cancel(
                            owned.victim(), command.cancelAckTimeoutMs()));
                } else {
                    acknowledgements.add(CompletableFuture.completedFuture(
                            EngineCancelChannel.CancelOutcome.failed()));
                }
            }

            CompletableFuture<PreemptionResult> protocol =
                    CompletableFuture.allOf(
                            acknowledgements.toArray(new CompletableFuture[0]))
                    .thenCompose(ignored -> handleAcknowledgements(
                            capability, acknowledgements, terminalSettlements));
            return protocol.handle((result, failure) -> {
                if (failure != null) {
                    return capability.abort(
                            Outcome.CONTROL_FAILED,
                            "coordinator_continuation_failed:"
                                    + failureDetail(failure));
                }
                return result == null
                        ? capability.abort(
                                Outcome.CONTROL_FAILED,
                                "coordinator_returned_null_result")
                        : result;
            });
        } catch (RuntimeException | Error failure) {
            return CompletableFuture.completedFuture(capability.abort(
                    Outcome.CONTROL_FAILED,
                    "coordinator_setup_failed:" + failureDetail(failure)));
        }
    }

    private CompletableFuture<PreemptionResult> handleAcknowledgements(
            AttemptCapability capability,
            List<CompletableFuture<EngineCancelChannel.CancelOutcome>> acknowledgements,
            List<TerminalSettlement> terminalSettlements) {
        PreemptionCommand command = capability.command();
        RequestRegistry requests = capability.requests();
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
                    boolean transitioned =
                            command.endpoint().markPriorityCancelAccepted(
                                    attempt.token(), victim.requestId())
                            && requests.tryApplyUpdate(
                                    terminal.owned().claim(),
                                    PreemptionUpdate.Step.CANCEL_ACCEPTED);
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
                    command.endpoint().markPriorityCancelNotFound(
                            attempt.token(), victim.requestId());
                    requests.tryApplyUpdate(
                            terminal.owned().claim(),
                            PreemptionUpdate.Step.CANCEL_NOT_FOUND);
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
                        Math.max(1, command.cancelCompletionTimeoutMs()));
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
        PreemptionCommand command = capability.command();
        RequestRegistry requests = capability.requests();
        PreemptionAttempt attempt = capability.attempt();
        ClaimedVictim owned = terminal.owned();
        PreemptionAttempt.Victim victim = owned.victim();
        // Endpoint accounting remains the resource-owning CAS. The remaining
        // transitions are exact-token followers, but no WorkerStatus future
        // is required for this stronger proof.
        boolean endpointSettled = command.endpoint().settlePriorityTombstoned(
                attempt.token(), reservation(command.endpoint(), victim));
        boolean inflightSettled = endpointSettled
                && requests.tryApplyUpdate(
                        owned.claim(),
                        new PreemptionUpdate.Tombstoned(
                                command.detail()));
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
        private final PreemptionCommand command;
        private final RequestRegistry requests;
        private final PreemptionAttempt attempt;
        private final List<ClaimedVictim> claims = new ArrayList<>();
        private boolean endpointBegun;
        private boolean committed;
        private boolean closed;
        private String cleanupFailure;

        private AttemptCapability(
                PreemptionCommand command,
                RequestRegistry requests,
                PreemptionAttempt attempt) {
            this.command = command;
            this.requests = requests;
            this.attempt = attempt;
        }

        private PreemptionCommand command() {
            return command;
        }

        private RequestRegistry requests() {
            return requests;
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
            return claims;
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
            command.endpoint().markPriorityCancelUnknown(
                    attempt.token(), owned.requestId());
            requests.tryApplyUpdate(
                    owned.claim(), PreemptionUpdate.Step.CANCEL_UNKNOWN);
            transferred(owned);
        }

        private PreemptionResult finish(boolean hasNotFound) {
            if (attempt.allVictimsTerminal()
                    && command.admissionOpen().getAsBoolean()
                    && command.endpoint().commitPriorityPreemption(
                            attempt.token())
                    && attempt.markCommitted()) {
                committed = true;
                closed = true;
                return PreemptionResult.of(
                        Outcome.COMMITTED, "committed");
            }
            boolean cleanSingleNotFound = hasNotFound
                    && attempt.victims().size() == 1
                    && !attempt.isTerminal(
                            attempt.victims().get(0).requestId());
            return abort(
                    cleanSingleNotFound
                            ? Outcome.NOT_FOUND_UNRESOLVED
                            : Outcome.CONTROL_FAILED,
                    cleanSingleNotFound
                            ? "cancel_not_found"
                            : "cancel_terminal_unknown");
        }

        private PreemptionResult abort(Outcome outcome, String detail) {
            close();
            attempt.markAborted();
            String resultDetail = cleanupFailure == null
                    ? detail : detail + ";cleanup_failed=" + cleanupFailure;
            return PreemptionResult.of(outcome, resultDetail);
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
                    command.endpoint().abortPriorityPreemption(
                            attempt.token());
                } catch (RuntimeException | Error failure) {
                    recordCleanupFailure("endpoint_abort", failure);
                }
            }
            for (ClaimedVictim owned : claims) {
                if (owned.disposition == ClaimDisposition.RELEASABLE) {
                    try {
                        requests.tryApplyUpdate(
                                owned.claim(), PreemptionUpdate.Step.RELEASE);
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
