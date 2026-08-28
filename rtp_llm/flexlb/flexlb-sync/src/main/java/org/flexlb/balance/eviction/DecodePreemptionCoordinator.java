package org.flexlb.balance.eviction;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeEndpoint.DecodeRequestView;
import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.balance.preemption.VictimTerminal;
import org.flexlb.balance.scheduler.PreemptionRegistration;
import org.flexlb.balance.scheduler.RequestRegistry;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
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

    record PreemptionResult(
            boolean committed, boolean controlFailure, String detail) {
        public PreemptionResult {
            if (committed && controlFailure) {
                throw new IllegalArgumentException(
                        "committed preemption cannot be a control failure");
            }
            Objects.requireNonNull(detail, "detail");
        }
    }

    record PreemptionCommand(
            DecodeEndpoint endpoint,
            long incomingRequestId,
            long incomingKvTokens,
            long incomingExpectedKvTokens,
            int incomingPriority,
            DecodeEndpoint.AdmissionCapacity capacity,
            List<DecodeRequestView> victims,
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
            Set<Long> victimIds = new LinkedHashSet<>();
            for (DecodeRequestView victim : victims) {
                if (victim.requestId() <= 0L
                        || victim.reservationToken() <= 0L) {
                    throw new IllegalArgumentException(
                            "victim requestId and reservation token must be positive");
                }
                if (victim.phase() == null
                        || !victim.phase().requiresEngineCancel()) {
                    throw new IllegalArgumentException(
                            "coordinator accepts only Engine-Cancel victims");
                }
                if (!victimIds.add(victim.requestId())) {
                    throw new IllegalArgumentException(
                            "duplicate victim " + victim.requestId());
                }
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

    CompletableFuture<PreemptionResult> preempt(
            PreemptionCommand command) {
        long token = nextToken();
        List<CancelTarget> targets =
                new ArrayList<>(command.victims().size());
        List<DecodeEndpoint.ReservationHandle> victimReservations =
                new ArrayList<>(command.victims().size());
        long endpointGenerationId =
                command.endpoint().getStatus().getGenerationId();
        for (DecodeRequestView victim : command.victims()) {
            Optional<CancelTarget> target = requests.findCancelTarget(
                    victim.requestId(), victim.reservationToken());
            if (target.isEmpty()) {
                return CompletableFuture.completedFuture(new PreemptionResult(
                        false, true,
                        "cancel_owner_missing:" + victim.requestId()));
            }
            targets.add(target.get());
            victimReservations.add(new DecodeEndpoint.ReservationHandle(
                    endpointGenerationId,
                    victim.requestId(),
                    victim.reservationToken()));
        }

        AttemptCapability capability = new AttemptCapability(
                command, requests, token);
        try {
            for (int index = 0; index < command.victims().size(); index++) {
                DecodeRequestView victim = command.victims().get(index);
                Optional<PreemptionRegistration> claimAttempt = requests.tryClaim(
                        victim.requestId(), victim.reservationToken(),
                        token, command.detail());
                if (claimAttempt.isEmpty()) {
                    return CompletableFuture.completedFuture(capability.abort(
                            false, "victim_inflight_gone"));
                }
                PreemptionRegistration claim = claimAttempt.get();
                ClaimedVictim owned = capability.add(
                        victim, targets.get(index), claim);
                if (claim.requestId() != victim.requestId()
                        || claim.attemptToken() != token) {
                    return CompletableFuture.completedFuture(capability.abort(
                            true,
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
                return CompletableFuture.completedFuture(capability.abort(
                        begin == DecodeEndpoint.PreemptionBeginResult.ENDPOINT_RETIRED,
                        "begin_" + begin.name().toLowerCase()));
            }
            capability.endpointBegun();
            if (!command.endpoint().markPriorityCancelInFlight(token)) {
                return CompletableFuture.completedFuture(capability.abort(
                        true,
                        "endpoint_cancel_linearization_failed"));
            }
            for (ClaimedVictim owned : capability.claims) {
                if (!requests.tryApplyPreemptionPhase(
                        owned.claim(),
                        PreemptionCancelPhase.CANCEL_IN_FLIGHT)) {
                    return CompletableFuture.completedFuture(capability.abort(
                            true,
                            "inflight_cancel_linearization_failed:"
                                    + owned.requestId()));
                }
            }
            if (!capability.markCancelStarted()) {
                return CompletableFuture.completedFuture(capability.abort(
                        true,
                        "attempt_cancel_linearization_failed"));
            }

            // Capture every terminal capability before the first outbound
            // side effect. The exact claim remains the only lookup key.
            for (ClaimedVictim owned : capability.claims) {
                owned.terminalCompletion = owned.claim()
                        .terminalObservation()
                        .handle((terminal, failure) -> failure == null
                                && capability.recordTerminal(owned, terminal))
                        .toCompletableFuture();
            }

            List<CompletableFuture<EngineCancelChannel.CancelAck>>
                    acknowledgements = new ArrayList<>(capability.claims.size());
            for (ClaimedVictim owned : capability.claims) {
                if (capability.outboundStarted(owned)) {
                    acknowledgements.add(cancel(
                            owned, command.cancelAckTimeoutMs()));
                } else {
                    acknowledgements.add(CompletableFuture.completedFuture(
                            EngineCancelChannel.CancelAck.FAILED));
                }
            }

            CompletableFuture<PreemptionResult> protocol =
                    CompletableFuture.allOf(
                            acknowledgements.toArray(new CompletableFuture[0]))
                    .thenCompose(ignored -> handleAcknowledgements(
                            capability, acknowledgements));
            return protocol.handle((result, failure) -> {
                if (failure != null) {
                    return capability.abort(
                            true,
                            "coordinator_continuation_failed:"
                                    + failureDetail(failure));
                }
                return result == null
                        ? capability.abort(
                                true,
                                "coordinator_returned_null_result")
                        : result;
            });
        } catch (RuntimeException | Error failure) {
            return CompletableFuture.completedFuture(capability.abort(
                    true,
                    "coordinator_setup_failed:" + failureDetail(failure)));
        }
    }

    private CompletableFuture<PreemptionResult> handleAcknowledgements(
            AttemptCapability capability,
            List<CompletableFuture<EngineCancelChannel.CancelAck>> acknowledgements) {
        PreemptionCommand command = capability.command;
        RequestRegistry requests = capability.requests;
        // A transport-unknown ACK is not a negative acknowledgement: the
        // Prefill may have installed the intent before the reply was lost.
        // Such a child therefore waits for the canonical victim terminal
        // transaction exactly like an ACCEPTED child.
        List<ClaimedVictim> pendingTerminals = new ArrayList<>();
        boolean hasNotFound = false;

        List<DecodeRequestView> victims = command.victims();
        for (int i = 0; i < victims.size(); i++) {
            DecodeRequestView victim = victims.get(i);
            ClaimedVictim owned = capability.claims.get(i);
            if (capability.isTerminal(owned)) {
                continue;
            }
            EngineCancelChannel.CancelAck outcome = acknowledgements.get(i).join();
            switch (outcome) {
                case ACCEPTED -> {
                    boolean transitioned =
                            command.endpoint().recordPriorityCancelPhase(
                                    capability.token, victim.requestId(),
                                    PreemptionCancelPhase.CANCEL_REQUESTED)
                            && requests.tryApplyPreemptionPhase(
                                    owned.claim(),
                                    PreemptionCancelPhase.CANCEL_REQUESTED);
                    if (transitioned) {
                        capability.transferred(owned);
                        owned.acceptedAcknowledgement = true;
                        pendingTerminals.add(owned);
                    } else {
                        capability.transferUnknown(owned);
                        pendingTerminals.add(owned);
                    }
                }
                case NOT_FOUND -> {
                    command.endpoint().recordPriorityCancelPhase(
                            capability.token, victim.requestId(),
                            PreemptionCancelPhase.NOT_FOUND_STALE);
                    requests.tryApplyPreemptionPhase(
                            owned.claim(),
                            PreemptionCancelPhase.NOT_FOUND_STALE);
                    capability.transferred(owned);
                    if (!capability.isTerminal(owned)) {
                        hasNotFound = true;
                    }
                }
                case TOMBSTONED -> {
                    // Unlike NOT_FOUND, TOMBSTONED atomically proves absence
                    // and prevents every racing late Enqueue. It is therefore
                    // a terminal victim proof and contributes freed capacity
                    // to this same transaction.
                    settleTombstoned(capability, owned);
                }
                case FAILED, UNSUPPORTED -> {
                    capability.transferUnknown(owned);
                    pendingTerminals.add(owned);
                }
            }
        }

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
        for (ClaimedVictim pending : pendingTerminals) {
            // The unbounded continuation was installed before the first RPC.
            // Timing out this admission wait cannot cancel or lose a later
            // canonical victim terminal.
            boundedSettlements.add(withDeadline(
                    pending.terminalCompletion, completionDeadlineNanos)
                    .exceptionally(ignored -> false));
        }

        final boolean ackNotFound = hasNotFound;
        return CompletableFuture.allOf(boundedSettlements.toArray(new CompletableFuture[0]))
                .thenApply(ignored -> {
                    for (int i = 0; i < boundedSettlements.size(); i++) {
                        if (!boundedSettlements.get(i).join()
                                && !pendingTerminals.get(i)
                                        .acceptedAcknowledgement) {
                            capability.transferUnknown(pendingTerminals.get(i));
                        }
                    }
                    return capability.finish(ackNotFound);
                });
    }

    private static boolean settleTombstoned(
            AttemptCapability capability,
            ClaimedVictim owned) {
        PreemptionCommand command = capability.command;
        RequestRegistry requests = capability.requests;
        DecodeRequestView victim = owned.victim();
        // Endpoint accounting remains the resource-owning CAS. The remaining
        // transitions are exact-token followers, but no WorkerStatus future
        // is required for this stronger proof.
        boolean endpointSettled = command.endpoint().settlePriorityTombstoned(
                capability.token, reservation(command.endpoint(), victim));
        boolean inflightSettled = endpointSettled
                && requests.trySettlePreemptionTombstone(
                        owned.claim(), command.detail());
        boolean attemptSettled = inflightSettled
                && capability.recordTerminal(owned);
        return endpointSettled && inflightSettled && attemptSettled;
    }

    private static DecodeEndpoint.ReservationHandle reservation(
            DecodeEndpoint endpoint,
            DecodeRequestView victim) {
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

    private CompletableFuture<EngineCancelChannel.CancelAck> cancel(
            ClaimedVictim victim,
            long timeoutMs) {
        try {
            CompletableFuture<EngineCancelChannel.CancelAck> stage =
                    cancelChannel.cancel(
                            victim.target(), victim.requestId(), timeoutMs);
            if (stage == null) {
                return CompletableFuture.completedFuture(
                        EngineCancelChannel.CancelAck.FAILED);
            }
            return stage.handle((outcome, failure) -> failure == null
                            && outcome != null
                    ? outcome : EngineCancelChannel.CancelAck.FAILED);
        } catch (RuntimeException | Error failure) {
            // The capability marked this victim OUTBOUND before invocation,
            // so close conservatively transfers its exact claims to UNKNOWN.
            return CompletableFuture.completedFuture(
                    EngineCancelChannel.CancelAck.FAILED);
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

    private enum ClaimDisposition {
        RELEASABLE,
        OUTBOUND,
        TRANSFERRED,
        TERMINAL
    }

    /** Exact opaque slot claim paired with its immutable endpoint victim. */
    private static final class ClaimedVictim {
        private final DecodeRequestView victim;
        private final CancelTarget target;
        private final PreemptionRegistration claim;
        private CompletableFuture<Boolean> terminalCompletion;
        private boolean acceptedAcknowledgement;
        private volatile ClaimDisposition disposition =
                ClaimDisposition.RELEASABLE;

        private ClaimedVictim(
                DecodeRequestView victim,
                CancelTarget target,
                PreemptionRegistration claim) {
            this.victim = victim;
            this.target = target;
            this.claim = claim;
        }

        private DecodeRequestView victim() {
            return victim;
        }

        private CancelTarget target() {
            return target;
        }

        private PreemptionRegistration claim() {
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
        private final long token;
        private final List<ClaimedVictim> claims = new ArrayList<>();
        /** Historical outbound boundary retained after an abort. */
        private boolean cancelStarted;
        private boolean endpointBegun;
        private boolean committed;
        private boolean closed;
        private String cleanupFailure;

        private AttemptCapability(
                PreemptionCommand command,
                RequestRegistry requests,
                long token) {
            if (token <= 0) {
                throw new IllegalArgumentException("token is required");
            }
            this.command = command;
            this.requests = requests;
            this.token = token;
        }

        private ClaimedVictim add(
                DecodeRequestView victim,
                CancelTarget target,
                PreemptionRegistration claim) {
            ClaimedVictim owned = new ClaimedVictim(victim, target, claim);
            claims.add(owned);
            return owned;
        }

        private void endpointBegun() {
            endpointBegun = true;
        }

        /** Linearization immediately before the first outbound Cancel RPC. */
        private synchronized boolean markCancelStarted() {
            if (cancelStarted) {
                return false;
            }
            cancelStarted = true;
            return true;
        }

        private synchronized boolean outboundStarted(ClaimedVictim owned) {
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

        private synchronized boolean recordTerminal(
                ClaimedVictim owned,
                VictimTerminal terminal) {
            if (terminal == null
                    || terminal.requestId() != owned.requestId()) {
                return false;
            }
            return recordTerminal(owned);
        }

        /** Exactly-once convergence for terminal facts from either callback. */
        private synchronized boolean recordTerminal(ClaimedVictim owned) {
            if (owned.disposition == ClaimDisposition.TERMINAL) {
                return true;
            }
            if (!cancelStarted) {
                return false;
            }
            owned.disposition = ClaimDisposition.TERMINAL;
            return true;
        }

        private synchronized boolean isTerminal(ClaimedVictim owned) {
            return owned.disposition == ClaimDisposition.TERMINAL;
        }

        private synchronized boolean allVictimsTerminal() {
            if (claims.isEmpty()) {
                return false;
            }
            return claims.stream().allMatch(
                    owned -> owned.disposition == ClaimDisposition.TERMINAL);
        }

        private synchronized void transferred(ClaimedVictim owned) {
            if (isTerminal(owned)) {
                owned.disposition = ClaimDisposition.TERMINAL;
            } else {
                owned.disposition = ClaimDisposition.TRANSFERRED;
            }
        }

        private void transferUnknown(ClaimedVictim owned) {
            if (!shouldTransferUnknown(owned)) {
                return;
            }
            command.endpoint().recordPriorityCancelPhase(
                    token, owned.requestId(),
                    PreemptionCancelPhase.CANCEL_UNKNOWN);
            requests.tryApplyPreemptionPhase(
                    owned.claim(), PreemptionCancelPhase.CANCEL_UNKNOWN);
            transferred(owned);
        }

        private synchronized boolean shouldTransferUnknown(
                ClaimedVictim owned) {
            return owned.disposition == ClaimDisposition.OUTBOUND;
        }

        private PreemptionResult finish(boolean hasNotFound) {
            if (allVictimsTerminal()
                    && command.admissionOpen().getAsBoolean()
                    && command.endpoint().commitPriorityPreemption(
                            token)) {
                markCommitted();
                return new PreemptionResult(
                        true, false, "committed");
            }
            boolean cleanSingleNotFound = hasNotFound
                    && claims.size() == 1
                    && !isTerminal(claims.get(0));
            return abort(
                    !cleanSingleNotFound,
                    cleanSingleNotFound
                            ? "cancel_not_found"
                            : "cancel_terminal_unknown");
        }

        private PreemptionResult abort(
                boolean controlFailure, String detail) {
            close();
            String resultDetail = cleanupFailure == null
                    ? detail : detail + ";cleanup_failed=" + cleanupFailure;
            return new PreemptionResult(
                    false, controlFailure, resultDetail);
        }

        @Override
        public void close() {
            if (!beginClose()) {
                return;
            }

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
                            token);
                } catch (RuntimeException | Error failure) {
                    recordCleanupFailure("endpoint_abort", failure);
                }
            }
            for (ClaimedVictim owned : claims) {
                if (owned.disposition == ClaimDisposition.RELEASABLE) {
                    try {
                        requests.tryReleasePreemption(owned.claim());
                    } catch (RuntimeException | Error failure) {
                        recordCleanupFailure(
                                "release_claim:" + owned.requestId(),
                                failure);
                    }
                }
            }
        }

        private synchronized void markCommitted() {
            committed = true;
            closed = true;
        }

        private synchronized boolean beginClose() {
            if (closed || committed) {
                return false;
            }
            closed = true;
            return true;
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
