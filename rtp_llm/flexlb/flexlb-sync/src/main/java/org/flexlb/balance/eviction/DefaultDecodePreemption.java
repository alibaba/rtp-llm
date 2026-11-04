package org.flexlb.balance.eviction;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity;
import org.flexlb.balance.endpoint.DecodeResources.DecodeRequestView;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.eviction.DecodePreemptionPolicy.DecodeEvictionProposal;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.balance.preemption.VictimResolution;
import org.flexlb.balance.scheduler.AdmissionHandle;
import org.flexlb.balance.scheduler.ContextResult;
import org.flexlb.balance.scheduler.DeliveryDispatcher;
import org.flexlb.balance.scheduler.PreemptionRegistration;
import org.flexlb.balance.scheduler.RequestContext;
import org.flexlb.balance.scheduler.RequestLifecycle;
import org.flexlb.balance.scheduler.RequestRepository;
import org.flexlb.balance.scheduler.RequestTaskExecutor;
import org.flexlb.balance.scheduler.TaskTimer;
import org.flexlb.config.ConfigService;
import org.flexlb.config.PreemptionConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter.CancelEvent;
import org.flexlb.service.monitor.RequestSchedulerReporter.EvictionEvent;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;
import org.flexlb.util.PriorityNormalizer;

import javax.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BooleanSupplier;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

/**
 * Obtains capacity at one selected Decode generation through local withdrawal or Engine cancellation.
 *
 * <p>The scheduler supplies the incoming request context and blocked endpoint. This class owns the
 * two-phase protocol and token fencing, then returns the reservation to the caller.
 * Engine acknowledgement is only control evidence; the canonical victim
 * resolution transaction may complete before or after that acknowledgement.</p>
 */
public class DefaultDecodePreemption implements DecodePreemption {
    private static final RequestLifecycle.PreemptionEvent START_CANCEL =
            new RequestLifecycle.PreemptionEvent(
                    PreemptionCancelPhase.CANCEL_IN_FLIGHT, false, null, null, null);

    record PreemptionCommand(
            DecodeEndpoint endpoint,
            Request request,
            SchedulingMetadata metadata,
            List<DecodeRequestView> victims,
            long cancelAckTimeoutMs,
            long preemptionTimeoutMs,
            BooleanSupplier admissionOpen,
            String detail) {
        public PreemptionCommand {
            checkArgument(
                    endpoint != null && victims != null && !victims.isEmpty(),
                    "endpoint and victims are required");
            victims = List.copyOf(victims);
            Objects.requireNonNull(request, "request");
            Objects.requireNonNull(metadata, "metadata");
            checkArgument(
                    request.getRequestId() != null && !request.getRequestId().isBlank(),
                    "incoming request id is required");
            Set<String> victimIds = new LinkedHashSet<>();
            for (DecodeRequestView victim : victims) {
                checkArgument(
                        victim.requestId() != null
                                && !victim.requestId().isBlank()
                                && victim.reservationToken() > 0L,
                        "victim requestId and reservation token must be positive");
                checkArgument(
                        victim.phase() != null && victim.phase().requiresEngineCancel(),
                        "capacity acquisition accepts only Engine-Cancel victims");
                checkArgument(
                        victimIds.add(victim.requestId()),
                        "duplicate victim %s",
                        victim.requestId());
            }
            checkArgument(admissionOpen != null, "admission gate is required");
        }
    }

    private final RequestSchedulerReporter reporter;
    private volatile boolean shutdown;
    private final java.util.concurrent.atomic.AtomicInteger activeWork =
            new java.util.concurrent.atomic.AtomicInteger();
    private final ThreadLocal<Boolean> acquiring = new ThreadLocal<>();
    private final DeliveryDispatcher dispatcher;
    private final RequestRepository requests;
    private final RequestTaskExecutor executor;
    private final TaskTimer timer;

    private final AtomicLong tokenSequence = new AtomicLong(1);

    private final DecodePreemptionPolicy selectionPolicy;
    private final AdmissionCapacity capacity;
    private final PreemptionConfig preemption;

    public DefaultDecodePreemption(
            DeliveryDispatcher dispatcher,
            RequestRepository requests,
            RequestTaskExecutor executor,
            TaskTimer timer,
            RequestSchedulerReporter reporter,
            ConfigService configService,
            AdmissionCapacity capacity,
            DecodePreemptionPolicy selectionPolicy) {
        this.reporter = Objects.requireNonNull(reporter, "reporter");
        this.dispatcher = Objects.requireNonNull(dispatcher, "dispatcher");
        this.requests = Objects.requireNonNull(requests, "requests");
        this.executor = Objects.requireNonNull(executor, "executor");
        this.timer = Objects.requireNonNull(timer, "timer");

        this.selectionPolicy = Objects.requireNonNull(selectionPolicy, "selectionPolicy");
        this.capacity = Objects.requireNonNull(capacity, "capacity");
        var config = configService.loadBalanceConfig();
        var policy =
                config.isQueue() && config.isPriorityOrdering()
                        ? config.priorityOrdering().getPreemption()
                        : null;
        if (policy == null) {
            this.preemption = null;
        } else {
            this.preemption = new PreemptionConfig();
            this.preemption.setTimeoutMs(policy.getTimeoutMs());
            this.preemption.setAllowedVictimStages(
                    policy.getAllowedVictimStages() == null
                            ? java.util.Set.of()
                            : java.util.Set.copyOf(policy.getAllowedVictimStages()));
        }
    }

    CompletableFuture<PreemptionResult> preempt(RequestContext context, PreemptionCommand command) {
        Objects.requireNonNull(context, "context");
        long token = nextToken();
        List<DecodeResources.ReservationHandle> victimReservations =
                new ArrayList<>(command.victims().size());
        long endpointGenerationId = command.endpoint().getStatus().getGenerationId();
        Attempt attempt = new Attempt(command, token);
        try {
            for (DecodeRequestView victim : command.victims()) {
                var owner = requests.ownerOf(victim.requestId());
                if (owner == null) {
                    return CompletableFuture.completedFuture(
                            attempt.abort(true, "cancel_owner_missing:" + victim.requestId()));
                }
                var reservation =
                        new DecodeResources.ReservationHandle(
                                endpointGenerationId,
                                victim.requestId(),
                                victim.reservationToken());
                Optional<PreemptionRegistration> claimAttempt = owner.tryClaim(reservation, token);
                if (claimAttempt.isEmpty()) {
                    return CompletableFuture.completedFuture(
                            attempt.abort(false, "victim_inflight_gone"));
                }
                PreemptionRegistration claim = claimAttempt.get();
                ClaimedVictim owned = new ClaimedVictim(claim);
                attempt.claims.add(owned);
                victimReservations.add(reservation);
                if (!java.util.Objects.equals(
                                claim.owner().data().getRequestId(), victim.requestId())
                        || claim.attemptToken() != token
                        || !reservation.equals(claim.reservation())) {
                    return CompletableFuture.completedFuture(
                            attempt.abort(
                                    true,
                                    "lifecycle_returned_mismatched_claim:" + owned.requestId()));
                }
            }

            DecodeResources.PreemptionBeginResult begin =
                    command.endpoint()
                            .beginPreemption(
                                    token,
                                    victimReservations,
                                    command.request().getRequestId(),
                                    command.request().getHardKvTokens(),
                                    command.request().getExpectedKvTokens(),
                                    command.metadata().priority(),
                                    capacity);
            if (begin != DecodeResources.PreemptionBeginResult.SUCCESS) {
                return CompletableFuture.completedFuture(
                        attempt.abort(
                                begin == DecodeResources.PreemptionBeginResult.ENDPOINT_RETIRED,
                                "begin_" + begin.name().toLowerCase()));
            }
            attempt.endpointBegun = true;
            attempt.observe();
            for (ClaimedVictim owned : attempt.claims) {
                var scheduler = owned.claim.owner().data().getLifecycle();
                if (!scheduler.onPreemption(owned.claim, START_CANCEL)) {
                    return CompletableFuture.completedFuture(
                            attempt.abort(
                                    true,
                                    "inflight_cancel_linearization_failed:" + owned.requestId()));
                }
            }
            for (ClaimedVictim owned : attempt.claims) {
                owned.acknowledgement =
                        attempt.outboundStarted(owned)
                                ? cancel(owned, command, attempt.expiresAt)
                                : CompletableFuture.completedFuture(
                                        PreemptionRegistration.Reply.UNKNOWN);
            }

            CompletableFuture<PreemptionResult> protocol =
                    CompletableFuture.allOf(
                                    attempt.claims.stream()
                                            .map(owned -> owned.acknowledgement)
                                            .toArray(CompletableFuture[]::new))
                            .thenComposeAsync(
                                    ignored -> handleAcknowledgements(context, attempt),
                                    work -> executor.execute(context, work))
                            .whenComplete((ignored, failure) -> attempt.settlement.cancel(false));
            return protocol.handle(
                    (result, failure) -> {
                        if (failure != null) {
                            return attempt.abort(
                                    true,
                                    "coordinator_continuation_failed:" + failureDetail(failure));
                        }
                        return result;
                    });
        } catch (RuntimeException | Error failure) {
            return CompletableFuture.completedFuture(
                    attempt.abort(true, "coordinator_setup_failed:" + failureDetail(failure)));
        }
    }

    private CompletableFuture<PreemptionResult> handleAcknowledgements(
            RequestContext context, Attempt attempt) {
        // A transport-unknown ACK is not a negative acknowledgement: the
        // Prefill may have installed the intent before the reply was lost.
        // Such a child therefore waits for the canonical victim resolution
        // transaction exactly like an ACCEPTED child.
        int waiting = 0;
        boolean hasNotFound = false;

        for (ClaimedVictim owned : attempt.claims) {
            if (owned.requestResolved()) {
                continue;
            }
            var progress = owned.acknowledgement.join();
            if (progress == PreemptionRegistration.Reply.UNKNOWN) {
                attempt.transferUnknown(owned);
            } else {
                attempt.transferred(owned);
            }
            if (progress == PreemptionRegistration.Reply.NOT_FOUND) {
                hasNotFound |= !owned.requestResolved();
            } else {
                waiting++;
            }
        }

        if (waiting == 0) {
            attempt.settlement.complete(attempt.allVictimsResolved());
            return CompletableFuture.completedFuture(
                    attempt.finish(hasNotFound, attempt.settlement.getNow(false)));
        }
        // Keep early rejection when the waited victims resolve but a NOT_FOUND victim has not.
        if (hasNotFound) {
            var pending =
                    CompletableFuture.allOf(
                            attempt.claims.stream()
                                    .filter(
                                            owned ->
                                                    owned.acknowledgement.join()
                                                            != PreemptionRegistration.Reply
                                                                    .NOT_FOUND)
                                    .map(owned -> owned.claim.completion().toCompletableFuture())
                                    .toArray(CompletableFuture<?>[]::new));
            pending.whenComplete(
                    (ignored, failure) ->
                            attempt.settlement.complete(attempt.allVictimsResolved()));
            attempt.settlement.whenComplete((ignored, failure) -> pending.cancel(false));
        }
        final boolean ackNotFound = hasNotFound;
        // Late facts still settle victims; they cannot change the frozen admission result.
        return attempt.settlement.thenApplyAsync(
                resolved -> attempt.finish(ackNotFound, resolved),
                work -> executor.execute(context, work));
    }

    private CompletableFuture<PreemptionRegistration.Reply> cancel(
            ClaimedVictim victim, PreemptionCommand command, long expiresAt) {
        try {
            long remaining =
                    java.util.concurrent.TimeUnit.NANOSECONDS.toMillis(
                            expiresAt - System.nanoTime());
            if (remaining <= 0) {
                return CompletableFuture.completedFuture(PreemptionRegistration.Reply.UNKNOWN);
            }
            return dispatcher.preempt(
                    victim.claim,
                    command.endpoint(),
                    command.detail(),
                    Math.min(command.cancelAckTimeoutMs(), remaining));
        } catch (RuntimeException | Error failure) {
            // The attempt marked this victim OUTBOUND before invocation,
            // so close conservatively transfers its exact claims to UNKNOWN.
            return CompletableFuture.completedFuture(PreemptionRegistration.Reply.UNKNOWN);
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
        TRANSFERRED
    }

    /** Exact opaque request claim paired with its immutable endpoint victim. */
    private final class ClaimedVictim {
        private final PreemptionRegistration claim;
        private CompletableFuture<PreemptionRegistration.Reply> acknowledgement;
        private volatile ClaimDisposition disposition = ClaimDisposition.RELEASABLE;

        private ClaimedVictim(PreemptionRegistration claim) {
            this.claim = claim;
        }

        private String requestId() {
            return claim.reservation().requestId();
        }

        private boolean requestResolved() {
            VictimResolution resolution = claim.result();
            return resolution != null
                    && java.util.Objects.equals(resolution.requestId(), requestId());
        }
    }

    /**
     * The one owner of endpoint admission plus every exact RequestContext claim.
     * A non-committed close is total: uncertain outbound claims transfer to
     * reconciliation, the incoming endpoint reservation aborts, and only
     * claims which never crossed an outbound boundary are released.
     */
    private final class Attempt implements AutoCloseable {
        private final PreemptionCommand command;
        private final long token;
        private final List<ClaimedVictim> claims = new ArrayList<>();
        private final long expiresAt;
        private final CompletableFuture<Boolean> settlement = new CompletableFuture<>();
        private boolean endpointBegun;
        private boolean closed;
        private String cleanupFailure;

        private Attempt(PreemptionCommand command, long token) {
            this.command = command;
            this.token = token;
            this.expiresAt =
                    System.nanoTime()
                            + java.util.concurrent.TimeUnit.MILLISECONDS.toNanos(
                                    command.preemptionTimeoutMs());
            activeWork.incrementAndGet();
        }

        private void observe() {
            var signals =
                    claims.stream()
                            .map(v -> v.claim.completion().toCompletableFuture())
                            .toArray(CompletableFuture[]::new);
            var all = CompletableFuture.allOf(signals);
            all.whenComplete((ignored, failure) -> settlement.complete(allVictimsResolved()));
            settlement.whenComplete((ignored, failure) -> all.cancel(false));
            if (!settlement.isDone()) {
                long remaining = expiresAt - System.nanoTime();
                if (remaining <= 0) {
                    settlement.complete(false);
                } else {
                    var deadline =
                            timer.schedule(
                                    () -> settlement.complete(allVictimsResolved()),
                                    Math.max(
                                            1L,
                                            java.util.concurrent.TimeUnit.NANOSECONDS.toMillis(
                                                    remaining)));
                    settlement.whenComplete((ignored, failure) -> deadline.cancel(false));
                }
            }
        }

        private synchronized boolean outboundStarted(ClaimedVictim owned) {
            if (owned.requestResolved() || settlement.isDone() && !settlement.getNow(false)) {
                return false;
            }
            if (owned.disposition != ClaimDisposition.RELEASABLE) {
                throw new IllegalStateException(
                        "Cancel outbound ownership changed request_id=" + owned.requestId());
            }
            // Install the conservative resource hold before RPC invocation. No ACK changes this
            // fact.
            if (!command.endpoint()
                    .updatePreemption(
                            token,
                            DecodeResources.PreemptionUpdate.handedOff(owned.claim.reservation()))
                    .status()
                    .applied()) {
                return false;
            }
            owned.disposition = ClaimDisposition.OUTBOUND;
            return true;
        }

        private synchronized boolean allVictimsResolved() {
            if (claims.isEmpty()) {
                return false;
            }
            return claims.stream().allMatch(ClaimedVictim::requestResolved);
        }

        private synchronized void transferred(ClaimedVictim owned) {
            owned.disposition = ClaimDisposition.TRANSFERRED;
        }

        private void transferUnknown(ClaimedVictim owned) {
            if (owned.disposition != ClaimDisposition.OUTBOUND || owned.requestResolved()) {
                return;
            }
            owned.claim
                    .owner()
                    .data()
                    .getLifecycle()
                    .onPreemption(
                            owned.claim,
                            new RequestLifecycle.PreemptionEvent(
                                    PreemptionCancelPhase.CANCEL_UNKNOWN, false, null, null, null));
            transferred(owned);
        }

        private PreemptionResult finish(boolean hasNotFound, boolean victimsResolved) {
            DecodeResources.ReservationHandle incoming =
                    victimsResolved
                                    && !closed
                                    && !shutdown
                                    && command.admissionOpen().getAsBoolean()
                            ? command.endpoint().commitPreemption(token)
                            : null;
            if (incoming != null) {
                if (beginClose()) activeWork.decrementAndGet();
                return new PreemptionResult(incoming, false, "committed");
            }
            boolean cleanSingleNotFound =
                    hasNotFound && claims.size() == 1 && !claims.get(0).requestResolved();
            return abort(
                    !cleanSingleNotFound,
                    cleanSingleNotFound ? "cancel_not_found" : "cancel_terminal_unknown");
        }

        private PreemptionResult abort(boolean controlFailure, String detail) {
            close();
            String resultDetail =
                    cleanupFailure == null ? detail : detail + ";cleanup_failed=" + cleanupFailure;
            return new PreemptionResult(null, controlFailure, resultDetail);
        }

        @Override
        public void close() {
            if (!beginClose()) {
                return;
            }
            try {
                settlement.complete(false);

                // OUTBOUND means the call may have reached Prefill even if its
                // Java invocation or continuation failed. Transfer before endpoint
                // abort so neither owner can be mistaken for locally releasable.
                for (ClaimedVictim owned : claims) {
                    if (owned.disposition == ClaimDisposition.OUTBOUND) {
                        cleanup(
                                "transfer_unknown:" + owned.requestId(),
                                () -> transferUnknown(owned));
                    }
                }
                if (endpointBegun) {
                    cleanup("endpoint_abort", () -> command.endpoint().abortPreemption(token));
                }
                for (ClaimedVictim owned : claims) {
                    if (owned.disposition == ClaimDisposition.RELEASABLE
                            && !owned.requestResolved()) {
                        cleanup(
                                "release_claim:" + owned.requestId(),
                                () ->
                                        owned.claim
                                                .owner()
                                                .data()
                                                .getLifecycle()
                                                .onPreemption(
                                                        owned.claim,
                                                        new RequestLifecycle.PreemptionEvent(
                                                                null, true, null, null, null)));
                    }
                }
            } finally {
                activeWork.decrementAndGet();
            }
        }

        private synchronized boolean beginClose() {
            if (closed) {
                return false;
            }
            closed = true;
            return true;
        }

        private void cleanup(String operation, Runnable action) {
            try {
                action.run();
            } catch (RuntimeException | Error failure) {
                if (cleanupFailure == null) {
                    cleanupFailure = operation + ":" + failureDetail(failure);
                }
            }
        }
    }

    private long nextToken() {
        long token = tokenSequence.getAndIncrement();
        checkState(token > 0, "preemption attempt token exhausted");
        return token;
    }

    @Override
    public void stopAdmission() {
        shutdown = true;
    }

    @Override
    public boolean awaitIdle(long deadlineNanos) {
        if (Boolean.TRUE.equals(acquiring.get()) && activeWork.get() != 0) return false;
        boolean interrupted = false;
        try {
            while (activeWork.get() != 0) {
                if (deadlineNanos != Long.MAX_VALUE && System.nanoTime() >= deadlineNanos)
                    return false;
                interrupted |= Thread.interrupted();
                java.util.concurrent.locks.LockSupport.parkNanos(100_000L);
            }
            return true;
        } finally {
            if (interrupted) Thread.currentThread().interrupt();
        }
    }

    @Override
    @PreDestroy
    public void close() {
        stopAdmission();
        checkState(awaitIdle(Long.MAX_VALUE), "preemption cannot wait for its own acquisition");
    }

    @Override
    public CompletableFuture<PreemptionResult> tryReclaim(
            RequestContext ctx, WorkerEndpoint endpoint) {
        activeWork.incrementAndGet();
        var previous = acquiring.get();
        acquiring.set(Boolean.TRUE);
        try {
            return startReclaim(ctx, endpoint);
        } finally {
            if (previous == null) acquiring.remove();
            else acquiring.set(previous);
            activeWork.decrementAndGet();
        }
    }

    /**
     * Starts one Decode preemption transaction. The caller owns the returned reservation and must
     * either attach it to its placement attempt or release it.
     */
    private CompletableFuture<PreemptionResult> startReclaim(
            RequestContext ctx, WorkerEndpoint blockedEndpoint) {
        Request request = ctx.data().getRequest();
        SchedulingMetadata metadata = ctx.data().getSchedulingMetadata();
        if (shutdown
                || ctx.data().getFuture().isDone()
                || System.currentTimeMillis() >= ctx.data().getRequestExpiresAtMs()
                || !PriorityNormalizer.hasPriority(metadata.priority())
                || preemption == null) {
            return null;
        }
        if (!(blockedEndpoint instanceof DecodeEndpoint decodeEndpoint)
                || (!preemption.allows(VictimStage.DECODE_RESERVED)
                        && !preemption.allows(VictimStage.DECODE_ENGINE_OWNED))) {
            return null;
        }
        DecodeEvictionProposal proposal = planDecodeEviction(request, metadata, decodeEndpoint);
        if (proposal == null
                || !ctx.data()
                        .getLifecycle()
                        .isAdmissionOpen(request.getRequestId(), ctx.data().getFuture())) {
            return null;
        }
        if (proposal.requiresEngineCancel()) {
            return startEngineCancelPreemption(ctx, proposal, decodeEndpoint, request, metadata);
        }
        List<DecodeResources.ReservationHandle> victims =
                new ArrayList<>(proposal.victims().size());
        for (DecodeRequestView victim : proposal.victims()) {
            victims.add(
                    new DecodeResources.ReservationHandle(
                            decodeEndpoint.getStatus().getGenerationId(),
                            victim.requestId(),
                            victim.reservationToken()));
        }
        DecodeResources.ReservationHandle incoming =
                replaceQueuedDecodeReservations(
                        decodeEndpoint, victims, request, metadata.priority());
        if (incoming == null) {
            reportEviction(
                    EvictionEvent.COMMIT,
                    ctx.data().getPriority(),
                    ctx.data().getRequestId(),
                    proposal.evictionCase(),
                    "conflict");
            return null;
        }
        report(
                ctx.data().getRequestId(),
                "local preemption",
                () -> {
                    reporter.reportPreemptionVictims(
                            ctx.data().getRequestId(),
                            ctx.data().getPriority(),
                            proposal.endpointId(),
                            proposal.evictionCase(),
                            proposal.victims(),
                            false);
                    recordDecodePlanObservability(ctx, proposal);
                });
        return CompletableFuture.completedFuture(
                new PreemptionResult(incoming, false, "committed"));
    }

    /** Metrics observe plans and commits without owning the reservation transaction. */
    private void reportEviction(
            EvictionEvent event,
            int priority,
            String requestId,
            String evictionCase,
            String outcome) {
        report(
                requestId,
                event == EvictionEvent.PLAN ? "eviction plan" : "eviction commit",
                () -> reporter.reportEviction(event, priority, evictionCase, outcome));
    }

    // ==================== Decode eviction ====================
    /**
     * Build one side-effect-free plan from one exact cluster snapshot.
     */
    private DecodeEvictionProposal planDecodeEviction(
            Request request, SchedulingMetadata metadata, DecodeEndpoint selectedEndpoint) {
        DecodeResources.ResourceSnapshot selected = selectedEndpoint.resourceSnapshot();
        if (selectedEndpoint.isRetired()) {
            return null;
        }
        Map<String, String> failures = new HashMap<>();
        var decision =
                selectionPolicy.planDecode(
                        request, metadata.priority(), capacity, selected, preemption, failures);
        if (decision.deficit().fits()) {
            return null;
        }
        DecodeEvictionProposal proposal = decision.proposal();
        if (proposal != null) {
            validateProposal(proposal, selected, request.getRequestId(), metadata.priority());
        }
        if (proposal == null) {
            reportEviction(
                    EvictionEvent.PLAN,
                    metadata.priority(),
                    request.getRequestId(),
                    decision.evictionCase(),
                    "infeasible");
            Logger.debug(
                    "[decode-capacity] Decode eviction infeasible:"
                            + " request_id={} priority={} worker={} reasons={}",
                    request.getRequestId(),
                    metadata.priority(),
                    selected.routing().address(),
                    failures);
            return null;
        }
        reportEviction(
                EvictionEvent.PLAN,
                metadata.priority(),
                request.getRequestId(),
                proposal.evictionCase(),
                "feasible");
        return proposal;
    }

    /** Validate the policy's authority; exact live token/capacity checks remain in the ledger. */
    private void validateProposal(
            DecodeEvictionProposal proposal,
            DecodeResources.ResourceSnapshot snapshot,
            String incomingId,
            int incomingPriority) {
        checkArgument(
                Objects.equals(proposal.endpointId(), snapshot.routing().address()),
                "preemption proposal must target the selected endpoint");
        checkArgument(!proposal.victims().isEmpty(), "preemption proposal requires victims");
        Set<String> ids = new LinkedHashSet<>();
        for (DecodeRequestView victim : proposal.victims()) {
            checkArgument(
                    victim.equals(snapshot.requests().get(victim.requestId())),
                    "preemption victim must match the supplied snapshot");
            checkArgument(
                    !java.util.Objects.equals(victim.requestId(), incomingId)
                            && ids.add(victim.requestId())
                            && victim.reservationToken() > 0L
                            && victim.priorityKnown()
                            && PriorityNormalizer.hasPriority(victim.priority())
                            && victim.priority() < incomingPriority
                            && !victim.claimedForPreemption(),
                    "preemption victim identity or priority is invalid");
            checkArgument(
                    preemption.allows(
                            victim.phase().isMasterQueued()
                                    ? VictimStage.DECODE_RESERVED
                                    : VictimStage.DECODE_ENGINE_OWNED),
                    "preemption victim stage is disabled");
        }
    }

    /**
     * Record the single committed Decode-eviction plan.
     */
    private static void recordDecodePlanObservability(
            RequestContext ctx, DecodeEvictionProposal proposal) {
        long totalCost = proposal.diagnosticCost();
        ctx.data().setPlanType("decode_evict");
        ctx.data().setPlanCost(totalCost);
        ctx.data().setVictimCount(proposal.victims().size());
        Logger.debug(
                "[decode-capacity] decode eviction committed: request_id={} priority={} case={} "
                        + "victims={} total_cost={} freed_kv={} worker={}",
                ctx.data().getRequestId(),
                ctx.data().getPriority(),
                proposal.evictionCase(),
                proposal.victims().size(),
                totalCost,
                proposal.freedKvTokens(),
                proposal.endpointId());
    }

    private CompletableFuture<PreemptionResult> startEngineCancelPreemption(
            RequestContext ctx,
            DecodeEvictionProposal proposal,
            DecodeEndpoint endpoint,
            Request request,
            SchedulingMetadata metadata) {
        var command =
                new PreemptionCommand(
                        endpoint,
                        request,
                        metadata,
                        proposal.victims(),
                        50L,
                        preemption.getTimeoutMs(),
                        () ->
                                ctx.data()
                                        .getLifecycle()
                                        .isAdmissionOpen(
                                                request.getRequestId(), ctx.data().getFuture()),
                        "preempted by higher-priority request " + ctx.data().getRequestId());
        report(
                ctx.data().getRequestId(),
                "cancel requests",
                () ->
                        reporter.reportPreemptionCancelRequests(
                                proposal.endpointId(), proposal.victims()));
        return preempt(ctx, command)
                .whenComplete(
                        (result, error) ->
                                report(
                                        ctx.data().getRequestId(),
                                        "engine preemption",
                                        () -> {
                                            if (error != null
                                                    || result == null
                                                    || result.controlFailure()) {
                                                reporter.reportEngineCancel(
                                                        CancelEvent.TIMEOUT,
                                                        proposal.endpointId(),
                                                        ctx.data().getPriority());
                                            } else if (result.committed()) {
                                                reporter.reportPreemptionVictims(
                                                        ctx.data().getRequestId(),
                                                        ctx.data().getPriority(),
                                                        proposal.endpointId(),
                                                        proposal.evictionCase(),
                                                        proposal.victims(),
                                                        true);
                                                recordDecodePlanObservability(ctx, proposal);
                                            }
                                        }));
    }

    private static void report(String requestId, String operation, Runnable metrics) {
        try {
            metrics.run();
        } catch (Throwable failure) {
            Logger.warn(
                    "[decode-capacity] failed to report {}: request_id={}",
                    operation,
                    requestId,
                    failure);
        }
    }

    private DecodeResources.ReservationHandle replaceQueuedDecodeReservations(
            DecodeEndpoint endpoint,
            List<DecodeResources.ReservationHandle> victims,
            Request request,
            int priority) {
        List<AdmissionHandle> claimed = new ArrayList<>(victims.size());
        DecodeResources.ReservationHandle incoming = null;
        try {
            for (DecodeResources.ReservationHandle victim : victims) {
                var owner = requests.ownerOf(victim.requestId());
                AdmissionHandle withdrawal =
                        owner == null
                                ? null
                                : AdmissionHandle.tryBeginPreemption(
                                        owner, endpoint, victim, priority);
                if (withdrawal == null) {
                    return null;
                }
                claimed.add(withdrawal);
            }
            incoming =
                    endpoint.replaceQueuedRequests(
                            victims,
                            request.getRequestId(),
                            request.getHardKvTokens(),
                            request.getExpectedKvTokens(),
                            priority,
                            capacity);
            return incoming;
        } finally {
            Throwable failure = null;
            for (AdmissionHandle withdrawal : claimed) {
                boolean committed = incoming != null;
                failure =
                        Failures.run(
                                failure,
                                () ->
                                        withdrawal.complete(
                                                committed
                                                        ? ContextResult.PublicationResult.WITHDRAWN
                                                        : ContextResult.PublicationResult
                                                                .RESTORED));
            }
            if (failure != null && incoming != null) {
                var exact = incoming;
                failure =
                        Failures.run(
                                failure,
                                () -> {
                                    var released =
                                            endpoint.release(
                                                            exact,
                                                            DecodeResources.ReleaseReason
                                                                    .LOCAL_ROLLBACK)
                                                    .status();
                                    checkState(
                                            released
                                                            == DecodeResources
                                                                    .ReservationReleaseResult
                                                                    .RELEASED
                                                    || released
                                                            == DecodeResources
                                                                    .ReservationReleaseResult.STALE,
                                            "failed withdrawal retained incoming Decode resources");
                                });
            }
            Failures.rethrow(failure, "queued request withdrawal failed");
        }
    }
}
