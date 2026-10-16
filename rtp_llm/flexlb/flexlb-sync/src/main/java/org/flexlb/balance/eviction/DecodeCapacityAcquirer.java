package org.flexlb.balance.eviction;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity;
import org.flexlb.balance.endpoint.DecodeResources.DecodeRequestView;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.balance.preemption.VictimResolution;
import org.flexlb.balance.scheduler.AbstractRequestScheduler;
import org.flexlb.balance.scheduler.AdmissionHandle;
import org.flexlb.balance.scheduler.CancelReason;
import org.flexlb.balance.scheduler.PreemptionRegistration;
import org.flexlb.balance.scheduler.QueuedRequestScheduler;
import org.flexlb.balance.scheduler.RequestContext;
import org.flexlb.balance.scheduler.RequestRepository;
import org.flexlb.balance.scheduler.RequestTaskScheduler;
import org.flexlb.config.ConfigService;
import org.flexlb.config.PreemptionConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.enums.DecodeTaskPhase;
import org.flexlb.service.monitor.RequestSchedulerReporter.CancelEvent;
import org.flexlb.service.monitor.RequestSchedulerReporter.EvictionEvent;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;
import org.flexlb.util.PriorityNormalizer;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

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
import javax.annotation.PreDestroy;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

/**
 * Obtains capacity at one selected Decode generation through local withdrawal or Engine cancellation.
 *
 * <p>The scheduler supplies the blocked endpoint and consumes one result.  This class
 * owns the two-phase protocol, token fencing and exactly-once child settlement.
 * Engine acknowledgement is only control evidence; the canonical victim
 * resolution transaction may complete before or after that acknowledgement.</p>
 */
@Component
public class DecodeCapacityAcquirer {

    public record PreemptionResult(
            DecodeResources.ReservationHandle reservation, boolean controlFailure, String detail) {
        public boolean committed() { return reservation != null; }

        public PreemptionResult {
            checkArgument(reservation == null || !controlFailure, "committed preemption cannot be a control failure");
            Objects.requireNonNull(detail, "detail");
        }
    }

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
            checkArgument(endpoint != null && victims != null && !victims.isEmpty(),
                    "endpoint and victims are required");
            victims = List.copyOf(victims);
            Objects.requireNonNull(request, "request");
            Objects.requireNonNull(metadata, "metadata");
            checkArgument(request.getRequestId() > 0L, "incoming request id must be positive");
            Set<Long> victimIds = new LinkedHashSet<>();
            for (DecodeRequestView victim : victims) {
                checkArgument(victim.requestId() > 0L && victim.reservationToken() > 0L,
                        "victim requestId and reservation token must be positive");
                checkArgument(victim.phase() != null && victim.phase().requiresEngineCancel(),
                        "capacity acquisition accepts only Engine-Cancel victims");
                checkArgument(victimIds.add(victim.requestId()), "duplicate victim %s", victim.requestId());
            }
            checkArgument(admissionOpen != null, "admission gate is required");
        }
    }

    private final RequestSchedulerReporter reporter;
    private volatile boolean shutdown;
    private final EngineGrpcClient engineClient;
    private final RequestRepository requests;
    private final RequestTaskScheduler runtime;
    private final AtomicLong tokenSequence = new AtomicLong(1);

    private final AdmissionCapacity capacity;
    private final PreemptionConfig preemption;

    @Autowired
    public DecodeCapacityAcquirer(EngineGrpcClient engineClient, RequestRepository requests,
                                  RequestTaskScheduler runtime, RequestSchedulerReporter reporter,
                                  ConfigService configService, AdmissionCapacity capacity) {
        this.reporter = Objects.requireNonNull(reporter, "reporter");
        this.engineClient = Objects.requireNonNull(engineClient, "engineClient");
        this.requests = Objects.requireNonNull(requests, "requests");
        this.runtime = Objects.requireNonNull(runtime, "runtime");
        this.capacity = Objects.requireNonNull(capacity, "capacity");
        var config = configService.loadBalanceConfig();
        var policy = config.isQueue() && config.isPriorityOrdering()
                ? config.priorityOrdering().getPreemption() : null;
        if (policy == null) {
            this.preemption = null;
        } else {
            this.preemption = new PreemptionConfig();
            this.preemption.setTimeoutMs(policy.getTimeoutMs());
            this.preemption.setAllowedVictimStages(policy.getAllowedVictimStages() == null
                    ? java.util.Set.of() : java.util.Set.copyOf(policy.getAllowedVictimStages()));
        }
    }

    CompletableFuture<PreemptionResult> preempt(
            RequestContext context, PreemptionCommand command) {
        Objects.requireNonNull(context, "context");
        long token = nextToken();
        List<DecodeResources.ReservationHandle> victimReservations =
                new ArrayList<>(command.victims().size());
        long endpointGenerationId = command.endpoint().getStatus().getGenerationId();
        AttemptCapability capability = new AttemptCapability(command, token);
        try {
            for (DecodeRequestView victim : command.victims()) {
                var owner = requests.ownerOf(victim.requestId());
                if (owner == null) {
                    return CompletableFuture.completedFuture(capability.abort(true,
                            "cancel_owner_missing:" + victim.requestId()));
                }
                var reservation = new DecodeResources.ReservationHandle(endpointGenerationId,
                        victim.requestId(), victim.reservationToken());
                Optional<PreemptionRegistration> claimAttempt = owner.tryClaim(reservation, token, command.detail());
                if (claimAttempt.isEmpty()) {
                    return CompletableFuture.completedFuture(capability.abort(
                            false, "victim_inflight_gone"));
                }
                PreemptionRegistration claim = claimAttempt.get();
                ClaimedVictim owned = new ClaimedVictim(reservation, claim);
                capability.claims.add(owned);
                victimReservations.add(reservation);
                if (claim.owner().data().getRequestId() != victim.requestId()
                        || claim.attemptToken() != token) {
                    return CompletableFuture.completedFuture(capability.abort(
                            true,
                            "lifecycle_returned_mismatched_claim:" + owned.requestId()));
                }
            }

            DecodeResources.PreemptionBeginResult begin =
                    command.endpoint().beginPreemption(
                            token,
                            victimReservations,
                            command.request().getRequestId(),
                            command.request().getHardKvTokens(),
                            command.request().getExpectedKvTokens(),
                            command.metadata().priority(),
                            capacity);
            if (begin != DecodeResources.PreemptionBeginResult.SUCCESS) {
                return CompletableFuture.completedFuture(capability.abort(
                        begin == DecodeResources.PreemptionBeginResult.ENDPOINT_RETIRED,
                        "begin_" + begin.name().toLowerCase()));
            }
            capability.endpointBegun = true;
            for (ClaimedVictim owned : capability.claims) {
                if (!owned.claim.owner().data().getScheduler().onPreemption(owned.claim,
                            new AbstractRequestScheduler.PreemptionEvent(PreemptionCancelPhase.CANCEL_IN_FLIGHT, false, null, null, null))) {
                    return CompletableFuture.completedFuture(capability.abort(
                            true,
                            "inflight_cancel_linearization_failed:"
                                    + owned.requestId()));
                }
            }
            for (ClaimedVictim owned : capability.claims) {
                owned.acknowledgement = capability.outboundStarted(owned)
                        ? cancel(owned, command.cancelAckTimeoutMs())
                        : CompletableFuture.completedFuture(EngineGrpcClient.CancelReply.FAILED);
            }

            CompletableFuture<PreemptionResult> protocol = CompletableFuture.allOf(
                            capability.claims.stream().map(owned -> owned.acknowledgement)
                                    .toArray(CompletableFuture[]::new))
                    .thenCompose(ignored -> handleAcknowledgements(context, capability));
            return protocol.handle((result, failure) -> {
                if (failure != null) {
                    return capability.abort(
                            true,
                            "coordinator_continuation_failed:"
                                    + failureDetail(failure));
                }
                return result;
            });
        } catch (RuntimeException | Error failure) {
            return CompletableFuture.completedFuture(capability.abort(
                    true,
                    "coordinator_setup_failed:" + failureDetail(failure)));
        }
    }

    private CompletableFuture<PreemptionResult> handleAcknowledgements(
            RequestContext context, AttemptCapability capability) {
        PreemptionCommand command = capability.command;
        // A transport-unknown ACK is not a negative acknowledgement: the
        // Prefill may have installed the intent before the reply was lost.
        // Such a child therefore waits for the canonical victim resolution
        // transaction exactly like an ACCEPTED child.
        List<ClaimedVictim> pendingResolutions = new ArrayList<>();
        boolean hasNotFound = false;

        for (ClaimedVictim owned : capability.claims) {
            if (owned.requestResolved()) {
                continue;
            }
            EngineGrpcClient.CancelReply outcome = owned.acknowledgement.join();
            switch (outcome) {
                case ACCEPTED, REQUEST_FENCED -> {
                    boolean transitioned =
                            owned.claim.owner().data().getScheduler().onPreemption(owned.claim,
                            new AbstractRequestScheduler.PreemptionEvent(PreemptionCancelPhase.CANCEL_REQUESTED, false, null, null, null));
                    if (transitioned) {
                        capability.transferred(owned);
                    } else {
                        capability.transferUnknown(owned);
                    }
                    pendingResolutions.add(owned);
                }
                case NOT_FOUND -> {
                    owned.claim.owner().data().getScheduler().onPreemption(owned.claim,
                            new AbstractRequestScheduler.PreemptionEvent(PreemptionCancelPhase.NOT_FOUND_STALE, false, null, null, null));
                    capability.transferred(owned);
                    if (!owned.requestResolved()) {
                        hasNotFound = true;
                    }
                }
                case REQUEST_CLEANED -> {
                    // Only downstream cleanup proof can release Decode capacity without a WorkerStatus terminal.
                    if (owned.claim.owner().data().getScheduler().onPreemption(owned.claim,
                            new AbstractRequestScheduler.PreemptionEvent(null, false, command.endpoint(), owned.reservation, command.detail()))) {
                        capability.transferred(owned);
                        pendingResolutions.add(owned);
                    }
                }
                case FAILED, UNSUPPORTED -> {
                    capability.transferUnknown(owned);
                    pendingResolutions.add(owned);
                }
            }
        }

        if (pendingResolutions.isEmpty()) {
            return CompletableFuture.completedFuture(
                    capability.finish(hasNotFound, capability.allVictimsResolved()));
        }
        // The completion budget begins only after the ACK phase has ended; a
        // 40ms ACK followed by a 100ms cleanup therefore gets the full cleanup
        // window rather than sharing one 50ms deadline.
        CompletableFuture<?>[] resolutions = pendingResolutions.stream()
                .map(pending -> runtime.signals().preemption(pending.claim)
                        .handle((resolution, failure) -> null).toCompletableFuture())
                .toArray(CompletableFuture<?>[]::new);
        final boolean ackNotFound = hasNotFound;
        // Only this aggregate wait expires. Canonical request resolutions remain
        // live so late worker facts can still settle their exact claims.
        CompletableFuture<Boolean> settlement = CompletableFuture.allOf(resolutions)
                .handle((ignored, failure) -> capability.allVictimsResolved());
        // The timer only expires the wait. Endpoint/request settlement runs in the
        // background request pool and remains part of shutdown drain.
        RequestTaskScheduler.Cancellation deadline = runtime.scheduleTimeout(() -> settlement.complete(false),
                Math.max(1, command.preemptionTimeoutMs()));
        settlement.whenComplete((unused, failure) -> deadline.cancel());
        // First completion fixes eligibility, even if the deadline fires before
        // this continuation is registered. Late facts still settle victim resources.
        return settlement.thenApplyAsync(resolved -> capability.finish(ackNotFound, resolved),
                work -> runtime.executeRequestWork(context, work));
    }

    private CompletableFuture<EngineGrpcClient.CancelReply> cancel(
            ClaimedVictim victim,
            long timeoutMs) {
        try {
            var target = victim.claim.cancelTarget();
            if (target == null) { return CompletableFuture.completedFuture(EngineGrpcClient.CancelReply.FAILED); }
            CompletableFuture<EngineGrpcClient.CancelReply> stage = engineClient.cancelAsync(
                    target.prefillIp(), target.prefillGrpcPort(),
                    CancelReason.PRIORITY_PREEMPTED.toRequest(victim.requestId()), timeoutMs);
            if (stage == null) {
                return CompletableFuture.completedFuture(
                        EngineGrpcClient.CancelReply.FAILED);
            }
            return stage.handle((outcome, failure) -> failure == null
                            && outcome != null
                    ? outcome : EngineGrpcClient.CancelReply.FAILED);
        } catch (RuntimeException | Error failure) {
            // The capability marked this victim OUTBOUND before invocation,
            // so close conservatively transfers its exact claims to UNKNOWN.
            return CompletableFuture.completedFuture(
                    EngineGrpcClient.CancelReply.FAILED);
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
        private final DecodeResources.ReservationHandle reservation;
        private final PreemptionRegistration claim;
        private CompletableFuture<EngineGrpcClient.CancelReply> acknowledgement;
        private volatile ClaimDisposition disposition =
                ClaimDisposition.RELEASABLE;

        private ClaimedVictim(
                DecodeResources.ReservationHandle reservation,
                PreemptionRegistration claim) {
            this.reservation = reservation;
            this.claim = claim;
        }

        private long requestId() {
            return reservation.requestId();
        }

        private boolean requestResolved() {
            VictimResolution resolution = runtime.signals().resolved(claim);
            return resolution != null && resolution.requestId() == requestId();
        }
    }

    /**
     * The one owner of endpoint admission plus every exact RequestContext claim.
     * A non-committed close is total: uncertain outbound claims transfer to
     * reconciliation, the incoming endpoint reservation aborts, and only
     * claims which never crossed an outbound boundary are released.
     */
    private final class AttemptCapability implements AutoCloseable {
        private final PreemptionCommand command;
        private final long token;
        private final List<ClaimedVictim> claims = new ArrayList<>();
        private boolean endpointBegun;
        private boolean closed;
        private String cleanupFailure;

        private AttemptCapability(
                PreemptionCommand command, long token) {
            this.command = command;
            this.token = token;
        }

        private synchronized boolean outboundStarted(ClaimedVictim owned) {
            if (owned.requestResolved()) {
                return false;
            }
            if (owned.disposition != ClaimDisposition.RELEASABLE) {
                throw new IllegalStateException(
                        "Cancel outbound ownership changed request_id="
                                + owned.requestId());
            }
            // Install the conservative resource hold before RPC invocation. No ACK changes this fact.
            if (!command.endpoint().updatePreemption(token,
                    DecodeResources.PreemptionUpdate.handedOff(owned.reservation)).status().applied()) { return false; }
            owned.disposition = ClaimDisposition.OUTBOUND;
            return true;
        }

        private synchronized boolean allVictimsResolved() {
            if (claims.isEmpty()) {
                return false;
            }
            return claims.stream().allMatch(
                    ClaimedVictim::requestResolved);
        }

        private synchronized void transferred(ClaimedVictim owned) {
            owned.disposition = ClaimDisposition.TRANSFERRED;
        }

        private void transferUnknown(ClaimedVictim owned) {
            if (owned.disposition != ClaimDisposition.OUTBOUND || owned.requestResolved()) {
                return;
            }
            owned.claim.owner().data().getScheduler().onPreemption(owned.claim,
                            new AbstractRequestScheduler.PreemptionEvent(PreemptionCancelPhase.CANCEL_UNKNOWN, false, null, null, null));
            transferred(owned);
        }

        private PreemptionResult finish(boolean hasNotFound, boolean victimsResolved) {
            DecodeResources.ReservationHandle incoming = victimsResolved
                    && command.admissionOpen().getAsBoolean()
                    ? command.endpoint().commitPreemption(token) : null;
            if (incoming != null) {
                beginClose();
                return new PreemptionResult(incoming, false, "committed");
            }
            boolean cleanSingleNotFound = hasNotFound
                    && claims.size() == 1
                    && !claims.get(0).requestResolved();
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
                    null, controlFailure, resultDetail);
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
                    cleanup("transfer_unknown:" + owned.requestId(), () -> transferUnknown(owned));
                }
            }
            if (endpointBegun) {
                cleanup("endpoint_abort", () -> command.endpoint().abortPreemption(token));
            }
            for (ClaimedVictim owned : claims) {
                if (owned.disposition == ClaimDisposition.RELEASABLE && !owned.requestResolved()) {
                    cleanup("release_claim:" + owned.requestId(), () -> owned.claim.owner().data().getScheduler().onPreemption(owned.claim,
                            new AbstractRequestScheduler.PreemptionEvent(null, true, null, null, null)));
                }
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

    @PreDestroy
    public void shutdown() {
        shutdown = true;
    }

    /**
     * The caller retains its admission handle throughout this operation and consumes
     * any returned reservation. Null means no takeover occurred, including a local
     * victim conflict; an Engine attempt always returns its eventual terminal result.
     */
    public CompletableFuture<PreemptionResult> tryReclaim(
            RequestContext ctx, WorkerEndpoint blockedEndpoint) {
        Request request = ctx.data().getRequest();
        SchedulingMetadata metadata = ctx.data().getSchedulingMetadata();
        if (shutdown || ctx.data().getFuture().isDone()
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
        if (proposal == null || !ctx.data().getScheduler().isAdmissionOpen(request.getRequestId(), ctx.data().getFuture())) {
            return null;
        }
        if (proposal.requiresEngineCancel()) {
            return startEngineCancelPreemption(ctx, proposal, decodeEndpoint, request, metadata);
        }
        List<DecodeResources.ReservationHandle> victims = new ArrayList<>(proposal.victims().size());
        for (DecodeRequestView victim : proposal.victims()) {
            victims.add(new DecodeResources.ReservationHandle(
                    decodeEndpoint.getStatus().getGenerationId(), victim.requestId(), victim.reservationToken()));
        }
        DecodeResources.ReservationHandle incoming = replaceQueuedDecodeReservations(
                decodeEndpoint, victims, request, metadata.priority());
        if (incoming == null) {
            reportEviction(EvictionEvent.COMMIT, ctx.data().getPriority(), ctx.data().getRequestId(), proposal.evictionCase(), "conflict");
            return null;
        }
        report(ctx.data().getRequestId(), "local preemption", () -> {
            for (DecodeRequestView victim : proposal.victims()) {
                reportRequeuedVictim(ctx, victim, proposal);
            }
            reportEviction(EvictionEvent.COMMIT, ctx.data().getPriority(), ctx.data().getRequestId(), proposal.evictionCase(), "success");
            recordDecodePlanObservability(ctx, proposal);
        });
        return CompletableFuture.completedFuture(new PreemptionResult(incoming, false, "committed"));
    }

    /** Metrics observe plans and commits without owning the reservation transaction. */
    private void reportEviction(EvictionEvent event, int priority, long requestId,
                                String evictionCase, String outcome) {
        report(requestId, event == EvictionEvent.PLAN ? "eviction plan" : "eviction commit",
                () -> reporter.reportEviction(event, priority, evictionCase, outcome));
    }

    // ==================== Decode eviction ====================
    /**
     * Build one side-effect-free plan from one exact cluster snapshot.
     */
    private DecodeEvictionProposal planDecodeEviction(
            Request request, SchedulingMetadata metadata,
            DecodeEndpoint selectedEndpoint) {
        DecodeResources.ResourceSnapshot selected = selectedEndpoint.resourceSnapshot();
        if (selectedEndpoint.isRetired()) {
            return null;
        }
        Map<String, String> failures = new HashMap<>();
        var decision = EvictionPlanner.planDecode(request, metadata.priority(), capacity, selected, preemption, failures);
        if (decision.deficit().fits()) {
            return null;
        }
        DecodeEvictionProposal proposal = decision.proposal();
        if (proposal == null) {
            reportEviction(EvictionEvent.PLAN, metadata.priority(), request.getRequestId(),
                    decision.evictionCase(), "infeasible");
            Logger.debug(
                    "[decode-capacity] Decode eviction infeasible:"
                            + " request_id={} priority={} worker={} reasons={}",
                    request.getRequestId(), metadata.priority(), selected.routing().address(), failures);
            return null;
        }
        reportEviction(EvictionEvent.PLAN, metadata.priority(), request.getRequestId(),
                proposal.evictionCase(), "feasible");
        return proposal;
    }

    /**
     * Observability only: the request's scheduler completes withdrawal and requeue.
     */
    private void reportRequeuedVictim(RequestContext ctx, DecodeRequestView victim,
                                     DecodeEvictionProposal proposal) {
        String stage = "decode_reserved";
        report(ctx.data().getRequestId(), "requeued victim", () -> {
            reporter.reportVictim(victim.priority(), ctx.data().getPriority(),
                    stage, proposal.evictionCase());
            reporter.reportVictimKvTokens(
                    victim.priority(), stage, victim.kvTokens());
        });
        Logger.debug(
                "[decode-capacity] decode victim preempted: victim_id={} victim_priority={}"
                    + " stage={} outcome={} kv_tokens={} incoming_id={} incoming_priority={}"
                    + " worker={}",
                victim.requestId(),
                victim.priority(),
                stage,
                "requeued",
                victim.kvTokens(),
                ctx.data().getRequestId(),
                ctx.data().getPriority(),
                proposal.endpointId());
    }

    /**
     * Record the single committed Decode-eviction plan.
     */
    private static void recordDecodePlanObservability(RequestContext ctx,
                                                      DecodeEvictionProposal proposal) {
        long totalCost = proposal.priorityHarmProfile().totalCost();
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
            RequestContext ctx, DecodeEvictionProposal proposal,
            DecodeEndpoint endpoint, Request request, SchedulingMetadata metadata) {
        var command = new PreemptionCommand(
                endpoint, request, metadata, proposal.victims(), 50L,
                preemption.getTimeoutMs(),
                () -> ctx.data().getScheduler().isAdmissionOpen(request.getRequestId(), ctx.data().getFuture()),
                "preempted by higher-priority request " + ctx.data().getRequestId());
        reportCancelRequests(ctx, proposal);
        return preempt(ctx, command).whenComplete((result, error) ->
                report(ctx.data().getRequestId(), "engine preemption", () -> {
                    if (error != null || result == null || result.controlFailure()) {
                        reportCancelTimeout(ctx, proposal.endpointId());
                    } else if (result.committed()) {
                        reportCommittedEnginePreemption(ctx, proposal);
                        recordDecodePlanObservability(ctx, proposal);
                    }
                }));
    }

    /**
     * Metrics never participate in the committed reservation handoff.
     */
    private void reportCommittedEnginePreemption(
            RequestContext ctx, DecodeEvictionProposal proposal) {
        report(ctx.data().getRequestId(), "committed preemption", () -> {
            for (DecodeRequestView victim : proposal.victims()) {
                String stage = victim.phase() == DecodeTaskPhase.RUNNING
                        ? "decode_running" : "decode_cancel";
                reporter.reportVictim(victim.priority(), ctx.data().getPriority(),
                        stage, proposal.evictionCase());
                reporter.reportVictimKvTokens(
                        victim.priority(), stage, victim.kvTokens());
                reporter.reportEngineCancel(CancelEvent.CONFIRM,
                        proposal.endpointId(), victim.priority());
            }
            reporter.reportEviction(EvictionEvent.COMMIT, ctx.data().getPriority(),
                    proposal.evictionCase(), "success");
        });
    }

    private void reportCancelRequests(RequestContext ctx,
                                      DecodeEvictionProposal proposal) {
        report(ctx.data().getRequestId(), "cancel requests", () -> {
            for (DecodeRequestView victim : proposal.victims()) {
                reporter.reportEngineCancel(CancelEvent.REQUEST,
                        proposal.endpointId(), victim.priority());
                reporter.reportCancel(
                        victim.priority(), "PRIORITY_PREEMPTED");
            }
        });
    }

    private void reportCancelTimeout(RequestContext ctx, String endpointId) {
        report(ctx.data().getRequestId(), "cancel timeout", () -> {
            reporter.reportEngineCancel(CancelEvent.TIMEOUT, endpointId, ctx.data().getPriority());
        });
    }

    private static void report(long requestId, String operation, Runnable metrics) {
        try {
            metrics.run();
        } catch (Throwable failure) {
            Logger.warn("[decode-capacity] failed to report {}: request_id={}",
                    operation, requestId, failure);
        }
    }

    private DecodeResources.ReservationHandle replaceQueuedDecodeReservations(
            DecodeEndpoint endpoint, List<DecodeResources.ReservationHandle> victims,
            Request request, int priority) {
        List<AdmissionHandle> claimed = new ArrayList<>(victims.size());
        DecodeResources.ReservationHandle incoming = null;
        try {
            for (DecodeResources.ReservationHandle victim : victims) {
                var owner = requests.ownerOf(victim.requestId());
                AdmissionHandle withdrawal = owner instanceof QueuedRequestScheduler queue ? queue.claimQueuedRoute(endpoint, victim, priority) : null;
                if (withdrawal == null) {
                    return null;
                }
                claimed.add(withdrawal);
            }
            incoming = endpoint.replaceQueuedRequests(victims, request.getRequestId(), request.getHardKvTokens(),
                    request.getExpectedKvTokens(), priority, capacity);
            return incoming;
        } finally {
            Throwable failure = null;
            for (AdmissionHandle withdrawal : claimed) {
                boolean committed = incoming != null;
                failure = Failures.run(failure, () -> ((QueuedRequestScheduler) withdrawal.owner().data().getScheduler()).completeWithdrawal(withdrawal, committed));
            }
            if (failure != null) {
                if (incoming != null) {
                    endpoint.release(incoming, DecodeResources.ReleaseReason.LOCAL_ROLLBACK).status();
                }
                Failures.rethrow(failure, "request cleanup failed");
            }
        }
    }

}
