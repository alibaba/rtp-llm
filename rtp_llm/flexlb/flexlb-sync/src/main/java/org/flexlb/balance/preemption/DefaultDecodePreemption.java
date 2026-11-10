package org.flexlb.balance.preemption;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity;
import org.flexlb.balance.endpoint.DecodeResources.DecodeRequestView;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.preemption.DecodePreemptionPolicy.DecodePreemptionPlan;
import org.flexlb.balance.scheduler.AbstractRequestScheduler;
import org.flexlb.balance.scheduler.ContextResult;
import org.flexlb.balance.scheduler.DeliveryCoordinator;
import org.flexlb.balance.scheduler.PreemptionRegistration;
import org.flexlb.balance.scheduler.RequestContext;
import org.flexlb.balance.scheduler.RequestRepository;
import org.flexlb.balance.scheduler.RequestTaskExecutor;
import org.flexlb.balance.scheduler.RouteResult;
import org.flexlb.balance.scheduler.RoutingContext;
import org.flexlb.balance.scheduler.TaskTimer;
import org.flexlb.config.ConfigService;
import org.flexlb.config.PreemptionConfig;
import org.flexlb.config.PreemptionStage;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter.CancelEvent;
import org.flexlb.service.monitor.RequestSchedulerReporter.PreemptionEvent;
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
 * Engine acknowledgement does not release capacity: admission waits for the endpoint to record
 * release of every selected reservation.</p>
 */
public class DefaultDecodePreemption implements DecodePreemption {
    private static final PreemptionCompletion.PreemptionEvent START_CANCEL =
            new PreemptionCompletion.PreemptionEvent(
                    PreemptionCancelPhase.CANCEL_IN_FLIGHT, false, null, null, null);

    private final RequestSchedulerReporter reporter;
    private final PreemptionCompletion completion = new PreemptionCompletion();

    private volatile boolean shutdown;

    private final java.util.concurrent.atomic.AtomicInteger activeWork =
            new java.util.concurrent.atomic.AtomicInteger();
    private final ThreadLocal<Boolean> acquiring = new ThreadLocal<>();
    private final DeliveryCoordinator dispatcher;
    private final RequestRepository requests;
    private final RequestTaskExecutor executor;
    private final TaskTimer timer;
    private final AtomicLong tokenSequence = new AtomicLong(1);
    private final DecodePreemptionPolicy selectionPolicy;
    private final AdmissionCapacity capacity;
    private final PreemptionConfig preemption;

    public DefaultDecodePreemption(
            DeliveryCoordinator dispatcher,
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
            this.preemption.setAllowedPreemptionStages(
                    policy.getAllowedPreemptionStages() == null
                            ? java.util.Set.of()
                            : java.util.Set.copyOf(policy.getAllowedPreemptionStages()));
        }
    }

    @Override
    @PreDestroy
    public void close() {
        stopAdmission();
        checkState(awaitIdle(Long.MAX_VALUE), "preemption cannot wait for its own acquisition");
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

    /**
     * Registers preemption against the current request and its exact Decode reservation.
     * This records the cancellation target without sending a cancel request or releasing capacity.
     */
    public Optional<PreemptionRegistration> tryRegisterPreemption(
            DecodeResources.RequestReservation reservation, long preemptionId) {
        var ctx = requests.findActive(reservation.requestId());
        if (ctx == null || ctx.data().getRequestScheduler() == null) {
            return Optional.empty();
        }
        synchronized (ctx) {
            var route = ctx.snapshot().route();
            if (route == null) {
                return Optional.empty();
            }
            var prefill =
                    route.role(org.flexlb.dao.route.RoleType.PREFILL)
                            .toServerStatus(reservation.requestId());
            var target = new CancelTarget(prefill.getServerIp(), prefill.getGrpcPort());
            var registration = ctx.beginPreemption(reservation, preemptionId, target);
            return Optional.ofNullable(registration);
        }
    }

    /**
     * Checks and starts preemption of an unstarted route; the caller must complete or restore it.
     */
    public RoutingContext beginRoutePreemption(
            DecodeEndpoint endpoint,
            DecodeResources.RequestReservation reservationToPreempt,
            int incomingPriority) {
        if (requests.isClosed()) {
            return null;
        }
        RequestContext context = requests.findActive(reservationToPreempt.requestId());
        if (context == null
                || context.data().getRequestScheduler() == null
                || !context.data().getRequestScheduler().allowsRoutePreemption()) {
            return null;
        }
        synchronized (context) {
            RouteResult item = context.snapshot().route();
            // Context serializes the exact queue claim with delivery permit transfer.
            if (item == null
                    || item.priority() >= incomingPriority
                    || item.decodeEndpoint() != endpoint
                    || !Objects.equals(item.decodeReservation(), reservationToPreempt)
                    || !requests.isCurrent(context)
                    || context.data().getFuture().isDone()
                    || item.requestExpired(System.currentTimeMillis())
                    || !endpoint.ownership(reservationToPreempt).withdrawable()) {
                return null;
            }
            return context.beginRoutePreemption(item);
        }
    }

    @Override
    public CompletableFuture<PreemptionResult> tryPreempt(
            RequestContext ctx, WorkerEndpoint endpoint) {
        activeWork.incrementAndGet();
        var previous = acquiring.get();
        acquiring.set(Boolean.TRUE);
        try {
            return startPreemption(ctx, endpoint);
        } finally {
            if (previous == null) acquiring.remove();
            else acquiring.set(previous);
            activeWork.decrementAndGet();
        }
    }

    protected PreemptionCompletion completion() {
        return completion;
    }

    CompletableFuture<PreemptionResult> preempt(RequestContext context, PreemptionCommand command) {
        Objects.requireNonNull(context, "context");
        long token = nextToken();
        List<DecodeResources.RequestReservation> reservationsToPreempt =
                new ArrayList<>(command.requestsToPreempt().size());
        long endpointGenerationId = command.endpoint().getStatus().getGenerationId();
        Attempt attempt = new Attempt(command, token);
        try {
            for (DecodeRequestView requestToPreempt : command.requestsToPreempt()) {
                var owner = requests.ownerOf(requestToPreempt.requestId());
                if (owner == null) {
                    return CompletableFuture.completedFuture(
                            attempt.abort(
                                    true, "cancel_owner_missing:" + requestToPreempt.requestId()));
                }
                var reservation =
                        new DecodeResources.RequestReservation(
                                endpointGenerationId,
                                requestToPreempt.requestId(),
                                requestToPreempt.reservationToken());
                Optional<PreemptionRegistration> claimAttempt =
                        tryRegisterPreemption(reservation, token);
                if (claimAttempt.isEmpty()) {
                    return CompletableFuture.completedFuture(
                            attempt.abort(false, "victim_inflight_gone"));
                }
                PreemptionRegistration claim = claimAttempt.get();
                PreemptionRequest owned = new PreemptionRequest(claim, command.endpoint());
                attempt.claims.add(owned);
                reservationsToPreempt.add(reservation);
                if (!java.util.Objects.equals(
                                claim.owner().data().getRequestId(), requestToPreempt.requestId())
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
                                    reservationsToPreempt,
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
            for (PreemptionRequest owned : attempt.claims) {
                if (!owned.resourcesReleased()
                        && !completion().applyEvent(owned.claim, START_CANCEL)
                        && !owned.resourcesReleased()) {
                    return CompletableFuture.completedFuture(
                            attempt.abort(
                                    true,
                                    "inflight_cancel_linearization_failed:" + owned.requestId()));
                }
            }
            for (PreemptionRequest owned : attempt.claims) {
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
        // Both UNKNOWN and ACCEPTED therefore wait for the endpoint to release capacity.
        int waiting = 0;
        boolean hasNotFound = false;

        for (PreemptionRequest owned : attempt.claims) {
            if (owned.resourcesReleased()) {
                continue;
            }
            var progress = owned.acknowledgement.join();
            if (progress == PreemptionRegistration.Reply.UNKNOWN) {
                attempt.transferUnknown(owned);
            } else {
                attempt.transferred(owned);
            }
            if (progress == PreemptionRegistration.Reply.NOT_FOUND) {
                hasNotFound |= !owned.resourcesReleased();
            } else {
                waiting++;
            }
        }

        if (waiting == 0) {
            attempt.settlement.complete(attempt.allResourcesReleased());
            return CompletableFuture.completedFuture(
                    attempt.finish(hasNotFound, attempt.settlement.getNow(false)));
        }
        if (hasNotFound) {
            attempt.rejectWhenOnlyNotFoundRemain();
        }
        final boolean ackNotFound = hasNotFound;
        // Late facts still settle requests selected for preemption; they cannot change the frozen
        // admission result.
        return attempt.settlement.thenApplyAsync(
                resolved -> attempt.finish(ackNotFound, resolved),
                work -> executor.execute(context, work));
    }

    private CompletableFuture<PreemptionRegistration.Reply> cancel(
            PreemptionRequest requestToPreempt, PreemptionCommand command, long expiresAt) {
        try {
            long remaining =
                    java.util.concurrent.TimeUnit.NANOSECONDS.toMillis(
                            expiresAt - System.nanoTime());
            if (remaining <= 0) {
                return CompletableFuture.completedFuture(PreemptionRegistration.Reply.UNKNOWN);
            }
            return completion()
                    .cancelRequest(
                            dispatcher,
                            timer,
                            requestToPreempt.claim,
                            command.endpoint(),
                            command.detail(),
                            Math.min(command.cancelAckTimeoutMs(), remaining));
        } catch (RuntimeException | Error failure) {
            // The attempt marked this request selected for preemption OUTBOUND before invocation,
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

    private long nextToken() {
        long token = tokenSequence.getAndIncrement();
        checkState(token > 0, "preemption attempt token exhausted");
        return token;
    }

    /**
     * Starts one Decode preemption transaction. The caller owns the returned reservation and must
     * either attach it to its placement attempt or release it.
     */
    private CompletableFuture<PreemptionResult> startPreemption(
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
                || (!preemption.allows(PreemptionStage.DECODE_RESERVED)
                        && !preemption.allows(PreemptionStage.DECODE_ENGINE_OWNED))) {
            return null;
        }
        DecodePreemptionPlan proposal = planDecodePreemption(request, metadata, decodeEndpoint);
        if (proposal == null || !ctx.data().getRequestScheduler().isAdmissionOpen(ctx)) {
            return null;
        }
        if (proposal.requiresEngineCancel()) {
            return startEngineCancelPreemption(ctx, proposal, decodeEndpoint, request, metadata);
        }
        List<DecodeResources.RequestReservation> requestsToPreempt =
                new ArrayList<>(proposal.requestsToPreempt().size());
        for (DecodeRequestView requestToPreempt : proposal.requestsToPreempt()) {
            requestsToPreempt.add(
                    new DecodeResources.RequestReservation(
                            decodeEndpoint.getStatus().getGenerationId(),
                            requestToPreempt.requestId(),
                            requestToPreempt.reservationToken()));
        }
        DecodeResources.RequestReservation incoming =
                replaceQueuedDecodeReservations(
                        decodeEndpoint, requestsToPreempt, request, metadata.priority());
        if (incoming == null) {
            reportPreemption(
                    PreemptionEvent.COMMIT,
                    ctx.data().getPriority(),
                    ctx.data().getRequestId(),
                    proposal.preemptionReason(),
                    "conflict");
            return null;
        }
        report(
                ctx.data().getRequestId(),
                "local preemption",
                () -> {
                    reporter.reportPreemptedRequests(
                            ctx.data().getRequestId(),
                            ctx.data().getPriority(),
                            proposal.endpointId(),
                            proposal.preemptionReason(),
                            proposal.requestsToPreempt(),
                            false);
                    recordDecodePlanObservability(ctx, proposal);
                });
        return CompletableFuture.completedFuture(
                new PreemptionResult(incoming, false, "committed"));
    }

    // ==================== Decode preemption ====================
    /**
     * Build one side-effect-free plan from one exact cluster snapshot.
     */
    private DecodePreemptionPlan planDecodePreemption(
            Request request, SchedulingMetadata metadata, DecodeEndpoint selectedEndpoint) {
        DecodeResources.ResourceSnapshot selected = selectedEndpoint.resourceSnapshot();
        if (!selectedEndpoint.acceptsNewRequests()) {
            return null;
        }
        Map<String, String> failures = new HashMap<>();
        var decision =
                selectionPolicy.planDecode(
                        request, metadata.priority(), capacity, selected, preemption, failures);
        if (decision.deficit().fits()) {
            return null;
        }
        DecodePreemptionPlan proposal = decision.proposal();
        if (proposal != null) {
            validateProposal(proposal, selected, request.getRequestId(), metadata.priority());
        }
        if (proposal == null) {
            reportPreemption(
                    PreemptionEvent.PLAN,
                    metadata.priority(),
                    request.getRequestId(),
                    decision.preemptionReason(),
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
        reportPreemption(
                PreemptionEvent.PLAN,
                metadata.priority(),
                request.getRequestId(),
                proposal.preemptionReason(),
                "feasible");
        return proposal;
    }

    /**
     * Validate the policy's authority; exact live token/capacity checks remain in the ledger.
     */
    private void validateProposal(
            DecodePreemptionPlan proposal,
            DecodeResources.ResourceSnapshot snapshot,
            String incomingId,
            int incomingPriority) {
        checkArgument(
                Objects.equals(proposal.endpointId(), snapshot.routing().address()),
                "preemption proposal must target the selected endpoint");
        checkArgument(
                !proposal.requestsToPreempt().isEmpty(), "preemption proposal requires victims");
        Set<String> ids = new LinkedHashSet<>();
        for (DecodeRequestView requestToPreempt : proposal.requestsToPreempt()) {
            checkArgument(
                    requestToPreempt.equals(snapshot.requests().get(requestToPreempt.requestId())),
                    "preemption victim must match the supplied snapshot");
            checkArgument(
                    !java.util.Objects.equals(requestToPreempt.requestId(), incomingId)
                            && ids.add(requestToPreempt.requestId())
                            && requestToPreempt.reservationToken() > 0L
                            && requestToPreempt.priorityKnown()
                            && PriorityNormalizer.hasPriority(requestToPreempt.priority())
                            && requestToPreempt.priority() < incomingPriority
                            && !requestToPreempt.claimedForPreemption(),
                    "preemption victim identity or priority is invalid");
            checkArgument(
                    preemption.allows(
                            requestToPreempt.phase().isMasterQueued()
                                    ? PreemptionStage.DECODE_RESERVED
                                    : PreemptionStage.DECODE_ENGINE_OWNED),
                    "preemption victim stage is disabled");
        }
    }

    /**
     * Metrics observe plans and commits without owning the reservation transaction.
     */
    private void reportPreemption(
            PreemptionEvent event,
            int priority,
            String requestId,
            String preemptionReason,
            String outcome) {
        report(
                requestId,
                event == PreemptionEvent.PLAN ? "eviction plan" : "eviction commit",
                () -> reporter.reportPreemption(event, priority, preemptionReason, outcome));
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

    private CompletableFuture<PreemptionResult> startEngineCancelPreemption(
            RequestContext ctx,
            DecodePreemptionPlan proposal,
            DecodeEndpoint endpoint,
            Request request,
            SchedulingMetadata metadata) {
        var command =
                new PreemptionCommand(
                        endpoint,
                        request,
                        metadata,
                        proposal.requestsToPreempt(),
                        50L,
                        preemption.getTimeoutMs(),
                        () -> ctx.data().getRequestScheduler().isAdmissionOpen(ctx),
                        "preempted by higher-priority request " + ctx.data().getRequestId());
        report(
                ctx.data().getRequestId(),
                "cancel requests",
                () ->
                        reporter.reportPreemptionCancelRequests(
                                proposal.endpointId(), proposal.requestsToPreempt()));
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
                                                reporter.reportPreemptedRequests(
                                                        ctx.data().getRequestId(),
                                                        ctx.data().getPriority(),
                                                        proposal.endpointId(),
                                                        proposal.preemptionReason(),
                                                        proposal.requestsToPreempt(),
                                                        true);
                                                recordDecodePlanObservability(ctx, proposal);
                                            }
                                        }));
    }

    /**
     * Record the single committed Decode-preemption plan.
     */
    private static void recordDecodePlanObservability(
            RequestContext ctx, DecodePreemptionPlan proposal) {
        long totalCost = proposal.diagnosticCost();
        ctx.data().setPlanType("decode_evict");
        ctx.data().setPlanCost(totalCost);
        ctx.data().setPreemptedRequestCount(proposal.requestsToPreempt().size());
        Logger.debug(
                "[decode-capacity] decode eviction committed: request_id={} priority={} case={} "
                        + "victims={} total_cost={} freed_kv={} worker={}",
                ctx.data().getRequestId(),
                ctx.data().getPriority(),
                proposal.preemptionReason(),
                proposal.requestsToPreempt().size(),
                totalCost,
                proposal.freedKvTokens(),
                proposal.endpointId());
    }

    private DecodeResources.RequestReservation replaceQueuedDecodeReservations(
            DecodeEndpoint endpoint,
            List<DecodeResources.RequestReservation> reservationsToPreempt,
            Request request,
            int priority) {
        List<RoutingContext> claimed = new ArrayList<>(reservationsToPreempt.size());
        DecodeResources.RequestReservation incoming = null;
        try {
            for (DecodeResources.RequestReservation reservationToPreempt : reservationsToPreempt) {
                RoutingContext withdrawal =
                        beginRoutePreemption(endpoint, reservationToPreempt, priority);
                if (withdrawal == null) {
                    return null;
                }
                claimed.add(withdrawal);
            }
            incoming =
                    endpoint.replaceQueuedRequests(
                            reservationsToPreempt,
                            request.getRequestId(),
                            request.getHardKvTokens(),
                            request.getExpectedKvTokens(),
                            priority,
                            capacity);
            return incoming;
        } finally {
            Throwable failure = null;
            for (RoutingContext withdrawal : claimed) {
                boolean committed = incoming != null;
                failure =
                        Failures.run(
                                failure,
                                () ->
                                        AbstractRequestScheduler.completeRouting(
                                                withdrawal,
                                                committed
                                                        ? ContextResult.RoutingCompletion.WITHDRAWN
                                                        : ContextResult.RoutingCompletion
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

    /**
     * Holds the incoming request, requests to cancel and time limits for one preemption process.
     */
    record PreemptionCommand(
            DecodeEndpoint endpoint,
            Request request,
            SchedulingMetadata metadata,
            List<DecodeRequestView> requestsToPreempt,
            long cancelAckTimeoutMs,
            long preemptionTimeoutMs,
            BooleanSupplier admissionOpen,
            String detail) {
        public PreemptionCommand {
            checkArgument(
                    endpoint != null && requestsToPreempt != null && !requestsToPreempt.isEmpty(),
                    "endpoint and victims are required");
            requestsToPreempt = List.copyOf(requestsToPreempt);
            Objects.requireNonNull(request, "request");
            Objects.requireNonNull(metadata, "metadata");
            checkArgument(
                    request.getRequestId() != null && !request.getRequestId().isBlank(),
                    "incoming request id is required");
            Set<String> requestIdsToPreempt = new LinkedHashSet<>();
            for (DecodeRequestView requestToPreempt : requestsToPreempt) {
                checkArgument(
                        requestToPreempt.requestId() != null
                                && !requestToPreempt.requestId().isBlank()
                                && requestToPreempt.reservationToken() > 0L,
                        "victim requestId and reservation token must be positive");
                checkArgument(
                        requestToPreempt.phase() != null
                                && requestToPreempt.phase().requiresEngineCancel(),
                        "capacity acquisition accepts only Engine-Cancel victims");
                checkArgument(
                        requestIdsToPreempt.add(requestToPreempt.requestId()),
                        "duplicate victim %s",
                        requestToPreempt.requestId());
            }
            checkArgument(admissionOpen != null, "admission gate is required");
        }
    }

    private enum ClaimDisposition {
        RELEASABLE,
        OUTBOUND,
        TRANSFERRED
    }

    /**
     * Pairs a selected request with the endpoint that records release of its exact reservation.
     */
    private final class PreemptionRequest {
        private final PreemptionRegistration claim;

        private CompletableFuture<PreemptionRegistration.Reply> acknowledgement;
        private volatile ClaimDisposition disposition = ClaimDisposition.RELEASABLE;

        private final DecodeEndpoint endpoint;

        private PreemptionRequest(PreemptionRegistration claim, DecodeEndpoint endpoint) {
            this.claim = claim;
            this.endpoint = endpoint;
        }

        private boolean resourcesReleased() {
            return endpoint.isPreemptionResourceReleased(claim.attemptToken(), claim.reservation());
        }

        private String requestId() {
            return claim.reservation().requestId();
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

        private final List<PreemptionRequest> claims = new ArrayList<>();

        private final long expiresAt;

        private final CompletableFuture<Boolean> settlement = new CompletableFuture<>();

        private boolean endpointBegun;
        private boolean closed;

        private String cleanupFailure;

        private boolean rejectWhenOnlyNotFoundRemain;

        private Attempt(PreemptionCommand command, long token) {
            this.command = command;
            this.token = token;
            this.expiresAt =
                    System.nanoTime()
                            + java.util.concurrent.TimeUnit.MILLISECONDS.toNanos(
                                    command.preemptionTimeoutMs());
            activeWork.incrementAndGet();
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
                for (PreemptionRequest owned : claims) {
                    if (owned.disposition == ClaimDisposition.OUTBOUND) {
                        cleanup(
                                "transfer_unknown:" + owned.requestId(),
                                () -> transferUnknown(owned));
                    }
                }
                if (endpointBegun) {
                    cleanup("endpoint_abort", () -> command.endpoint().abortPreemption(token));
                }
                for (PreemptionRequest owned : claims) {
                    if (owned.disposition == ClaimDisposition.RELEASABLE
                            && !owned.resourcesReleased()) {
                        cleanup(
                                "release_claim:" + owned.requestId(),
                                () ->
                                        completion()
                                                .applyEvent(
                                                        owned.claim,
                                                        new PreemptionCompletion.PreemptionEvent(
                                                                null, true, null, null, null)));
                    }
                }
            } finally {
                activeWork.decrementAndGet();
            }
        }

        private void observe() {
            Runnable unsubscribe =
                    command.endpoint().watchEngineDispatchCapacity(this::resourcesChanged);
            settlement.whenComplete((ignored, failure) -> unsubscribe.run());
            resourcesChanged();
            if (!settlement.isDone()) {
                long remaining = expiresAt - System.nanoTime();
                if (remaining <= 0) {
                    settlement.complete(false);
                } else {
                    var deadline =
                            timer.schedule(
                                    () -> settlement.complete(allResourcesReleased()),
                                    Math.max(
                                            1L,
                                            java.util.concurrent.TimeUnit.NANOSECONDS.toMillis(
                                                    remaining)));
                    settlement.whenComplete((ignored, failure) -> deadline.cancel(false));
                }
            }
        }

        private synchronized void resourcesChanged() {
            if (settlement.isDone()) return;
            if (!command.admissionOpen().getAsBoolean()) {
                settlement.complete(false);
            } else if (allResourcesReleased()) {
                settlement.complete(true);
            } else if (rejectWhenOnlyNotFoundRemain
                    && claims.stream()
                            .filter(
                                    owned ->
                                            owned.acknowledgement.join()
                                                    != PreemptionRegistration.Reply.NOT_FOUND)
                            .allMatch(PreemptionRequest::resourcesReleased)) {
                settlement.complete(false);
            }
        }

        private synchronized boolean allResourcesReleased() {
            if (claims.isEmpty()) {
                return false;
            }
            return claims.stream().allMatch(PreemptionRequest::resourcesReleased);
        }

        private synchronized void rejectWhenOnlyNotFoundRemain() {
            rejectWhenOnlyNotFoundRemain = true;
            resourcesChanged();
        }

        private synchronized boolean outboundStarted(PreemptionRequest owned) {
            if (owned.resourcesReleased() || settlement.isDone() && !settlement.getNow(false)) {
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

        private void transferUnknown(PreemptionRequest owned) {
            if (owned.disposition != ClaimDisposition.OUTBOUND || owned.resourcesReleased()) {
                return;
            }
            completion()
                    .applyEvent(
                            owned.claim,
                            new PreemptionCompletion.PreemptionEvent(
                                    PreemptionCancelPhase.CANCEL_UNKNOWN, false, null, null, null));
            transferred(owned);
        }

        private synchronized void transferred(PreemptionRequest owned) {
            owned.disposition = ClaimDisposition.TRANSFERRED;
        }

        private PreemptionResult finish(boolean hasNotFound, boolean resourcesReleased) {
            DecodeResources.RequestReservation incoming =
                    resourcesReleased
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
                    hasNotFound && claims.size() == 1 && !claims.get(0).resourcesReleased();
            return abort(
                    !cleanSingleNotFound,
                    cleanSingleNotFound ? "cancel_not_found" : "cancel_terminal_unknown");
        }

        private synchronized boolean beginClose() {
            if (closed) {
                return false;
            }
            closed = true;
            return true;
        }

        private PreemptionResult abort(boolean controlFailure, String detail) {
            close();
            String resultDetail =
                    cleanupFailure == null ? detail : detail + ";cleanup_failed=" + cleanupFailure;
            return new PreemptionResult(null, controlFailure, resultDetail);
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
}
