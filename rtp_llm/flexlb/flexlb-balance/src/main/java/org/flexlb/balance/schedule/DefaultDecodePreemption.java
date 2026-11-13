package org.flexlb.balance.schedule;

import org.flexlb.balance.metrics.RequestSchedulerReporter;
import org.flexlb.balance.metrics.RequestSchedulerReporter.PreemptionEvent;
import org.flexlb.balance.schedule.DecodePreemptionPolicy.DecodePreemptionPlan;
import org.flexlb.config.ConfigService;
import org.flexlb.config.scheduler.PreemptionConfig;
import org.flexlb.config.scheduler.PreemptionStage;
import org.flexlb.util.Logger;
import org.flexlb.util.PriorityNormalizer;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.DecodeResources;
import org.flexlb.worker.endpoint.DecodeResources.AdmissionCapacity;
import org.flexlb.worker.endpoint.DecodeResources.DecodeRequestView;
import org.flexlb.worker.endpoint.Request;
import org.flexlb.worker.endpoint.WorkerEndpoint;

import javax.annotation.PreDestroy;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CompletableFuture;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

/**
 * Selects lower-priority Decode requests and asks their schedulers to cancel them.
 * Incoming requests keep waiting through the ordinary capacity queue; no capacity is reserved for a waiter.
 */
public class DefaultDecodePreemption implements DecodePreemption {
    private final RequestSchedulerReporter reporter;

    private volatile boolean shutdown;

    private final java.util.concurrent.atomic.AtomicInteger activeWork =
            new java.util.concurrent.atomic.AtomicInteger();
    private final ThreadLocal<Boolean> acquiring = new ThreadLocal<>();
    private final RequestRepository requests;
    private final DecodePreemptionPolicy selectionPolicy;
    private final AdmissionCapacity capacity;
    private final PreemptionConfig preemption;

    public DefaultDecodePreemption(
            RequestRepository requests,
            RequestSchedulerReporter reporter,
            ConfigService configService,
            AdmissionCapacity capacity,
            DecodePreemptionPolicy selectionPolicy) {
        this.reporter = Objects.requireNonNull(reporter, "reporter");
        this.requests = Objects.requireNonNull(requests, "requests");

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

    @Override
    public CompletableFuture<RequestResult> tryPreempt(
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

    private CompletableFuture<RequestResult> startPreemption(
            RequestContext ctx, WorkerEndpoint blockedEndpoint) {
        Request request = ctx.getRequest();
        SchedulingMetadata metadata = ctx.getSchedulingMetadata();
        if (shutdown
                || ctx.getFuture().isDone()
                || System.currentTimeMillis() >= ctx.getRequestExpiresAtMs()
                || !PriorityNormalizer.hasPriority(metadata.priority())
                || preemption == null) {
            return null;
        }
        if (!(blockedEndpoint instanceof DecodeEndpoint decodeEndpoint)
                || !preemption.allows(PreemptionStage.DECODE_ENGINE_OWNED)) {
            return null;
        }
        DecodePreemptionPlan proposal = planDecodePreemption(request, metadata, decodeEndpoint);
        if (proposal == null || !ctx.getRequestScheduler().isAdmissionOpen(ctx)) {
            return null;
        }
        return startEngineCancelPreemption(ctx, proposal, decodeEndpoint, request, metadata);
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
        if (decision.deficit().isZero()) {
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
                    selected.selection().address(),
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
     * Validate that the policy selected lower-priority requests from the supplied snapshot.
     */
    private void validateProposal(
            DecodePreemptionPlan proposal,
            DecodeResources.ResourceSnapshot snapshot,
            String incomingId,
            int incomingPriority) {
        checkArgument(
                Objects.equals(proposal.endpointId(), snapshot.selection().address()),
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
                            && requestToPreempt.reservation() != null
                            && PriorityNormalizer.hasPriority(requestToPreempt.priority())
                            && requestToPreempt.priority() < incomingPriority,
                    "preemption victim identity or priority is invalid");
            checkArgument(
                    preemption.allows(PreemptionStage.DECODE_ENGINE_OWNED),
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

    private CompletableFuture<RequestResult> startEngineCancelPreemption(
            RequestContext ctx,
            DecodePreemptionPlan proposal,
            DecodeEndpoint endpoint,
            Request request,
            SchedulingMetadata metadata) {
        report(
                ctx.getRequestId(),
                "cancel requests",
                () ->
                        reporter.reportPreemptionCancelRequests(
                                proposal.endpointId(), proposal.requestsToPreempt()));
        recordDecodePlanObservability(ctx, proposal);
        return cancelSelectedRequests(ctx, endpoint, proposal.requestsToPreempt());
    }

    /**
     * Record the selected requests; their schedulers independently report cancellation outcomes.
     */
    private static void recordDecodePlanObservability(
            RequestContext ctx, DecodePreemptionPlan proposal) {
        long totalCost = proposal.diagnosticCost();
        ctx.recordPreemptionStatistics(totalCost, proposal.requestsToPreempt().size());
        Logger.debug(
                "[decode-capacity] decode cancellation selected: request_id={} priority={} case={} "
                        + "victims={} total_cost={} estimated_kv={} worker={}",
                ctx.getRequestId(),
                ctx.getPriority(),
                proposal.preemptionReason(),
                proposal.requestsToPreempt().size(),
                totalCost,
                proposal.freedKvTokens(),
                proposal.endpointId());
    }

    private CompletableFuture<RequestResult> cancelSelectedRequests(
            RequestContext context,
            DecodeEndpoint endpoint,
            List<DecodeRequestView> selectedRequests) {
        for (var selected : selectedRequests) {
            if (shutdown
                    || context.getFuture().isDone()
                    || !context.getRequestScheduler().isAdmissionOpen(context)) {
                return CompletableFuture.completedFuture(RequestResult.closed());
            }
            var victim = requests.findActive(selected.requestId());
            if (victim == null) continue;
            var route = victim.snapshot().route();
            if (route == null
                    || route.decodeEndpoint() != endpoint
                    || route.decodeRegistration() != selected.reservation()) continue;
            victim.getRequestScheduler()
                    .cancel(victim, org.flexlb.balance.schedule.CancelReason.PRIORITY_PREEMPTED);
        }
        return CompletableFuture.completedFuture(
                RequestResult.fromDecision(
                        org.flexlb.balance.strategy.RoutingResult.blocked(
                                org.flexlb.worker.endpoint.CapacityKey.exact(
                                        org.flexlb.config.service.RoleType.DECODE,
                                        endpoint.getStatus().topologySnapshot().group(),
                                        endpoint.ipPort()))));
    }
}
