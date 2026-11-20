package org.flexlb.balance.schedule;

import org.flexlb.balance.metrics.RequestSchedulerReporter;
import org.flexlb.balance.request.Request;
import org.flexlb.balance.schedule.DecodePreemptionPlanner.DecodePreemptionPlan;
import org.flexlb.balance.strategy.RoleSelection;
import org.flexlb.config.ConfigService;
import org.flexlb.config.scheduler.PreemptionConfig;
import org.flexlb.config.scheduler.PreemptionStage;
import org.flexlb.util.Logger;
import org.flexlb.util.PriorityNormalizer;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.DecodeResources;
import org.flexlb.worker.endpoint.DecodeResources.AdmissionCapacity;
import org.flexlb.worker.endpoint.DecodeResources.DecodeRequestView;
import org.flexlb.worker.endpoint.WorkerEndpoint;

import java.util.Collections;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Consumer;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

/**
 * Registers one resource wait before cancelling selected Decode requests.
 * Capacity checks and result adoption run as short scheduler tasks; waiting itself holds no thread.
 */
final class DefaultRequestPreemption implements RequestPreemption {
    private final RequestRepository requests;
    private final RequestSchedulerReporter reporter;
    private final DecodePreemptionPlanner selectionPolicy;
    private final AdmissionCapacity capacity;
    private final PreemptionConfig preemption;

    private final int limit;

    private final Map<RequestContext.SelectionAttempt, DecodePreemptionWait> decodePreemptionWaits =
            new HashMap<>();

    private volatile Runnable wakeup = () -> {};

    private volatile long version;
    private boolean stopped;

    DefaultRequestPreemption(
            RequestRepository requests,
            RequestSchedulerReporter reporter,
            ConfigService config,
            AdmissionCapacity capacity,
            DecodePreemptionPlanner selectionPolicy) {
        this.requests = requests;
        this.reporter = reporter;
        this.capacity = capacity;
        this.selectionPolicy = selectionPolicy;
        var settings = config.loadBalanceConfig();
        this.preemption =
                settings.isQueue() && settings.isPriorityOrdering()
                        ? settings.priorityOrdering().getPriority().getPreemption()
                        : null;
        this.limit = Math.max(1, settings.getInternalRuntime().getQueuePlannerThreads());
    }

    @Override
    public void setWakeup(Runnable wakeup) {
        this.wakeup = Objects.requireNonNull(wakeup);
    }

    @Override
    public boolean start(
            RequestContext.SelectionAttempt selection,
            WorkerEndpoint target,
            Consumer<RoleSelection> resultHandler) {
        var request = selection.requestContext();
        if (!(target instanceof DecodeEndpoint endpoint)
                || preemption == null
                || !preemption.allows(PreemptionStage.DECODE_ENGINE_OWNED)
                || request.isResponseDone()) return false;
        var proposal =
                planDecodePreemption(
                        request.getRequest(), request.getSchedulingMetadata(), endpoint);
        if (proposal == null) return false;
        var resourceWait =
                endpoint.createResourceWait(
                        request,
                        selection.id(),
                        request.getPriority(),
                        request.getRequestExpiresAtMs(),
                        request.getRequest().getHardKvTokens(),
                        request.getRequest().getExpectedKvTokens(),
                        capacity);
        var preemptionWait = new DecodePreemptionWait(endpoint, resourceWait);
        synchronized (this) {
            if (stopped
                    || decodePreemptionWaits.size() >= limit
                    || decodePreemptionWaits.containsKey(selection)) return false;
            decodePreemptionWaits.put(selection, preemptionWait);
        }
        try {
            if (!request.bindResourceWait(selection, resourceWait)
                    || !endpoint.registerResourceWait(resourceWait)) {
                request.clearResourceWait(resourceWait);
                endpoint.removeResourceWait(resourceWait);
                return true;
            }
            resourceWait.result().whenComplete((resource, failure) -> wakeup.run());
            endpoint.tryAcquire(resourceWait);
            if (!resourceWait.isDone()) {
                recordDecodePlanObservability(request, proposal);
                for (var selected : proposal.requestsToPreempt()) {
                    if (!endpoint.isResourceWaitEffective(resourceWait, System.currentTimeMillis())
                            || admissionStopped()) break;
                    var victim = requests.findActive(selected.requestId());
                    if (victim == null) continue;
                    var route = victim.snapshot().route();
                    if (route == null
                            || route.decodeEndpoint() != endpoint
                            || route.decodeRegistration() != selected.reservation()) continue;
                    victim.getRequestScheduler().cancel(victim, CancelReason.PRIORITY_PREEMPTED);
                    endpoint.tryAcquire(resourceWait);
                    if (resourceWait.isDone()) break;
                }
            }
            return true;
        } catch (RuntimeException | Error failure) {
            request.clearResourceWait(resourceWait);
            endpoint.removeResourceWait(resourceWait);
            throw failure;
        } finally {
            synchronized (this) {
                preemptionWait.checkAndDeliverTask =
                        new Work(
                                selection,
                                () ->
                                        checkWaitAndDeliverResult(
                                                selection, preemptionWait, resultHandler),
                                resourceWait::isDone);
                version++;
            }
            wakeup.run();
        }
    }

    @Override
    public synchronized List<Work> pendingWork() {
        if (decodePreemptionWaits.isEmpty()) return List.of();
        var tasks = new java.util.ArrayList<Work>();
        for (var preemptionWait : decodePreemptionWaits.values()) {
            if (preemptionWait.checkAndDeliverTask != null) {
                tasks.add(preemptionWait.checkAndDeliverTask);
            }
        }
        return tasks;
    }

    @Override
    public synchronized void onCapacityChanged(WorkerEndpoint endpoint) {
        for (var preemptionWait : decodePreemptionWaits.values()) {
            if (preemptionWait.endpoint == endpoint) {
                version++;
                return;
            }
        }
    }

    @Override
    public long snapshotVersion() {
        return version;
    }

    @Override
    public void cancel(RequestContext request) {
        List<DecodePreemptionWait> waitsToCancel;
        synchronized (this) {
            waitsToCancel =
                    decodePreemptionWaits.entrySet().stream()
                            .filter(entry -> entry.getKey().requestContext() == request)
                            .map(Map.Entry::getValue)
                            .toList();
        }
        for (var preemptionWait : waitsToCancel) {
            request.clearResourceWait(preemptionWait.resourceWait);
            preemptionWait.endpoint.removeResourceWait(preemptionWait.resourceWait);
        }
        if (!waitsToCancel.isEmpty()) wakeup.run();
    }

    @Override
    public void close() {
        stopAdmission();
        checkState(isIdle(), "scheduler must finish preemption result cleanup before closing");
    }

    @Override
    public void stopAdmission() {
        var requests = Collections.newSetFromMap(new IdentityHashMap<RequestContext, Boolean>());
        synchronized (this) {
            stopped = true;
            for (var selection : decodePreemptionWaits.keySet()) {
                requests.add(selection.requestContext());
            }
        }
        requests.forEach(this::cancel);
    }

    @Override
    public synchronized boolean isIdle() {
        return decodePreemptionWaits.isEmpty();
    }

    private synchronized boolean admissionStopped() {
        return stopped;
    }

    private void checkWaitAndDeliverResult(
            RequestContext.SelectionAttempt selection,
            DecodePreemptionWait preemptionWait,
            Consumer<RoleSelection> resultHandler) {
        synchronized (this) {
            if (decodePreemptionWaits.get(selection) != preemptionWait) return;
        }
        var request = selection.requestContext();
        var endpoint = preemptionWait.endpoint;
        var resourceWait = preemptionWait.resourceWait;
        if (!resourceWait.isDone()) {
            if (!endpoint.isResourceWaitEffective(resourceWait, System.currentTimeMillis())) {
                request.clearResourceWait(resourceWait);
                endpoint.removeResourceWait(resourceWait);
            } else endpoint.tryAcquire(resourceWait);
        }
        if (!resourceWait.isDone()) return;
        var resource = resourceWait.result().toCompletableFuture().join();
        request.clearResourceWait(resourceWait);
        try {
            RoleSelection acquired = null;
            if (resource != null) {
                try {
                    var view = endpoint.selectionSnapshot();
                    acquired = RoleSelection.decode(endpoint, view.topology(), resource);
                } catch (RuntimeException | Error failure) {
                    endpoint.rollbackSelection(resource);
                    throw failure;
                }
            }
            resultHandler.accept(acquired);
        } finally {
            synchronized (this) {
                decodePreemptionWaits.remove(selection, preemptionWait);
            }
            wakeup.run();
        }
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
            reportPreemptionPlan(
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
        reportPreemptionPlan(
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
     * Reports plan outcomes without owning the reservation transaction.
     */
    private void reportPreemptionPlan(
            int priority, String requestId, String preemptionReason, String outcome) {
        report(
                requestId,
                "eviction plan",
                () -> reporter.reportPreemptionPlan(priority, preemptionReason, outcome));
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

    /**
     * Tracks the Decode endpoint and its resource wait until the result has been delivered.
     * The check and delivery task is published under the preemption monitor after installation.
     */
    private static final class DecodePreemptionWait {
        private final DecodeEndpoint endpoint;
        private final DecodeEndpoint.ResourceWait resourceWait;

        private Work checkAndDeliverTask;

        private DecodePreemptionWait(
                DecodeEndpoint endpoint, DecodeEndpoint.ResourceWait resourceWait) {
            this.endpoint = endpoint;
            this.resourceWait = resourceWait;
        }
    }
}
