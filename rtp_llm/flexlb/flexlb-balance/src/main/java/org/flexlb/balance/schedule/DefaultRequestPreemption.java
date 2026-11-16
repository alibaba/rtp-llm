package org.flexlb.balance.schedule;

import org.flexlb.balance.metrics.RequestSchedulerReporter;
import org.flexlb.balance.metrics.RequestSchedulerReporter.PreemptionEvent;
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
import org.flexlb.worker.endpoint.Request;
import org.flexlb.worker.endpoint.WorkerEndpoint;
import org.flexlb.worker.status.ServerStatus;

import java.util.HashMap;
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

    private final Map<RequestContext.SelectionAttempt, Pending> pending = new HashMap<>();

    private volatile Runnable wakeup = () -> {};

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
                        ? settings.priorityOrdering().getPreemption()
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
            Consumer<RoleSelection> handler) {
        var request = selection.requestContext();
        if (!(target instanceof DecodeEndpoint endpoint)
                || preemption == null
                || !preemption.allows(PreemptionStage.DECODE_ENGINE_OWNED)
                || request.isResponseDone()) return false;
        var proposal =
                planDecodePreemption(
                        request.getRequest(), request.getSchedulingMetadata(), endpoint);
        if (proposal == null) return false;
        var wait =
                endpoint.createResourceWait(
                        request,
                        selection.id(),
                        request.getPriority(),
                        request.getRequestExpiresAtMs(),
                        request.getRequest().getHardKvTokens(),
                        request.getRequest().getExpectedKvTokens(),
                        capacity);
        var accepted = new Pending(selection, endpoint, wait, handler);
        synchronized (this) {
            if (stopped || pending.size() >= limit || pending.containsKey(selection)) return false;
            pending.put(selection, accepted);
        }
        try {
            if (!request.bindResourceWait(selection, wait)
                    || !endpoint.registerResourceWait(wait)) {
                request.clearResourceWait(wait);
                endpoint.removeResourceWait(wait);
                return true;
            }
            wait.result().whenComplete((resource, failure) -> wakeup.run());
            endpoint.tryAcquire(wait);
            if (!wait.isDone()) {
                recordDecodePlanObservability(request, proposal);
                for (var selected : proposal.requestsToPreempt()) {
                    if (!wait.isEffective(System.currentTimeMillis()) || admissionStopped()) break;
                    var victim = requests.findActive(selected.requestId());
                    if (victim == null) continue;
                    var route = victim.snapshot().route();
                    if (route == null
                            || route.decodeEndpoint() != endpoint
                            || route.decodeRegistration() != selected.reservation()) continue;
                    victim.getRequestScheduler().cancel(victim, CancelReason.PRIORITY_PREEMPTED);
                    endpoint.tryAcquire(wait);
                    if (wait.isDone()) break;
                }
            }
            return true;
        } catch (RuntimeException | Error failure) {
            request.clearResourceWait(wait);
            endpoint.removeResourceWait(wait);
            throw failure;
        } finally {
            synchronized (this) {
                accepted.installing = false;
                accepted.checkRequested = true;
            }
            wakeup.run();
        }
    }

    @Override
    public synchronized List<Work> readyWork() {
        if (pending.isEmpty()) return List.of();
        var work = new java.util.ArrayList<Work>();
        for (var item : pending.values()) {
            if (!item.installing && (item.checkRequested || item.wait.isDone()))
                work.add(new Work(item.selection, () -> process(item)));
        }
        return work;
    }

    @Override
    public void requestCheck(WorkerEndpoint endpoint) {
        synchronized (this) {
            for (var item : pending.values())
                if (item.endpoint == endpoint) item.checkRequested = true;
        }
    }

    @Override
    public void requestPeriodicCheck() {
        synchronized (this) {
            for (var item : pending.values()) item.checkRequested = true;
        }
    }

    @Override
    public void cancel(RequestContext request) {
        List<Pending> affected;
        synchronized (this) {
            affected =
                    pending.values().stream()
                            .filter(item -> item.selection.requestContext() == request)
                            .toList();
        }
        for (var item : affected) {
            request.clearResourceWait(item.wait);
            item.endpoint.removeResourceWait(item.wait);
        }
        if (!affected.isEmpty()) wakeup.run();
    }

    @Override
    public void close() {
        stopAdmission();
        checkState(isIdle(), "scheduler must finish preemption result cleanup before closing");
    }

    @Override
    public void stopAdmission() {
        List<RequestContext> requests;
        synchronized (this) {
            stopped = true;
            requests = List.copyOf(waitingRequests());
        }
        requests.forEach(this::cancel);
    }

    @Override
    public synchronized Set<RequestContext> waitingRequests() {
        if (pending.isEmpty()) return Set.of();
        var requests =
                java.util.Collections.newSetFromMap(
                        new java.util.IdentityHashMap<RequestContext, Boolean>());
        for (var item : pending.keySet()) requests.add(item.requestContext());
        return requests;
    }

    @Override
    public synchronized boolean isIdle() {
        return pending.isEmpty();
    }

    private synchronized boolean admissionStopped() {
        return stopped;
    }

    private void process(Pending item) {
        synchronized (this) {
            if (pending.get(item.selection) != item || item.installing) return;
            item.checkRequested = false;
        }
        var request = item.selection.requestContext();
        if (!item.wait.isDone()) {
            if (!item.wait.isEffective(System.currentTimeMillis())) {
                request.clearResourceWait(item.wait);
                item.endpoint.removeResourceWait(item.wait);
            } else item.endpoint.tryAcquire(item.wait);
        }
        if (!item.wait.isDone()) return;
        var resource = item.wait.result().toCompletableFuture().join();
        request.clearResourceWait(item.wait);
        try {
            RoleSelection acquired = null;
            if (resource != null) {
                try {
                    var view = item.endpoint.selectionSnapshot();
                    acquired =
                            RoleSelection.decode(
                                    item.endpoint,
                                    ServerStatus.forWorker(
                                            org.flexlb.config.service.RoleType.DECODE,
                                            request.getRequestId(),
                                            view.topology(),
                                            view.workerStatus().fields()),
                                    resource);
                } catch (RuntimeException | Error failure) {
                    item.endpoint.rollbackSelection(resource);
                    throw failure;
                }
            }
            item.handler.accept(acquired);
        } finally {
            synchronized (this) {
                pending.remove(item.selection, item);
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
     * Keeps the exact selection and shared endpoint wait until installation and result cleanup finish.
     */
    private static final class Pending {
        private final RequestContext.SelectionAttempt selection;
        private final DecodeEndpoint endpoint;
        private final DecodeEndpoint.ResourceWait wait;
        private final Consumer<RoleSelection> handler;
        private boolean installing = true;
        private boolean checkRequested;

        private Pending(
                RequestContext.SelectionAttempt selection,
                DecodeEndpoint endpoint,
                DecodeEndpoint.ResourceWait wait,
                Consumer<RoleSelection> handler) {
            this.selection = selection;
            this.endpoint = endpoint;
            this.wait = wait;
            this.handler = handler;
        }
    }
}
