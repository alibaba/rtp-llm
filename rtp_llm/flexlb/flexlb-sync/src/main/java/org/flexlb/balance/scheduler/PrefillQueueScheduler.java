package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.DecodeResources.DecodeRoutingView;
import org.flexlb.balance.endpoint.EndpointStats;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.prediction.PrefillTimePredictor.InvalidPrefillPredictionException;
import org.flexlb.balance.projection.RequestTimeEstimation;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.service.monitor.RequestSchedulerReporter.QueueWaitSnapshot;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalLong;

import static com.google.common.base.Preconditions.checkState;

/**
 * Schedules one Prefill queue, including delivery attempts, capacity waits and retries.
 * The delivery scheduler runs one task at a time per queue and waits for tasks before clearing subscriptions.
 */
public final class PrefillQueueScheduler {
    private static final Map<String, Object> INITIAL_QUEUE_WAIT_SNAPSHOT =
            Map.of("cause", "waiting for Prefill decision");

    private final String key;
    private final PrefillEndpoint prefillEndpoint;
    private final QueueExecutionSettings settings;
    private final DeliveryCoordinator dispatcher;

    private Runnable removeCapacityListener;

    public final Runnable capacityAvailableSignal;

    /**
     * Last decision used to replace the capacity subscription.
     */
    private volatile Decision waiting;

    /**
     * Last waiting decision; timeout readers use the published immutable snapshot.
     */
    private volatile QueueWaitSnapshot latestQueueWaitSnapshot;

    public PrefillQueueScheduler(
            String key,
            PrefillEndpoint prefillEp,
            QueueExecutionSettings settings,
            DeliveryCoordinator dispatcher) {
        this.key = key;
        this.prefillEndpoint = Objects.requireNonNull(prefillEp, "prefillEndpoint");
        this.settings = Objects.requireNonNull(settings);
        this.capacityAvailableSignal =
                () -> prefillEndpoint.signal(PrefillEndpoint.Signal.PLACEMENT);
        this.dispatcher = Objects.requireNonNull(dispatcher, "dispatcher");
    }

    /**
     * Endpoint owns queue membership; this scheduler owns waiting decisions and subscriptions.
     */
    public PrefillEndpoint prefillEndpoint() {
        return prefillEndpoint;
    }

    public QueueExecutionSettings settings() {
        return settings;
    }

    /**
     * Read the last captured queue wait state without waiting for an Endpoint lock.
     */
    public Map<String, Object> getLatestQueueWaitSnapshot() {
        QueueWaitSnapshot snapshot = latestQueueWaitSnapshot;
        return snapshot == null ? INITIAL_QUEUE_WAIT_SNAPSHOT : snapshot.asMap();
    }

    /**
     * Capture fixed-size counters without formatting the PV record on the scheduling loop.
     */
    public void recordQueueWait(RouteResult head, String reason) {
        EndpointStats.QueueCounts counters = prefillEndpoint.queueCounters();
        DecodeRoutingView decode =
                head.decodeEndpoint() == null ? null : head.decodeEndpoint().routingView();
        latestQueueWaitSnapshot =
                new QueueWaitSnapshot(
                        key, reason, head.requestId(), head.enqueueSeq(), counters, decode);
    }

    // ==================== Queue ownership and projection ====================

    public GroupPlanner.Constraints schedulingConstraints() {
        WorkerStatus status = prefillEndpoint.getStatus();
        var engine = status == null ? null : status.committedEngineObservation();
        return settings.constraints(engine);
    }

    public CapacityBoundary processSelection(
            List<RouteResult> candidates,
            String reason,
            PrefillTimePredictor.Evaluator evaluator,
            OptionalLong prediction) {
        return dispatcher.submit(
                candidates, reason, evaluator, prediction, Delivery.PrefixPolicy.ORDERED_PREFIX);
    }

    public Decision commitBoundary(RouteResult blockedItem, CapacityBoundary blockedResult) {
        if (blockedItem != null
                && blockedResult != null
                && blockedResult.status() == CapacityBoundary.Status.FAILED) {
            dispatcher.failQueuedPreparation(blockedItem, blockedResult.cause());
            return Decision.NO_WAIT;
        }
        if (blockedItem != null && !blockedItem.selectedEndpointsAvailable()) {
            blockedItem
                    .ctx()
                    .data()
                    .getRequestScheduler()
                    .onRouteFailure(
                            blockedItem,
                            ContextResult.RouteFailureReason.ENDPOINT_UNAVAILABLE,
                            "Selected Endpoint unavailable during delivery preparation");
            return Decision.NO_WAIT;
        }
        if (blockedItem == null || blockedResult == null || stopping()) return Decision.NO_WAIT;
        if (blockedResult.unavailable()) {
            return captureQueue(1).head() == blockedItem
                            && !blockedItem.requestExpired(System.currentTimeMillis())
                    ? Decision.capacityBlocked(blockedItem, blockedResult)
                    : Decision.NO_WAIT;
        }
        return Decision.NO_WAIT;
    }

    public boolean stopping() {
        return prefillEndpoint.queueStopped();
    }

    public PrefillEndpoint.QueueSnapshot captureQueue(int limit) {
        return prefillEndpoint.captureQueue(limit);
    }

    /**
     * Subscribe before checking the resource so capacity changes cannot be missed.
     */
    public void observeWait(Decision decision) {
        Decision previous = waiting;
        waiting = stopping() ? null : decision;
        if (previous != waiting
                && ((previous != null && previous.unavailable() != null)
                        || (waiting != null && waiting.unavailable() != null))) {
            updateCapacityListener(
                    waiting == null || waiting.unavailable() == null ? null : waiting);
        }
    }

    /**
     * After subscribing, check for remaining work or capacity restored before the subscription.
     * Later changes request another turn through the delivery scheduler.
     */
    boolean needsImmediateScheduling(Decision decision, long nowMs) {
        if (stopping()) return false;
        RouteResult head = captureQueue(1).head();
        if (head == null) return false;
        return decision.request() != head
                || head.requestExpired(nowMs)
                || (decision.unavailable() != null
                        && capacityAvailable(head, decision.unavailable()));
    }

    boolean capacityAvailable(RouteResult item, CapacityBoundary boundary) {
        return dispatcher.capacityAvailable(item, boundary);
    }

    // ==================== Delivery ownership ====================

    /**
     * Executes one finite scheduling turn and returns its next wait condition.
     */
    Decision scheduleOnce() {

        int maxRequests = settings().maxRequests();
        PrefillEndpoint.QueueSnapshot snapshot = captureQueue(maxRequests);
        RouteResult head = snapshot.head();
        if (head == null) return Decision.NO_WAIT;
        long nowMs = System.currentTimeMillis();
        if (head.requestExpired(nowMs)) {
            removeExpiredRequest(head);
            return Decision.NO_WAIT;
        }
        // Tail arrivals require a check, but cannot unblock this head's unavailable resource.
        Decision blocked = waiting;
        if (blocked != null
                && blocked.request() == head
                && blocked.unavailable() != null
                && !capacityAvailable(head, blocked.unavailable())) return blocked;
        GroupPlanner.Constraints constraints = schedulingConstraints();
        if (Math.max(0L, head.seqLen()) > constraints.batchKvCapacity()) {
            return Decision.awaitingSchedulingChange(
                    head, head.expiresAtMs(), "Prefill KV capacity exhausted");
        }
        for (int index = 1; index < Math.min(maxRequests, snapshot.items().size()); index++) {
            RouteResult member = snapshot.items().get(index);
            if (member.requestExpired(nowMs)) {
                removeExpiredRequest(member);
                return Decision.NO_WAIT;
            }
        }
        PrefillTimePredictor.Evaluator evaluator = prefillEndpoint().predictor.evaluator();
        return scheduleGroup(snapshot, evaluator, constraints);
    }

    /**
     * Notify the request after its queued resources have been released.
     */
    void removeExpiredRequest(RouteResult item) {
        if (!prefillEndpoint.releaseQueuedRequest(item)) return;
        notifyRemovedRequest(
                item,
                () ->
                        item.ctx()
                                .data()
                                .getRequestScheduler()
                                .cancel(item.ctx(), 0L, CancelReason.DEADLINE_EXCEEDED));
    }

    /**
     * Called by the queue's sole scheduling task after a failure; request callbacks run outside the endpoint lock.
     */
    Throwable failQueuedRequests(Throwable cause) {
        Throwable failure = null;
        List<RouteResult> removed = List.of();
        try {
            removed = prefillEndpoint.closeQueue();
        } catch (Throwable problem) {
            failure = problem;
        }
        failure = Failures.run(failure, () -> observeWait(null));
        if (!removed.isEmpty()) {
            failure =
                    Failures.run(
                            failure, () -> prefillEndpoint.signal(PrefillEndpoint.Signal.CAPACITY));
        }
        for (var route : removed) {
            failure =
                    Failures.run(
                            failure,
                            () ->
                                    route.ctx()
                                            .data()
                                            .getRequestScheduler()
                                            .rejectRoute(route, cause));
        }
        return failure;
    }

    Runnable watchCapacity(RouteResult item, CapacityBoundary boundary, Runnable wake) {
        return dispatcher.watchCapacity(item, boundary, wake);
    }

    private Decision scheduleGroup(
            PrefillEndpoint.QueueSnapshot snapshot,
            PrefillTimePredictor.Evaluator evaluator,
            GroupPlanner.Constraints constraints) {

        var candidates = snapshot.items();
        if (candidates.isEmpty()) {
            return Decision.NO_WAIT;
        }
        RouteResult first = candidates.getFirst();
        try {
            var predictor =
                    settings().executionBudgetMs() > 0L
                            ? RequestTimeEstimation.newGroupPredictor(
                                    settings().dispatcherType(), evaluator)
                            : null;
            long planningAtMs = System.currentTimeMillis();
            var selection = settings().grouping().select(candidates, constraints, predictor);
            checkState(
                    selection.items().size() <= candidates.size()
                            && java.util.stream.IntStream.range(0, selection.items().size())
                                    .allMatch(i -> selection.items().get(i) == candidates.get(i)),
                    "Prefill grouping must select a contiguous queue prefix");
            if (selection.items().isEmpty()) {
                return Decision.NO_WAIT;
            }
            String reason =
                    settings().grouping().dispatchReason(selection, constraints, planningAtMs);
            if (reason == null) {
                return Decision.awaitingSchedulingChange(
                        first,
                        Math.min(
                                settings().grouping().nextCheckAtMs(selection, constraints),
                                first.expiresAtMs()),
                        "Prefill collection window");
            }
            // Recheck advisory capacity after prediction, before acquiring hard-capacity leases.
            constraints = schedulingConstraints();
            if ((selection.items().size() > 1
                            && !selection.fitsCompute(constraints.batchTokenCapacity()))
                    || !selection.fitsKv(constraints.batchKvCapacity())) {
                return Decision.NO_WAIT;
            }
            OptionalLong prediction =
                    settings().dispatcherType() != DispatcherConfig.Type.BATCH
                                    || selection.selectedPredictionMs().isEmpty()
                            ? OptionalLong.empty()
                            : OptionalLong.of(
                                    PrefillTimePredictor.committedDecisionGroupMs(
                                            selection.selectedPredictionMs().getAsDouble()));
            CapacityBoundary boundary;
            try {
                boundary = processSelection(selection.items(), reason, evaluator, prediction);
            } catch (InvalidPrefillPredictionException failure) {
                first = selection.items().get(failure.memberIndex());
                boundary = CapacityBoundary.failed(failure);
            }
            return commitBoundary(first, boundary);
        } catch (InvalidPrefillPredictionException failure) {
            return commitBoundary(
                    candidates.get(failure.memberIndex()), CapacityBoundary.failed(failure));
        }
    }

    private void notifyRemovedRequest(RouteResult item, Runnable notifyRequest) {
        try {
            notifyRequest.run();
        } catch (Throwable callbackFailure) {
            item.ctx().data().getRequestScheduler().recordRequestFailure(callbackFailure);
            Logger.error(
                    "PrefillQueueScheduler[{}] removed request callback failed request_id={}",
                    key,
                    item.requestId(),
                    callbackFailure);
        }
    }

    /**
     * The sole queue task replaces subscriptions; shutdown clears them after all tasks finish.
     */
    private void updateCapacityListener(Decision blocked) {
        Runnable previous = removeCapacityListener;
        removeCapacityListener = null;
        try {
            if (previous != null) {
                previous.run();
            }
        } finally {
            if (blocked != null) {
                removeCapacityListener =
                        watchCapacity(
                                blocked.request(), blocked.unavailable(), capacityAvailableSignal);
            }
        }
    }

    /**
     * Wait condition after a scheduling turn; NO_WAIT also covers completed delivery or removal.
     */
    public record Decision(
            RouteResult request, CapacityBoundary unavailable, long wakeAtMs, String waitReason) {

        public static final Decision NO_WAIT = new Decision(null, null, 0L, null);

        public static Decision capacityBlocked(RouteResult item, CapacityBoundary unavailable) {
            return new Decision(
                    Objects.requireNonNull(item),
                    Objects.requireNonNull(unavailable),
                    Long.MAX_VALUE,
                    unavailable.resource().getDescription());
        }

        public static Decision awaitingSchedulingChange(
                RouteResult head, long wakeAtMs, String waitReason) {
            return new Decision(Objects.requireNonNull(head), null, wakeAtMs, waitReason);
        }
    }
}
