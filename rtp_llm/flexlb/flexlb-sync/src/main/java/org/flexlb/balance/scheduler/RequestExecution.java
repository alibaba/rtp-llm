package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.DecodeResources.DecodeRoutingView;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.InvalidPrefillPredictionException;
import org.flexlb.balance.prediction.PrefillBatchFeatures;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.QueueSnapshot.AdmissionBlock;
import org.flexlb.balance.projection.RequestTimeEstimation;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.scheduler.DeliveryDispatcher.Member;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalLong;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.math.LongMath.saturatedAdd;

/**
 * Submission of queued requests, coordinating Prefill, Decode and delivery capacity.
 * The owning RequestScheduler supplies every scheduling decision. NON_BATCH
 * publishes individual routes; BATCH sends an engine batch. DIRECT does not use this executor.
 * Endpoint state owns request identities and capacity.
 */
public final class RequestExecution {

    /** Frozen decision evidence; the log representation is materialized only on failure. */
    @lombok.RequiredArgsConstructor(access = lombok.AccessLevel.PRIVATE)
    private static final class QueueWaitSnapshot {
        private final String endpoint;
        private final String cause;
        private final long capturedAtMs = System.currentTimeMillis();
        private final PrefillState.QueueCounters counters;
        private final DecodeRoutingView decode;
        private volatile Map<String, Object> snapshotValues;

        private Map<String, Object> asMap() {
            Map<String, Object> cached = snapshotValues;
            if (cached != null) {
                return cached;
            }
            Map<Integer, Integer> priorityCounts = new HashMap<>();
            int depth = 0;
            int[] requestCountsByPriority = counters.byPriority();
            for (int priority = 0; priority < requestCountsByPriority.length; priority++) {
                if (requestCountsByPriority[priority] > 0) {
                    priorityCounts.put(priority, requestCountsByPriority[priority]);
                }
                depth += requestCountsByPriority[priority];
            }
            Map<String, Object> details = new LinkedHashMap<>();
            details.put("cause", cause);
            details.put("capturedAtMs", capturedAtMs);
            details.put("endpoint", endpoint);
            details.put("queueDepth", depth);
            details.put("queueVersion", counters.version());
            details.put("priorityCounts", Collections.unmodifiableMap(priorityCounts));
            details.put("prefillRequests", counters.outstandingRequests());
            details.put("prefillBatchSlots", counters.batchSlots());
            if (decode != null) {
                details.put(
                        "decode",
                        Map.of(
                                "endpoint",
                                decode.address(),
                                "version",
                                decode.admissionVersion(),
                                "engineLoad",
                                decode.engineLoad(),
                                "totalLoad",
                                decode.totalLoad(),
                                "kvTotal",
                                decode.totalKv(),
                                "kvAvailable",
                                decode.placementUsage().hardKvAvailable()));
            }
            cached = Collections.unmodifiableMap(details);
            snapshotValues = cached;
            return cached;
        }
    }

    private enum RuntimeState {
        NEW,
        STARTING,
        RUNNING,
        STOPPING,
        STOPPED
    }

    /** Exact predicate captured by one scheduling cycle. */
    public record Decision(
            RequestRoute request,
            CapacityBoundary unavailable,
            long queueVersion,
            long schedulingInputVersion,
            long wakeAtMs,
            String waitReason) {

        public static final Decision NO_ACTION = new Decision(null, null, 0L, 0L, 0L, null);

        public static Decision capacityBlocked(
                RequestRoute item, CapacityBoundary unavailable, long queueVersion) {
            return new Decision(
                    Objects.requireNonNull(item),
                    Objects.requireNonNull(unavailable),
                    queueVersion,
                    0L,
                    0L,
                    unavailable.resource().getDescription());
        }

        public static Decision awaitingSchedulingChange(
                RequestRoute head,
                long queueVersion,
                long schedulingInputVersion,
                long wakeAtMs,
                String waitReason) {
            return new Decision(
                    Objects.requireNonNull(head),
                    null,
                    queueVersion,
                    schedulingInputVersion,
                    wakeAtMs,
                    waitReason);
        }
    }

    private static final Map<String, Object> INITIAL_QUEUE_WAIT_SNAPSHOT =
            Map.of("cause", "waiting for Prefill decision");

    private static final RequestTimeEstimation.AdmissionBlockSemantics BATCH_DELIVERY_BLOCK =
            new RequestTimeEstimation.AdmissionBlockSemantics(
                    "DELIVERY_CAPACITY_BATCH_ADMISSION",
                    RequestTimeEstimation.AfterProbeAdmission.BLOCKED,
                    "DELIVERY_CAPACITY_BATCH_ADMISSION");
    private static final RequestTimeEstimation.AdmissionBlockSemantics DECODE_DELIVERY_BLOCK =
            new RequestTimeEstimation.AdmissionBlockSemantics(
                    "DELIVERY_CAPACITY_DECODE_ENGINE",
                    RequestTimeEstimation.AfterProbeAdmission.UNAVAILABLE,
                    "DECODE_CAPACITY_SCOPE_UNKNOWN");

    private final String key;
    private final PrefillEndpoint prefillEndpoint;
    private final QueueExecutionSettings settings;
    private final GroupPlanner grouping;
    private final DeliveryDispatcher dispatcher;
    private Runnable removeCapacityListener;

    /** Shared P ownership guard serializes queue changes, wait predicates and subscriptions. */
    private final PrefillState.QueueGuard queueGuard;

    /** Canonical request ownership guarded exclusively by {@link #queueGuard}. */
    private final PrefillState prefillState;

    private final Dependencies dependencies;

    public PrefillEndpoint prefillEndpoint() {
        return prefillEndpoint;
    }

    public PrefillState.QueueSnapshot captureQueue(int limit) {
        return prefillState.captureQueue(limit);
    }

    public QueueExecutionSettings settings() {
        return settings;
    }

    public GroupPlanner grouping() {
        return grouping;
    }

    /** Fixed facilities for this queue generation; request callbacks retain their own owner. */
    public static final class Dependencies {
        final java.util.function.Consumer<RequestExecution> run;
        final DeliveryExecution deliveries;
        final org.flexlb.service.monitor.DeliveryMetricsReporter reporter;
        final java.util.function.Consumer<Throwable> failures;

        public Dependencies(
                DeliveryExecution deliveries,
                org.flexlb.service.monitor.DeliveryMetricsReporter reporter,
                java.util.function.Consumer<Throwable> failures,
                java.util.function.Consumer<RequestExecution> run) {
            this.run = Objects.requireNonNull(run);
            this.deliveries = Objects.requireNonNull(deliveries);
            this.reporter = Objects.requireNonNull(reporter);
            this.failures = Objects.requireNonNull(failures);
        }
    }

    private final Thread workerThread;
    private final CancellationException normalStopFailure;
    private final IllegalStateException stopAcknowledgementFailure;
    private final CompletableFuture<Throwable> stopCompletion = new CompletableFuture<>();
    private Thread terminationOwner;
    private volatile RuntimeState runtimeState = RuntimeState.NEW;
    private volatile boolean drainClaimed;
    public final Runnable capacityAvailableSignal;

    /** Exact active head for which this worker is waiting on a capacity event. */
    private Decision capacityBlockedHead;

    /** Last waiting decision; timeout readers never acquire queueGuard. */
    private volatile QueueWaitSnapshot latestQueueWaitSnapshot;

    public RequestExecution(
            String key,
            PrefillEndpoint prefillEp,
            QueueExecutionSettings settings,
            DeliveryDispatcher dispatcher,
            PrefillState prefillState,
            Dependencies dependencies) {
        this.key = key;
        this.dependencies = Objects.requireNonNull(dependencies);
        this.prefillEndpoint = Objects.requireNonNull(prefillEp, "prefillEndpoint");
        this.settings = Objects.requireNonNull(settings);
        this.prefillState = Objects.requireNonNull(prefillState, "prefillState");
        this.queueGuard = prefillState.queueGuard();
        this.capacityAvailableSignal =
                () -> {
                    signal(true);
                    prefillEndpoint.signal(PrefillEndpoint.Signal.PLACEMENT);
                };
        this.grouping = settings.grouping();
        this.dispatcher = Objects.requireNonNull(dispatcher, "dispatcher");
        this.normalStopFailure =
                new CancellationException("FlexLB worker scheduling queue stopped: " + key);
        this.stopAcknowledgementFailure =
                new IllegalStateException(
                        "FlexLB worker stop callback lost its exact retained owner: " + key);
        this.workerThread =
                Thread.ofVirtual()
                        .name("flexlb-prefill-queue-" + key)
                        .uncaughtExceptionHandler(
                                (thread, failure) ->
                                        Logger.error(
                                                "RequestExecution[{}] thread died unexpectedly",
                                                key,
                                                failure))
                        .unstarted(this::run);
    }

    private void run() {
        Throwable terminalFailure = normalStopFailure;
        try {
            dependencies.run.accept(this);
        } catch (Throwable failure) {
            terminalFailure = failure;
        } finally {
            try {
                Throwable cleanup = stopAndDrain(terminalFailure, false);
                if (terminalFailure != normalStopFailure) {
                    Failures.append(terminalFailure, cleanup);
                    Logger.error("RequestExecution[{}] scheduling failed", key, terminalFailure);
                } else if (cleanup != null) {
                    Logger.error("RequestExecution[{}] shutdown failed", key, cleanup);
                }
            } finally {
                markStopped();
            }
        }
    }

    public boolean stopping() {
        return drainClaimed;
    }

    private synchronized void markStopped() {
        runtimeState = RuntimeState.STOPPED;
    }

    public synchronized void start() {
        if (runtimeState != RuntimeState.NEW) {
            throw new IllegalStateException(
                    "Request execution cannot start from " + runtimeState);
        }
        runtimeState = RuntimeState.STARTING;
        try {
            workerThread.start();
            runtimeState = RuntimeState.RUNNING;
        } catch (RuntimeException | Error startFailure) {
            Throwable cleanupFailure = stopAndDrain(startFailure, false);
            runtimeState = RuntimeState.STOPPED;
            Failures.append(startFailure, cleanupFailure);
            throw startFailure;
        }
    }

    public boolean offer(RequestRoute item) {
        checkArgument(
                item.prefillEp() == prefillEndpoint,
                "incoming item belongs to another Prefill generation");
        if (runtimeState != RuntimeState.RUNNING || drainClaimed) {
            return false;
        }
        boolean accepted;
        queueGuard.lock();
        try {
            if (drainClaimed || prefillEndpoint.isGenerationRetiringOrRetired()) {
                return false;
            }
            accepted =
                    prefillState.updateQueue(
                                    PrefillState.QueueChange.ENQUEUE,
                                    item,
                                    System.currentTimeMillis())
                            != null;
            if (accepted) {
                queueGuard.signal();
            }
        } finally {
            queueGuard.unlock();
        }
        return accepted;
    }

    public Map<Integer, Integer> queueSizeByPriority() {
        Map<Integer, Integer> sizeByPriority = new HashMap<>();
        int[] counts = prefillState.captureQueueCounters().byPriority();
        for (int priority = 0; priority < counts.length; priority++) {
            if (counts[priority] > 0) {
                sizeByPriority.put(priority, counts[priority]);
            }
        }
        return sizeByPriority;
    }

    /** Caller holds P lock; availability only probes the already subscribed resource. */
    public AdmissionBlock admissionBlockLocked() {
        queueGuard.requireHeld();
        Decision blocked = capacityBlockedHead;
        return blocked == null
                        || !prefillState.queueWaitCurrentLocked(
                                blocked.request(), 0L, 0L, true, System.currentTimeMillis())
                        || capacityAvailable(blocked.request(), blocked.unavailable())
                ? null
                : new AdmissionBlock(
                        blocked.request().requestId(),
                        blocked.request().enqueueSeq(),
                        projectionSemantics(blocked.unavailable().resource()));
    }

    private static RequestTimeEstimation.AdmissionBlockSemantics projectionSemantics(
            CapacityBoundary.Resource resource) {
        return switch (resource) {
            case DISPATCH_EXECUTOR -> BATCH_DELIVERY_BLOCK;
            case DECODE_ENGINE -> DECODE_DELIVERY_BLOCK;
            case PREFILL_BATCH, PREFILL_REQUEST -> null;
        };
    }

    public Throwable stopAndAwait() {
        checkState(
                Thread.currentThread() != workerThread,
                "Prefill runtime cannot await its own worker thread");
        Throwable failure = stopAndDrain(normalStopFailure, true);
        boolean interrupted = false;
        while (true) {
            try {
                workerThread.join();
                break;
            } catch (InterruptedException interruption) {
                interrupted = true;
            }
        }
        synchronized (this) {
            runtimeState = RuntimeState.STOPPED;
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
        }
        return failure;
    }

    Throwable stopAndDrain(Throwable terminalFailure, boolean interruptWorker) {
        boolean alreadyStopping;
        synchronized (this) {
            alreadyStopping = drainClaimed;
            if (alreadyStopping) {
                checkState(
                        stopCompletion.isDone() || terminationOwner != Thread.currentThread(),
                        "Prefill runtime cannot await its active stop transaction");
            } else {
                drainClaimed = true;
                terminationOwner = Thread.currentThread();
                runtimeState =
                        runtimeState == RuntimeState.NEW
                                ? RuntimeState.STOPPED
                                : RuntimeState.STOPPING;
            }
        }
        if (alreadyStopping) {
            // join is uninterruptible and preserves the caller's interrupt flag.
            return stopCompletion.join();
        }

        Throwable cleanupFailure = null;
        try {
            try {
                queueGuard.lock();
                try {
                    cleanupFailure =
                            Failures.run(cleanupFailure, () -> setCapacityBlockedHeadLocked(null));
                    queueGuard.signalAll();
                } catch (Throwable wakeFailure) {
                    cleanupFailure = Failures.append(cleanupFailure, wakeFailure);
                } finally {
                    cleanupFailure = Failures.run(cleanupFailure, queueGuard::unlock);
                }
            } catch (Throwable lockFailure) {
                cleanupFailure = Failures.append(cleanupFailure, lockFailure);
            }

            if (interruptWorker && Thread.currentThread() != workerThread) {
                cleanupFailure = Failures.run(cleanupFailure, workerThread::interrupt);
            }

            while (true) {
                RequestRoute item;
                try {
                    item =
                            prefillState.updateQueue(
                                    PrefillState.QueueChange.STOP,
                                    null,
                                    System.currentTimeMillis());
                } catch (Throwable claimFailure) {
                    cleanupFailure = Failures.append(cleanupFailure, claimFailure);
                    break;
                }
                if (item == null) {
                    break;
                }
                cleanupFailure = Failures.run(cleanupFailure, capacityAvailableSignal);
                try {
                    item.ctx()
                            .data()
                            .getScheduler()
                            .onQueueEvent(
                                    item, RequestLifecycle.QueueKind.REJECTED, terminalFailure);
                    boolean acknowledged =
                            prefillState.updateQueue(
                                            PrefillState.QueueChange.ACK_STOP,
                                            item,
                                            System.currentTimeMillis())
                                    != null;
                    if (!acknowledged) {
                        cleanupFailure =
                                Failures.append(cleanupFailure, stopAcknowledgementFailure);
                    }
                } catch (Throwable settlementFailure) {
                    // A failed callback is never acknowledged. A failed acknowledgement
                    // leaves the exact owner for retirement; the stop owner reports failures.
                    cleanupFailure = Failures.append(cleanupFailure, settlementFailure);
                }
            }
        } catch (Throwable unexpectedCleanupFailure) {
            cleanupFailure = Failures.append(cleanupFailure, unexpectedCleanupFailure);
        } finally {
            synchronized (this) {
                terminationOwner = null;
                stopCompletion.complete(cleanupFailure);
            }
        }
        return cleanupFailure;
    }

    // ==================== endpoint queue operations ====================

    public boolean removeQueued(RequestRoute item) {
        checkArgument(
                item.prefillEp() == prefillEndpoint,
                "request belongs to another Prefill generation");
        boolean accepted;
        queueGuard.lock();
        try {
            if (drainClaimed || runtimeState != RuntimeState.RUNNING) {
                return false;
            }
            accepted =
                    prefillState.updateQueue(
                                    PrefillState.QueueChange.REMOVE,
                                    item,
                                    System.currentTimeMillis())
                            != null;
            if (accepted) {
                queueGuard.signal();
            }
        } finally {
            queueGuard.unlock();
        }
        if (accepted) {
            capacityAvailableSignal.run();
        }
        return accepted;
    }

    /** Read the last captured queue wait state without waiting or acquiring the queue lock. */
    public Map<String, Object> getLatestQueueWaitSnapshot() {
        QueueWaitSnapshot snapshot = latestQueueWaitSnapshot;
        return snapshot == null ? INITIAL_QUEUE_WAIT_SNAPSHOT : snapshot.asMap();
    }

    /** Capture fixed-size counters without formatting the PV record on the scheduling loop. */
    public void recordQueueWait(RequestRoute head, String reason) {
        PrefillState.QueueCounters counters = prefillState.captureQueueCounters();
        DecodeRoutingView decode = head.decodeEp() == null ? null : head.decodeEp().routingView();
        latestQueueWaitSnapshot = new QueueWaitSnapshot(key, reason, counters, decode);
    }

    // ==================== Queue ownership and projection ====================

    public GroupPlanner.Constraints schedulingConstraints() {
        WorkerStatus status = prefillEndpoint.getStatus();
        var engine = status == null ? null : status.committedEngineObservation();
        return settings.constraints(engine);
    }

    // ==================== Delivery ownership ====================

    /** Prepare an ordered prefix, commit once, and hand off outside all owner locks. */
    public CapacityBoundary processSelection(
            List<RequestRoute> candidates,
            String reason,
            PrefillTimePredictor.Evaluator evaluator,
            OptionalLong plannedPrediction,
            RequestRoute bypassedHead) {
        queueGuard.lock();
        try {
            if (candidates.isEmpty()
                    || drainClaimed
                    || !prefillState.ownsSelectionLocked(candidates, System.currentTimeMillis())) {
                return null;
            }
        } finally {
            queueGuard.unlock();
        }
        Preparation prep = new Preparation(candidates.size());
        try (prep) {
            try {
                prep.acquire(candidates, evaluator);
                if (prep.members.isEmpty()) {
                    return prep.boundary;
                }
                long prediction =
                        prep.commit(evaluator, plannedPrediction, candidates.size(), bypassedHead);
                if (prep.committed == null) {
                    return !prep.batch
                                    && !prefillState.requestCapacityAvailable(
                                            prefillEndpoint.requestLimit, 1)
                            ? CapacityBoundary.unavailable(
                                    CapacityBoundary.Resource.PREFILL_REQUEST)
                            : null;
                }
                prep.reservation = null;
                boolean removedFailedMember = prep.committed.removedFailedMember();
                if (removedFailedMember) {
                    removeAndNotify(prep.blocked, prep.boundary, true);
                }
                WorkSnapshot precedingWork = prep.committed.precedingWork().materialize();
                if (prep.batch) {
                    var command =
                            new DeliveryDispatcher.Batch(
                                    prep.batchId,
                                    prep.members,
                                    precedingWork,
                                    prediction,
                                    evaluator,
                                    prep.boundary != null && prep.boundary.unavailable()
                                            ? "delivery_capacity_prefix"
                                            : reason,
                                    prep.committed.remainingQueueDepth(),
                                    prep.committed,
                                    prep.executionPermit);
                    prep.transferred = true;
                    dispatcher.deliverBatch(command);
                } else {
                    publishRoutes(prep.members, prep.predictions, precedingWork, prep.committed);
                    prep.committed = null;
                }
                if (!removedFailedMember) {
                    prefillEndpoint.signal(PrefillEndpoint.Signal.PLACEMENT);
                }
                return null;
            } catch (Throwable failure) {
                if (prep.committed != null && !prep.transferred) {
                    String detail = String.valueOf(failure.getMessage());
                    for (Member member : prep.members) {
                        failure =
                                Failures.run(
                                        failure,
                                        () ->
                                                RequestLifecycle.preparationFailed(
                                                        member.route(), detail));
                    }
                }
                if (prep.committed != null) {
                    prep.primaryFailure = failure;
                    Logger.error("Committed delivery failed", failure);
                    return null;
                }
                throw Failures.propagate(failure, "delivery preparation failed");
            }
        }
    }

    /** Route publication and its committed queue owner finish before telemetry. */
    private void publishRoutes(
            List<Member> members,
            long[] predictions,
            WorkSnapshot precedingWork,
            PrefillState.CommittedHandoff committed) {
        DeliveryClaim[] claims = new DeliveryClaim[members.size()];
        for (int index = 0; index < members.size(); index++) {
            Member member = members.get(index);
            try {
                claims[index] = member.claimDelivery(DeliveryClaimKind.ROUTE_DECISION, 0L);
            } catch (Throwable failure) {
                try {
                    RequestLifecycle.preparationFailed(
                            member.route(), String.valueOf(failure.getMessage()));
                } catch (Throwable ownerFailure) {
                    dependencies.failures.accept(ownerFailure);
                }
            }
        }
        long unstarted = 0L;
        int delivered = 0;
        Throwable failure = null;
        for (int index = 0; index < claims.length; index++) {
            DeliveryClaim claim = claims[index];
            if (claim == null) {
                continue;
            }
            unstarted = saturatedAdd(unstarted, predictions[index]);
            try {
                long nowMs = System.currentTimeMillis();
                var data = claim.owner().data();
                data.getScheduler()
                        .onDeliveryFact(
                                new ContextResult.RouteDelivered(
                                        claim,
                                        PrefillPredictionBoundary.predictedVisibilityDeadlineMs(
                                                precedingWork,
                                                unstarted,
                                                data.getConfig()
                                                        .getRequestLifecycle()
                                                        .getDecision()
                                                        .getLifetime(),
                                                nowMs),
                                        nowMs));
                delivered++;
            } catch (Throwable problem) {
                claims[index] = null;
                failure = Failures.append(failure, problem);
            }
        }
        failure = Failures.append(failure, DeliveryDispatcher.closeCommitted(members, committed));
        members.clear();
        if (failure != null) {
            dependencies.failures.accept(failure);
            try {
                Logger.error("Route publication failed", failure);
            } catch (Throwable ignored) {
            }
        }
        reportRoutes(claims, delivered, committed.remainingQueueDepth());
    }

    private void reportRoutes(DeliveryClaim[] claims, int delivered, int remainingQueueDepth) {
        if (delivered != 0) {
            try {
                long nowMs = System.currentTimeMillis();
                String engineIp = prefillEndpoint.getIp();
                for (var claim : claims) {
                    if (claim != null) {
                        dependencies.reporter.reportBatchWaitTimeMs(
                                RoleType.PREFILL.name(),
                                engineIp,
                                Math.max(0L, nowMs - claim.item().enqueuedAtMs()),
                                claim.item().priority());
                    }
                }
                dependencies.reporter.reportBatcherQueueSize(
                        RoleType.PREFILL.name(), engineIp, remainingQueueDepth);
            } catch (Throwable monitoringFailure) {
                Logger.error("Route delivery telemetry failed", monitoringFailure);
            }
        }
    }

    boolean capacityAvailable(RequestRoute item, CapacityBoundary boundary) {
        return switch (boundary.resource()) {
            case DISPATCH_EXECUTOR -> dependencies.deliveries.deliveryCapacityAvailable();
            case PREFILL_REQUEST ->
                    prefillState.requestCapacityAvailable(prefillEndpoint.requestLimit, 1);
            case PREFILL_BATCH ->
                    prefillEndpoint.ownership.batchCapacityAvailable(
                            dependencies.deliveries.maxInflightBatches);
            case DECODE_ENGINE ->
                    item.decodeEp()
                            .shouldRetryDispatch(
                                    item.decodeReservation(),
                                    dependencies.deliveries.decodeDeliveryCapacity);
        };
    }

    Runnable watchCapacity(RequestRoute item, CapacityBoundary boundary, Runnable wake) {
        return switch (boundary.resource()) {
            case DISPATCH_EXECUTOR -> dependencies.deliveries.watchDeliveryCapacity(wake);
            case DECODE_ENGINE -> {
                yield item.decodeEp().watchEngineDispatchCapacity(wake);
            }
            case PREFILL_BATCH, PREFILL_REQUEST -> () -> {};
        };
    }

    /** Serialize a delivery commit with queue drain and exact generation ownership. */
    PrefillState.CommittedHandoff commitSelection(
            java.util.function.LongFunction<PrefillState.CommittedHandoff> commit) {
        queueGuard.lock();
        try {
            return drainClaimed ? null : commit.apply(System.currentTimeMillis());
        } finally {
            queueGuard.unlock();
        }
    }

    public Decision commitBoundary(
            RequestRoute blockedItem, CapacityBoundary blockedResult, long queueVersion) {
        if (blockedItem == null || blockedResult == null) {
            return Decision.NO_ACTION;
        }
        boolean capacityBlocked = blockedResult.unavailable();
        boolean failed = blockedResult.status() == CapacityBoundary.Status.FAILED;
        boolean removed = false;
        queueGuard.lock();
        try {
            if (drainClaimed) {
                return Decision.NO_ACTION;
            }
            long nowMs = System.currentTimeMillis();
            if (capacityBlocked) {
                if (!prefillState.queueWaitCurrentLocked(blockedItem, 0L, 0L, true, nowMs)) {
                    return Decision.NO_ACTION;
                }
            } else if (failed) {
                removed =
                        prefillState.updateQueue(
                                        PrefillState.QueueChange.REMOVE_UNEXPIRED,
                                        blockedItem,
                                        nowMs)
                                != null;
            }
        } finally {
            queueGuard.unlock();
        }
        if (capacityBlocked) {
            return Decision.capacityBlocked(blockedItem, blockedResult, queueVersion);
        }
        if (removed) {
            removeAndNotify(blockedItem, blockedResult, true);
        }
        return Decision.NO_ACTION;
    }

    /** Once removed, the request must be notified even if a capacity listener fails. */
    void removeAndNotify(RequestRoute item, CapacityBoundary failure, boolean alreadyRemoved) {
        if (!alreadyRemoved
                && prefillState.updateQueue(
                                PrefillState.QueueChange.REMOVE, item, System.currentTimeMillis())
                        == null) {
            return;
        }
        Throwable problem = Failures.run(null, capacityAvailableSignal);
        try {
            if (failure == null) {
                item.ctx()
                        .data()
                        .getScheduler()
                        .onQueueEvent(item, RequestLifecycle.QueueKind.EXPIRED, null);
            } else {
                RequestLifecycle.preparationFailed(
                        item, String.valueOf(failure.cause().getMessage()));
            }
        } catch (Throwable callbackFailure) {
            dependencies.failures.accept(callbackFailure);
            problem =
                    Failures.run(
                            problem,
                            () ->
                                    Logger.error(
                                            "RequestExecution[{}] removed request callback failed request_id={}",
                                            key,
                                            item.requestId(),
                                            callbackFailure));
        }
        Failures.rethrow(problem, "queue removal notification failed");
    }

    // ==================== Group decisions ====================

    /** Checks ordering and the queue-owned allowance; it neither reserves nor consumes capacity. */
    boolean queueAllowsBypass(RequestRoute head, RequestRoute candidate) {
        queueGuard.lock();
        try {
            return prefillState.canBypassLocked(head, candidate);
        } finally {
            queueGuard.unlock();
        }
    }

    // ==================== Internal: Run loop ====================

    /**
     * All waits share the queue lock and exact queue predicate. Capacity waits
     * subscribe before testing availability, so release-before-await is safe.
     * A null decision waits for the first item; other decisions retain their
     * exact head and either resource or queue/input-version predicate.
     */
    public void awaitWork(Decision waiting) throws InterruptedException {
        boolean capacityWait = waiting != null && waiting.unavailable() != null;
        queueGuard.lockInterruptibly();
        try {
            if (capacityWait && !drainClaimed) {
                setCapacityBlockedHeadLocked(waiting);
            }
            while (!drainClaimed
                    && prefillState.queueWaitCurrentLocked(
                            waiting == null ? null : waiting.request(),
                            waiting == null ? 0L : waiting.queueVersion(),
                            waiting == null ? 0L : waiting.schedulingInputVersion(),
                            capacityWait,
                            System.currentTimeMillis())
                    && (!capacityWait
                            || !capacityAvailable(waiting.request(), waiting.unavailable()))) {
                if (waiting == null) {
                    setCapacityBlockedHeadLocked(null);
                }
                long wakeAtMs =
                        waiting == null
                                ? Long.MAX_VALUE
                                : capacityWait
                                        ? waiting.request().expiresAtMs()
                                        : waiting.wakeAtMs();
                long nowMs = System.currentTimeMillis();
                if (wakeAtMs <= nowMs) {
                    return;
                }
                if (wakeAtMs == Long.MAX_VALUE) {
                    queueGuard.await();
                } else {
                    queueGuard.awaitNanos(TimeUnit.MILLISECONDS.toNanos(wakeAtMs - nowMs));
                }
            }
        } finally {
            try {
                if (capacityWait) {
                    setCapacityBlockedHeadLocked(null);
                }
            } finally {
                queueGuard.unlock();
            }
        }
    }

    /** Called after Prefill or Decode capacity changes, outside endpoint locks. */

    /** Own the blocked head, its listener and projection invalidation under queueGuard. */
    private void setCapacityBlockedHeadLocked(Decision blocked) {
        queueGuard.requireHeld();
        if (capacityBlockedHead == blocked) {
            return;
        }
        Runnable previous = removeCapacityListener;
        removeCapacityListener = null;
        capacityBlockedHead = blocked;
        prefillState.schedulingInputsChangedLocked();
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

    /** Wake after the status transaction has already committed its planning revision. */
    public void wake() {
        queueGuard.lock();
        try {
            queueGuard.signal();
        } finally {
            queueGuard.unlock();
        }
    }

    /** Wake decisions whose advisory worker-status or predictor input changed. */
    public void signal(boolean capacityOnly) {
        queueGuard.lock();
        try {
            if (!capacityOnly || capacityBlockedHead != null) {
                prefillState.schedulingInputsChangedLocked();
                queueGuard.signal();
            }
        } finally {
            queueGuard.unlock();
        }
    }

    private final class Preparation implements AutoCloseable {
        final boolean batch = settings.dispatcherType() == DispatcherConfig.Type.BATCH;
        final ArrayList<Member> members;
        final long[] predictions;
        DeliveryExecution.DeliveryPermit executionPermit;
        PrefillState.BatchReservation reservation;
        PrefillState.CommittedHandoff committed;
        long batchId;
        boolean transferred;
        Throwable primaryFailure;
        CapacityBoundary boundary;
        RequestRoute blocked;

        Preparation(int size) {
            members = new ArrayList<>(size);
            predictions = batch ? null : new long[size];
        }

        void acquire(List<RequestRoute> candidates, PrefillTimePredictor.Evaluator evaluator) {
            long payloadBytes = 16L;
            for (RequestRoute item : candidates) {
                long itemBytes = batch ? item.batchPayloadSizeUpperBound() : 0L;
                if (batch
                        && !members.isEmpty()
                        && itemBytes
                                > org.flexlb.constant.GrpcConstants.MAX_MESSAGE_SIZE
                                        - payloadBytes) {
                    break;
                }
                synchronized (item.ctx()) {
                    var state = item.ctx().snapshot();
                    boundary =
                            state.preparedDelivery() && state.route() == item
                                    ? acquire(item, evaluator)
                                    : CapacityBoundary.OWNERSHIP_LOST;
                    if (boundary != null) {
                        blocked = item;
                        break;
                    }
                    payloadBytes += itemBytes;
                }
            }
        }

        long commit(
                PrefillTimePredictor.Evaluator evaluator,
                OptionalLong plannedPrediction,
                int candidateCount,
                RequestRoute bypassedHead) {
            List<RequestRoute> items = members.stream().map(Member::route).toList();
            long prediction =
                    !batch
                            ? 0L
                            : plannedPrediction.isPresent() && items.size() == candidateCount
                                    ? plannedPrediction.getAsLong()
                                    : PrefillPredictionBoundary.predictCommittedBatchMs(
                                            evaluator,
                                            PrefillBatchFeatures.from(
                                                    items,
                                                    RequestRoute::seqLen,
                                                    RequestRoute::hitCache));
            long[] committedPredictions =
                    batch
                            ? new long[] {prediction}
                            : predictions.length == items.size()
                                    ? predictions
                                    : Arrays.copyOf(predictions, items.size());
            RequestRoute failed =
                    boundary != null && boundary.status() == CapacityBoundary.Status.FAILED
                            ? blocked
                            : null;
            committed =
                    commitSelection(
                            now -> {
                                if (bypassedHead != null
                                        && bypassedHead
                                                .decodeEp()
                                                .shouldRetryDispatch(
                                                        bypassedHead.decodeReservation(),
                                                        dependencies
                                                                .deliveries
                                                                .decodeDeliveryCapacity)) {
                                    return null;
                                }
                                return prefillEndpoint.commitLocked(
                                        new PrefillState.CommitPlan(
                                                batch
                                                        ? PrefillState.CommitKind.BATCH
                                                        : PrefillState.CommitKind.QUEUED,
                                                items,
                                                List.of(),
                                                reservation,
                                                committedPredictions,
                                                failed,
                                                now,
                                                bypassedHead));
                            });
            return prediction;
        }

        CapacityBoundary acquire(RequestRoute item, PrefillTimePredictor.Evaluator evaluator) {
            if (!batch
                    && !prefillState.requestCapacityAvailable(
                            prefillEndpoint.requestLimit, members.size() + 1)) {
                return CapacityBoundary.unavailable(CapacityBoundary.Resource.PREFILL_REQUEST);
            }
            if (batch && members.isEmpty()) {
                try {
                    var execution = dependencies.deliveries.tryReserveDelivery();
                    if (!execution.accepted()) {
                        return execution.boundary();
                    }
                    executionPermit = execution.value();
                    batchId = dependencies.deliveries.nextBatchId();
                    checkState(batchId > 0L, "batch id supplier returned a non-positive id");
                    var acquired =
                            prefillEndpoint.reserveBatch(
                                    item, batchId, dependencies.deliveries.maxInflightBatches);
                    if (acquired.status() != PrefillState.CapacityStatus.ACQUIRED) {
                        return switch (acquired.status()) {
                            case CAPACITY_FULL ->
                                    CapacityBoundary.unavailable(
                                            CapacityBoundary.Resource.PREFILL_BATCH);
                            case REQUEST_NOT_ACTIVE -> CapacityBoundary.OWNERSHIP_LOST;
                            default ->
                                    CapacityBoundary.failed(
                                            new IllegalStateException(
                                                    "Prefill batch admission failed: request_id="
                                                            + item.requestId()
                                                            + " status="
                                                            + acquired.status()));
                        };
                    }
                    reservation = acquired.reservation();
                } catch (Throwable failure) {
                    return CapacityBoundary.failed(failure);
                }
            }
            long prediction;
            try {
                prediction =
                        batch
                                ? 0L
                                : PrefillPredictionBoundary.predictSingleRequestMs(
                                        evaluator, item.seqLen(), item.hitCache());
            } catch (InvalidPrefillPredictionException failure) {
                return CapacityBoundary.failed(failure);
            }
            var attempt = Member.prepare(item, dependencies.deliveries.decodeDeliveryCapacity);
            if (!attempt.accepted()) {
                return attempt.boundary();
            }
            Member member = attempt.value();
            try {
                if (!batch) {
                    predictions[members.size()] = prediction;
                }
                members.add(member);
                return null;
            } catch (Throwable failure) {
                return CapacityBoundary.failed(Failures.run(failure, member::close));
            }
        }

        @Override
        public void close() {
            if (transferred) {
                return;
            }
            Throwable failure = null;
            for (Member member : members) {
                failure = Failures.run(failure, member::close);
            }
            if (reservation != null) {
                failure = Failures.run(failure, () -> prefillEndpoint.release(reservation, null));
            }
            if (committed != null) {
                failure = Failures.append(failure, Failures.close(committed));
            }
            if (executionPermit != null) {
                failure = Failures.append(failure, Failures.close(executionPermit));
            }
            if (primaryFailure != null) {
                Failures.append(primaryFailure, failure);
            } else {
                Failures.rethrow(failure, "temporary delivery resource cleanup failed");
            }
        }
    }
}
