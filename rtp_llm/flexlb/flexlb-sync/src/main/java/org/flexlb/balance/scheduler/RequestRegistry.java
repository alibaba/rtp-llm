package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.scheduler.RequestSlot.AdmissionHandle;
import org.flexlb.balance.scheduler.RequestSlot.DeliveryClaim;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.BiConsumer;
import java.util.function.BooleanSupplier;
import java.util.function.LongPredicate;

import static org.flexlb.dao.loadbalance.Response.buildErrorResponse;

/**
 * Canonical request directory and global admission barrier.
 * ID/item entry points resolve an exact slot and dispatch one event; request
 * policy is implemented by RequestSlot. Removal invalidates the slot under
 * its monitor so retained capabilities cannot operate on a replacement.
 */
@Component
public class RequestRegistry {

    private final RequestCompletionPublisher completionPublisher;
    private final RequestTerminalCleanup terminalCleanup;

    private final ExpirationTimer expirationTimer;

    private final AtomicBoolean shuttingDown = new AtomicBoolean();

    private final Object admissionQuiescenceMonitor = new Object();
    private int inFlightAdmissionHandles;
    private volatile GlobalQueueCoordinator globalQueue;

    private final Object registrationLock = new Object();
    private final BatchSchedulerReporter reporter;
    private final RequestSchedulerReporter requestReporter;

    private final ConcurrentMap<Long, RequestSlot> requestSlots =
            new ConcurrentHashMap<>();
    @Autowired
    public RequestRegistry(ConfigService configService,
                            BatchSchedulerReporter reporter,
                            RequestSchedulerReporter requestReporter) {
        this.reporter = Objects.requireNonNull(reporter, "reporter");
        this.requestReporter = Objects.requireNonNull(requestReporter);
        this.expirationTimer = new ExpirationTimer(
                this,
                Objects.requireNonNull(configService, "configService"));
        this.completionPublisher = new RequestCompletionPublisher(
                completionPublisherWorkers(configService), reporter);
        this.terminalCleanup = new RequestTerminalCleanup(expirationTimer);
    }

    void attachGlobalQueue(GlobalQueueCoordinator queue) {
        if (globalQueue != null) { throw new IllegalStateException("global queue already attached"); }
        globalQueue = Objects.requireNonNull(queue, "queue");
    }

    private record WithdrawnRoute(RequestSlot slot, ScheduledRequest item, AdmissionHandle claim) { }

    /** Transfer queued Decode capacity, then return each victim to its original global queue identity. */
    public boolean replaceQueuedDecodeReservations(
            DecodeEndpoint endpoint, List<DecodeEndpoint.ReservationHandle> victims,
            long incomingRequestId, long hardKv, long expectedKv, int priority,
            DecodeEndpoint.AdmissionCapacity capacity) {
        GlobalQueueCoordinator queue = globalQueue;
        if (queue == null || shuttingDown.get()) { return false; }
        List<WithdrawnRoute> claimed = new ArrayList<>(victims.size());
        boolean replaced = false;
        try {
            for (DecodeEndpoint.ReservationHandle victim : victims) {
                WithdrawnRoute withdrawal = claimQueuedRoute(endpoint, victim, priority);
                if (withdrawal == null) { return false; }
                try {
                    claimed.add(withdrawal);
                } catch (Throwable failure) {
                    withdrawal.claim().close();
                    throw failure;
                }
            }
            replaced = endpoint.replaceQueuedRequests(
                    victims, incomingRequestId, hardKv, expectedKv, priority, capacity);
            return replaced;
        } finally {
            Throwable failure = null;
            for (WithdrawnRoute withdrawal : claimed) {
                try {
                    if (replaced) {
                        withdrawal.slot().detachWithdrawnRoute(withdrawal.claim(), withdrawal.item());
                    }
                } catch (Throwable detachFailure) {
                    failure = RequestTerminalCleanup.appendFailure(failure, detachFailure);
                    try {
                        withdrawal.claim().terminate(buildErrorResponse(
                                StrategyErrorType.DISPATCH_FAILED, "queued route withdrawal failed"));
                    } catch (Throwable terminalFailure) {
                        failure = RequestTerminalCleanup.appendFailure(failure, terminalFailure);
                    }
                } finally {
                    try {
                        withdrawal.claim().close();
                        if (replaced && withdrawal.slot().isOpen()
                                && !queue.requeue(withdrawal.item())) {
                            completeError(withdrawal.item().future(), StrategyErrorType.DISPATCH_FAILED,
                                    "scheduler closed during route withdrawal");
                        }
                    } catch (Throwable closeFailure) {
                        failure = RequestTerminalCleanup.appendFailure(failure, closeFailure);
                        try {
                            completeError(withdrawal.item().future(), StrategyErrorType.DISPATCH_FAILED,
                                    "queued route requeue failed");
                        } catch (Throwable terminalFailure) {
                            failure = RequestTerminalCleanup.appendFailure(failure, terminalFailure);
                        }
                    }
                }
            }
            if (failure != null) {
                if (replaced) {
                    DecodeEndpoint.ReservationHandle incoming = endpoint.reservationHandle(incomingRequestId);
                    if (incoming != null) { endpoint.release(incoming, DecodeEndpoint.ReleaseReason.LOCAL_ROLLBACK); }
                }
                RequestTerminalCleanup.rethrowCleanup(failure);
            }
        }
    }

    private WithdrawnRoute claimQueuedRoute(DecodeEndpoint endpoint,
                                            DecodeEndpoint.ReservationHandle victim, int incomingPriority) {
        if (!enterAdmissionHandleGate()) { return null; }
        AdmissionHandle claim = null;
        boolean retained = false;
        try {
            RequestSlot slot = requestSlots.get(victim.requestId());
            if (slot == null) { return null; }
            synchronized (slot) {
                if (!isCurrentSlot(slot)) { return null; }
                ScheduledRequest active = slot.activeItem();
                if (active == null || active.priority() >= incomingPriority) { return null; }
                claim = slot.tryBeginRouteWithdrawal(endpoint, victim);
                if (claim == null) { return null; }
                WithdrawnRoute withdrawal = new WithdrawnRoute(slot, active, claim);
                retained = true;
                return withdrawal;
            }
        } finally {
            if (!retained) {
                if (claim == null) { exitAdmissionHandleGate(); } else { claim.close(); }
            }
        }
    }

    private static int completionPublisherWorkers(ConfigService configService) {
        try {
            FlexlbConfig config = configService.loadBalanceConfig();
            return config == null || config.getInternalRuntime() == null
                    ? 0 : config.getInternalRuntime()
                            .getBatchDispatchCompletionThreads();
        } catch (Throwable ignored) {
            return 0;
        }
    }

    boolean isCurrentSlot(RequestSlot slot) {
        return slot != null && requestSlots.get(slot.requestId()) == slot;
    }

    RequestSlot requestSlot(long requestId) {
        return requestSlots.get(requestId);
    }

    public boolean isShuttingDown() {
        return shuttingDown.get();
    }

    public List<RequestSlot> snapshotSlots() {
        return List.copyOf(requestSlots.values());
    }

    public boolean removeExactTerminalRecord(
            RequestSlot exactSlot, long updatedBeforeMs) {
        synchronized (exactSlot) {
            if (!exactSlot.isRemovableTerminalRecord(updatedBeforeMs)
                    || !requestSlots.remove(
                            exactSlot.requestId(), exactSlot)) {
                return false;
            }
            exactSlot.detachGeneration();
            return true;
        }
    }

    void expireInactiveRequest(RequestSlot exactSlot, long nowMs) {
        if (exactSlot != null) { exactSlot.expireInactiveRequest(nowMs); }
    }

    void processPrefillStatus(PrefillEndpoint source, RoleType role, PrefillState.WorkerStatusFact fact) {
        RequestSlot slot = requestSlot(fact.item().requestId());
        if (slot != null) { slot.processPrefillStatus(source, role, fact); }
    }

    void processDecodeStatus(DecodeEndpoint source, DecodeEndpoint.WorkerStatusFact fact) {
        RequestSlot slot = requestSlot(fact.reservation().requestId());
        if (slot != null) { slot.processDecodeStatus(source, fact); }
    }

    void confirmDecodeAcceptance(DecodeEndpoint source, DecodeEndpoint.ReservationHandle reservation) {
        processDecodeStatus(source, DecodeEndpoint.WorkerStatusFact.accepted(reservation));
    }

    void projectPrefillRetirementItem(PrefillEndpoint source, ScheduledRequest exact) {
        if (exact == null || exact.prefillEp() != source) { return; }
        RequestSlot slot = requestSlot(exact.requestId());
        if (slot != null) { slot.recordPrefillRetirement(source, exact); }
    }

    void projectDecodeRetirementReservation(DecodeEndpoint source, DecodeEndpoint.ReservationHandle exact) {
        RequestSlot slot = requestSlot(exact.requestId());
        if (slot != null) { slot.recordDecodeRetirement(source, exact); }
    }

    CompletableFuture<Response> register(
            BalanceContext context) {
        if (context == null || context.getRequest() == null) {
            return CompletableFuture.completedFuture(buildErrorResponse(
                    StrategyErrorType.INVALID_REQUEST, null));
        }
        if (shuttingDown.get()) {
            return CompletableFuture.completedFuture(buildErrorResponse(
                    StrategyErrorType.DISPATCH_FAILED,
                    "request scheduler is shutting down"));
        }

        RequestSlot slot = null;
        boolean registered = false;
        try {
            // Never execute endpoint cleanup, timer operations or public callbacks here.
            // Slot reducers only remove from the concurrent index, never acquire this lock.
            synchronized (registrationLock) {
                if (requestSlots.containsKey(context.getRequestId())) {
                    return CompletableFuture.completedFuture(buildErrorResponse(
                            StrategyErrorType.INVALID_REQUEST,
                            "duplicate request_id: " + context.getRequestId()));
                }
                if (context.requestExpired(System.currentTimeMillis())) {
                    Response failure = buildErrorResponse(
                            context.getConfig().isQueue() ? StrategyErrorType.RESOURCE_EXHAUSTED : StrategyErrorType.BATCH_SLO_EXPIRED,
                            "request scheduling deadline has expired before placement");
                    if (globalQueue != null) {
                        context.setSchedulingDiagnostics(globalQueue.waitDiagnostics());
                    }
                    return CompletableFuture.completedFuture(failure);
                }
                if (shuttingDown.get()) {
                    return CompletableFuture.completedFuture(buildErrorResponse(
                            StrategyErrorType.DISPATCH_FAILED,
                            "request scheduler is shutting down"));
                }
                slot = new RequestSlot(completionPublisher, context, expirationTimer, terminalCleanup, this::exitAdmissionHandleGate, context.getConfig().isQueue(), globalQueue);
                context.setEnqueueTime(System.currentTimeMillis());
                synchronized (slot) {
                    slot.configureInactivityTimeout(
                            context.getConfig().getRequestLifecycle().getRequest().getTimeoutMs());
                    requestSlots.put(context.getRequestId(), slot);
                    registered = true;
                }
            }
            RequestFuture future = slot.future();
            if (shuttingDown.get()) {
                completeError(
                        future,
                        StrategyErrorType.DISPATCH_FAILED,
                        "request scheduler is shutting down");
                return future;
            }
            if (context.requestExpired(System.currentTimeMillis())) {
                slot.cancelRequest(0L, CancelReason.DEADLINE_EXCEEDED);
                return future;
            }
            attachRequestExpiration(context, future);
            return future;
        } catch (Throwable failure) {
            Logger.error(
                    "Request registration failed for request id: {}",
                    context.getRequestId(),
                    failure);
            String detail = "Submit failed: " + failure.getMessage();
            if (registered) {
                completeError(
                        slot.future(),
                        StrategyErrorType.DISPATCH_FAILED,
                        detail);
                return slot.future();
            }
            return CompletableFuture.completedFuture(buildErrorResponse(
                    StrategyErrorType.DISPATCH_FAILED, detail));
        }
    }

    void attachRequestExpiration(
            BalanceContext context,
            CompletableFuture<Response> future) {
        if (shuttingDown.get()) {
            return;
        }
        RequestSlot slot = requestSlots.get(context.getRequestId());
        if (slot == null || !slot.ownsFuture(future)) {
            return;
        }
        if (context.getConfig().isQueue()) {
            expirationTimer.attachRequestDeadline(slot, context.getRequestExpiresAtMs());
        }
        expirationTimer.attachInactivityDeadline(slot);
    }

    boolean commitItemForPublication(ScheduledRequest item, BooleanSupplier publication) {
        return commitRoute(item, publication) == PlacementResult.Status.SUCCESS;
    }

    PlacementResult.Status commitRoute(ScheduledRequest item, BooleanSupplier publication) {
        Objects.requireNonNull(publication, "publication");
        if (shuttingDown.get() || item == null || item.future().isDone()) { return PlacementResult.Status.CLOSED; }
        RequestSlot slot = requestSlots.get(item.requestId());
        return slot == null || !slot.ownsFuture(item.future())
                ? PlacementResult.Status.CLOSED : slot.commitRoute(item, publication);
    }

    public boolean isAdmissionOpen(long requestId, CompletableFuture<?> future) {
        if (shuttingDown.get()) {
            return false;
        }
        RequestSlot slot = requestSlots.get(requestId);
        if (slot == null || !slot.ownsFuture(future)) {
            return false;
        }
        synchronized (slot) {
            return isCurrentSlot(slot) && slot.isOpen();
        }
    }

    public AdmissionHandle claimAdmissionHandle(
            long requestId, CompletableFuture<?> future) {
        if (!enterAdmissionHandleGate()) {
            return null;
        }
        boolean transferred = false;
        try {
            RequestSlot slot = requestSlots.get(requestId);
            if (slot == null || !slot.ownsFuture(future)) {
                return null;
            }
            AdmissionHandle handle;
            synchronized (slot) {
                handle = isCurrentSlot(slot)
                        ? slot.tryBeginAdmissionHandle()
                        : null;
            }
            transferred = handle != null;
            return handle;
        } finally {
            if (!transferred) {
                exitAdmissionHandleGate();
            }
        }
    }

    private boolean enterAdmissionHandleGate() {
        synchronized (admissionQuiescenceMonitor) {
            if (shuttingDown.get()) {
                return false;
            }
            if (inFlightAdmissionHandles == Integer.MAX_VALUE) {
                throw new IllegalStateException(
                        "admission handle counter overflow");
            }
            inFlightAdmissionHandles++;
            return true;
        }
    }

    void exitAdmissionHandleGate() {
        synchronized (admissionQuiescenceMonitor) {
            if (inFlightAdmissionHandles <= 0) {
                throw new IllegalStateException(
                        "admission handle counter underflow");
            }
            inFlightAdmissionHandles--;
            if (inFlightAdmissionHandles == 0) {
                admissionQuiescenceMonitor.notifyAll();
            }
        }
    }

    private void awaitAdmissionHandleQuiescence() {
        boolean interrupted = false;
        synchronized (admissionQuiescenceMonitor) {
            while (inFlightAdmissionHandles != 0) {
                try {
                    admissionQuiescenceMonitor.wait();
                } catch (InterruptedException interruption) {
                    interrupted = true;
                }
            }
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
        }
    }

    public void finishYielded(ScheduledRequest victim, String detail) {
        finishVictim(
                victim,
                StrategyErrorType.PRIORITY_PREEMPTED,
                detail);
    }

    void onQueuedItemPreempted(ScheduledRequest victim, ScheduledRequest incoming) {
        finishYielded(victim, "preempted by higher-priority request " + incoming.requestId());
        requestReporter.reportVictim(victim.priority(), incoming.priority(),
                "prefill_queued", "prefill_inflight_requests");
        requestReporter.reportPriorityPreempt("prefill_queued");
    }

    public void finishYieldedReservation(
            long requestId, long reservationToken, String detail) {
        if (reservationToken <= 0L) {
            throw new IllegalArgumentException("reservationToken must be positive");
        }
        RequestSlot entry = requestSlots.get(requestId);
        ScheduledRequest victim = null;
        if (entry != null) {
            synchronized (entry) {
                victim = entry.activeItemForReservation(reservationToken);
            }
        }
        if (victim != null) {
            finishYielded(victim, detail);
            return;
        }
        Logger.debug("finishYieldedReservation miss: request_id={} token={} detail={}",
                requestId, reservationToken, detail);
        try {
            requestReporter.reportInflightSettleMiss("yielded");
        } catch (RuntimeException metricFailure) {
            Logger.warn("Failed to report yielded settle miss: request_id={}",
                    requestId, metricFailure);
        }
    }

    private void finishVictim(ScheduledRequest item, StrategyErrorType error, String detail) {
        RequestSlot slot = entryFor(item);
        if (slot != null) { slot.recordSchedulingFailure(error, detail); }
    }

    public Optional<PreemptionRegistration> tryClaim(
            long requestId, long reservationToken, long attemptToken, String detail) {
        RequestSlot entry = requestSlots.get(requestId);
        if (entry == null) {
            return Optional.empty();
        }
        synchronized (entry) {
            return Optional.ofNullable(entry.tryInstallPreemption(
                    reservationToken, attemptToken, detail));
        }
    }

    public Optional<CancelTarget> findCancelTarget(
            long requestId, long reservationToken) {
        RequestSlot entry = requestSlots.get(requestId);
        if (entry == null) {
            return Optional.empty();
        }
        synchronized (entry) {
            CancelTarget target = cancelTarget(
                    entry.activeItemForReservation(reservationToken));
            return target == null || !target.isRoutable()
                    ? Optional.empty() : Optional.of(target);
        }
    }

    public RequestState cancelRequest(long requestId, long expectedBatchId, CancelReason reason) {
        Objects.requireNonNull(reason, "reason");
        RequestSlot slot = requestSlot(requestId);
        return slot == null ? null : slot.cancelRequest(expectedBatchId, reason);
    }

    private static CancelTarget cancelTarget(
            ScheduledRequest item) {
        ServerStatus prefill = item == null ? null : item.prefill();
        return prefill == null ? null
                : new CancelTarget(
                        prefill.getServerIp(), prefill.getGrpcPort());
    }

    public int liveRequestCount() {
        int live = 0;
        for (Map.Entry<Long, RequestSlot> candidate : requestSlots.entrySet()) {
            RequestSlot slot = candidate.getValue();
            synchronized (slot) {
                if (requestSlots.get(candidate.getKey()) == slot
                        && slot.isLiveGeneration()) {
                    live++;
                }
            }
        }
        return live;
    }

    public long oldestLiveSlotAgeMs() {
        long oldest = Long.MAX_VALUE;
        long now = System.currentTimeMillis();
        for (Map.Entry<Long, RequestSlot> candidate : requestSlots.entrySet()) {
            RequestSlot slot = candidate.getValue();
            synchronized (slot) {
                if (requestSlots.get(candidate.getKey()) == slot
                        && slot.isLiveGeneration()) {
                    oldest = Math.min(oldest, slot.createdAtMs());
                }
            }
        }
        return oldest == Long.MAX_VALUE ? 0L
                : Math.max(0L, now - oldest);
    }

    public List<RequestState> snapshotActiveRequests() {
        List<RequestState> snapshots = new ArrayList<>(requestSlots.size());
        for (Map.Entry<Long, RequestSlot> candidate : requestSlots.entrySet()) {
            RequestSlot entry = candidate.getValue();
            synchronized (entry) {
                if (requestSlots.get(candidate.getKey()) == entry
                        && entry.isLiveGeneration()) {
                    snapshots.add(entry.snapshot());
                }
            }
        }
        snapshots.sort((left, right) -> {
            int createdOrder = Long.compare(left.createdAtMs(), right.createdAtMs());
            return createdOrder != 0
                    ? createdOrder : Long.compare(left.requestId(), right.requestId());
        });
        return List.copyOf(snapshots);
    }

    public RequestState getRequestState(long requestId,
                                        long expectedBatchId) {
        RequestSlot entry = requestSlots.get(requestId);
        if (entry == null) {
            return null;
        }
        synchronized (entry) {
            RequestState snapshot = entry.snapshot();
            return snapshot != null && snapshot.matchesBatch(expectedBatchId) ? snapshot : null;
        }
    }

    /**
     * Retain registered IDs, including terminal records, during endpoint orphan cleanup.
     * Called under endpoint locks: never acquire a Slot lock here. Read the current
     * directory rather than a snapshot, which could miss a newly registered request.
     */
    boolean retainForSchedulerCleanup(long requestId) {
        return requestSlots.containsKey(requestId);
    }

    public void onQueuedItemExpired(ScheduledRequest exact) {
        RequestSlot slot = entryFor(exact);
        if (slot != null) { slot.cancelRequest(0L, CancelReason.DEADLINE_EXCEEDED); }
    }

    public void onQueueOfferFailure(ScheduledRequest exact, Throwable error) {
        RequestSlot slot = entryFor(exact);
        if (slot != null) { slot.recordSchedulingFailure(StrategyErrorType.DISPATCH_FAILED,
                "Worker scheduling queue rejected request: "
                        + (error == null ? "endpoint publication failed" : error.getMessage())); }
    }

    public CapacityBoundary.Attempt<BatchDeliveryStrategy.BatchTransaction> prepareBatchDelivery(
            ScheduledRequest item, BatchDeliveryStrategy strategy) {
        RequestSlot slot = entryFor(item);
        return slot == null ? CapacityBoundary.Attempt.rejected(CapacityBoundary.OWNERSHIP_LOST)
                : slot.prepareDispatch(item, () -> strategy.prepareAdmission(item));
    }

    public CapacityBoundary.Attempt<ScheduledRequest> prepareBatchMember(
            ScheduledRequest item, BatchDeliveryStrategy.BatchTransaction transaction) {
        RequestSlot slot = entryFor(item);
        return slot == null ? CapacityBoundary.Attempt.rejected(CapacityBoundary.OWNERSHIP_LOST)
                : slot.prepareDispatch(item, () -> transaction.append(item));
    }

    public CapacityBoundary.Attempt<ScheduledRequest> prepareRouteMember(
            ScheduledRequest item, RouteDeliveryStrategy.RouteTransaction transaction,
            PrefillTimePredictor.Evaluator evaluator) {
        RequestSlot slot = entryFor(item);
        return slot == null ? CapacityBoundary.Attempt.rejected(CapacityBoundary.OWNERSHIP_LOST)
                : slot.prepareDispatch(item, () -> transaction.append(item, evaluator));
    }

    public DeliveryClaim claimBatchDelivery(ScheduledRequest item, BatchDeliveryStrategy.BatchTransaction transaction) {
        RequestSlot slot = entryFor(item);
        return slot == null ? null : slot.claimDelivery(item, DeliveryClaimKind.BATCH_ENQUEUE,
                transaction.batchId(), () -> transaction.transferToEndpoint(item));
    }

    public DeliveryClaim claimRouteDelivery(ScheduledRequest item, PrefillAdmissionResources.CommittedAdmissionOwner admission) {
        RequestSlot slot = entryFor(item);
        return slot == null ? null : slot.claimDelivery(item, DeliveryClaimKind.ROUTE_DECISION,
                0L, () -> admission.transferToEndpoint(item));
    }

    public void setDeliveryPrediction(DeliveryClaim claim, WorkSnapshot precedingWork, long predictedMs) {
        claim.slot.setDeliveryPrediction(claim, precedingWork, predictedMs);
    }

    public void publishRoute(DeliveryClaim claim, WorkSnapshot precedingWork, long predictedMs) {
        claim.slot.publishRoute(claim, precedingWork, predictedMs);
    }

    public void failDeliveryPreparation(ScheduledRequest exact, Throwable cause) {
        RequestSlot slot = entryFor(exact);
        if (slot != null) { slot.failDeliveryPreparation(exact, cause); }
    }

    private RequestSlot entryFor(ScheduledRequest item) {
        RequestSlot entry = requestSlots.get(item.requestId());
        if (entry == null) {
            return null;
        }
        synchronized (entry) {
            return requestSlots.get(item.requestId()) == entry
                    && entry.ownsActiveItem(item) ? entry : null;
        }
    }

    private static void completeError(CompletableFuture<Response> future,
                                      StrategyErrorType errorType,
                                      String message) {
        if (future.isDone()) {
            return;
        }
        future.complete(buildErrorResponse(errorType, message));
    }

    boolean publishDecisionResponseAsync(long requestId, CompletableFuture<Response> future, Response response) {
        RequestSlot slot = requestSlots.get(requestId);
        return slot != null && slot.ownsFuture(future) && slot.terminateLocallyAndPublishResponse(response);
    }

    public boolean closeAdmissionAndAwaitMutations() {
        if (!shuttingDown.compareAndSet(false, true)) {
            return false;
        }
        awaitAdmissionHandleQuiescence();
        return true;
    }

    public void closeOutstandingAndTerminalize() {
        if (!shuttingDown.get()) {
            throw new IllegalStateException(
                    "admission must close before terminal shutdown");
        }
        completeOutstandingRequestsForShutdown();
    }

    public void maintainExpiration(
            BiConsumer<Long, LongPredicate> exactSweeper) {
        expirationTimer.maintain(exactSweeper);
    }

    public void closeExpiration() {
        expirationTimer.close();
    }

    public void closePublisher() {
        completionPublisher.close();
    }

    private void completeOutstandingRequestsForShutdown() {
        List<TerminalAction> actions = new ArrayList<>();
        for (RequestSlot slot : requestSlots.values()) {
            TerminalAction action = slot.claimShutdownAction();
            if (action != null) { actions.add(action); }
        }
        actions.forEach(terminalCleanup::submitTerminal);
    }

}
