package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.preemption.CancelTarget;
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
import java.util.function.Function;
import java.util.function.LongPredicate;
import java.util.function.Supplier;

import static org.flexlb.balance.scheduler.RequestResponses.buildErrorResponse;

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
    private int inFlightAdmissionMutations;

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
                expirationTimer, reporter, completionPublisherWorkers(configService));
        this.terminalCleanup = new RequestTerminalCleanup(expirationTimer, completionPublisher);
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

    public boolean removeExactTombstone(
            RequestSlot exactSlot, long updatedBeforeMs) {
        synchronized (exactSlot) {
            if (!exactSlot.isRemovableTombstone(updatedBeforeMs)
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

    void onPrefillFact(PrefillEndpoint source, RoleType role, PrefillState.WorkerStatusFact fact) {
        applyEngineFact(fact.item().requestId(),
                slot -> slot.observePrefillFact(source, role, fact, System.currentTimeMillis()));
    }

    void onDecodeFact(DecodeEndpoint source, DecodeEndpoint.WorkerStatusFact fact) {
        applyEngineFact(fact.reservation().requestId(),
                slot -> slot.observeDecodeFact(source, fact, System.currentTimeMillis()));
    }

    void onDecodeAccepted(DecodeEndpoint source, DecodeEndpoint.ReservationHandle reservation) {
        onDecodeFact(source, DecodeEndpoint.WorkerStatusFact.accepted(reservation));
    }

    private void applyEngineFact(long requestId, Function<RequestSlot, RequestSlot.EngineObservation> observation) {
        RequestSlot slot = requestSlot(requestId);
        if (slot != null) { slot.observeEngineFact(observation); }
    }

    void projectPrefillRetirementItem(PrefillEndpoint source, ScheduledRequest exact) {
        if (exact == null || exact.prefillEp() != source) { return; }
        RequestSlot slot = entryFor(exact);
        if (slot != null) { slot.onPrefillRetired(source, exact); }
    }

    void projectDecodeRetirementReservation(DecodeEndpoint source, DecodeEndpoint.ReservationHandle exact) {
        RequestSlot slot = requestSlot(exact.requestId());
        if (slot != null) { slot.onDecodeRetired(source, exact); }
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
                    return CompletableFuture.completedFuture(buildErrorResponse(
                            StrategyErrorType.BATCH_SLO_EXPIRED,
                            "request scheduling deadline has expired"));
                }
                if (shuttingDown.get()) {
                    return CompletableFuture.completedFuture(buildErrorResponse(
                            StrategyErrorType.DISPATCH_FAILED,
                            "request scheduler is shutting down"));
                }
                slot = new RequestSlot(completionPublisher, context.getRequestId(), expirationTimer, terminalCleanup, this::exitAdmissionMutationGate);
                context.setEnqueueTime(System.currentTimeMillis());
                synchronized (slot) {
                    slot.configureDeadlineError(StrategyErrorType.BATCH_SLO_EXPIRED);
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
                completeError(
                        future,
                        StrategyErrorType.BATCH_SLO_EXPIRED,
                        "request scheduling deadline has expired");
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

    public AdmissionMutation claimAdmissionMutation(
            long requestId, CompletableFuture<?> future) {
        if (!enterAdmissionMutationGate()) {
            return null;
        }
        boolean transferred = false;
        try {
            RequestSlot slot = requestSlots.get(requestId);
            if (slot == null || !slot.ownsFuture(future)) {
                return null;
            }
            AdmissionMutation mutation;
            synchronized (slot) {
                mutation = isCurrentSlot(slot)
                        ? slot.tryBeginAdmissionMutation(
                                (exact, failure) -> slot.onAdmissionFailed(exact, failure),
                                exact -> slot.onAdmissionCompleted(exact))
                        : null;
            }
            transferred = mutation != null;
            return mutation;
        } finally {
            if (!transferred) {
                exitAdmissionMutationGate();
            }
        }
    }

    private boolean enterAdmissionMutationGate() {
        synchronized (admissionQuiescenceMonitor) {
            if (shuttingDown.get()) {
                return false;
            }
            if (inFlightAdmissionMutations == Integer.MAX_VALUE) {
                throw new IllegalStateException(
                        "admission mutation counter overflow");
            }
            inFlightAdmissionMutations++;
            return true;
        }
    }

    void exitAdmissionMutationGate() {
        synchronized (admissionQuiescenceMonitor) {
            if (inFlightAdmissionMutations <= 0) {
                throw new IllegalStateException(
                        "admission mutation counter underflow");
            }
            inFlightAdmissionMutations--;
            if (inFlightAdmissionMutations == 0) {
                admissionQuiescenceMonitor.notifyAll();
            }
        }
    }

    private void awaitAdmissionMutationQuiescence() {
        boolean interrupted = false;
        synchronized (admissionQuiescenceMonitor) {
            while (inFlightAdmissionMutations != 0) {
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
                StrategyErrorType.NO_AVAILABLE_WORKER,
                detail);
    }

    void onQueuedItemPreempted(ScheduledRequest victim, ScheduledRequest incoming) {
        finishYielded(victim, "yielded to higher-priority request " + incoming.requestId());
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
        if (slot != null) { slot.onFailure(error, detail); }
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
            return batchMatches(snapshot, expectedBatchId) ? snapshot : null;
        }
    }

    public boolean ownsRequestGeneration(long requestId) {
        RequestSlot slot = requestSlots.get(requestId);
        if (slot == null) {
            return false;
        }
        synchronized (slot) {
            return isCurrentSlot(slot) && slot.isLiveGeneration();
        }
    }

    public void onQueuedItemExpired(ScheduledRequest exact) {
        RequestSlot slot = entryFor(exact);
        if (slot != null) { slot.cancelRequest(0L, CancelReason.DEADLINE_EXCEEDED); }
    }

    public void onQueueOfferFailure(ScheduledRequest exact, Throwable error) {
        RequestSlot slot = entryFor(exact);
        if (slot != null) { slot.onFailure(StrategyErrorType.DISPATCH_FAILED,
                "Worker scheduling queue rejected request: "
                        + (error == null ? "endpoint publication failed" : error.getMessage())); }
    }

    public void onPreparedDeliveryFailure(
            ScheduledRequest exactItem,
            Throwable error) {
        failPrepared(exactItem, error);
    }

    public <T> Optional<T> prepareIfOwned(ScheduledRequest exact, Supplier<T> preparation) {
        RequestSlot slot = entryFor(exact);
        return slot == null ? Optional.empty() : slot.prepareIfOwned(exact, preparation);
    }

    public DeliveryClaim tryClaimRouteDelivery(
            ScheduledRequest exactItem,
            BooleanSupplier endpointHandoff) {
        return tryClaimForDelivery(
                exactItem, DeliveryClaimKind.ROUTE_DECISION, 0L,
                endpointHandoff);
    }

    public DeliveryClaim tryClaimBatchDelivery(
            ScheduledRequest exactItem,
            long batchId,
            BooleanSupplier endpointHandoff) {
        if (batchId <= 0L) {
            throw new IllegalArgumentException("batchId must be positive");
        }
        return tryClaimForDelivery(
                exactItem, DeliveryClaimKind.BATCH_ENQUEUE, batchId,
                endpointHandoff);
    }

    private DeliveryClaim tryClaimForDelivery(ScheduledRequest item, DeliveryClaimKind kind,
            long correlationId, BooleanSupplier handoff) {
        RequestSlot slot = entryFor(item);
        return slot == null ? null : slot.claimDelivery(item, kind, correlationId, handoff);
    }

    public void failPrepared(ScheduledRequest exact, Throwable cause) {
        RequestSlot slot = entryFor(exact);
        if (slot != null) { slot.onPreparationFailed(exact, cause); }
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
        return slot != null && slot.ownsFuture(future) && slot.publishDecisionResponse(response);
    }

    private static boolean batchMatches(RequestState snapshot,
                                        long expectedBatchId) {
        if (snapshot == null) {
            return false;
        }
        return expectedBatchId == 0 || snapshot.batchId() == expectedBatchId;
    }

    public boolean closeAdmissionAndAwaitMutations() {
        if (!shuttingDown.compareAndSet(false, true)) {
            return false;
        }
        awaitAdmissionMutationQuiescence();
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
            TerminalAction action = slot.onShutdown();
            if (action != null) { actions.add(action); }
        }
        actions.forEach(terminalCleanup::submitTerminal);
    }

}
