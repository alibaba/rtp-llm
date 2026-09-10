package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.balance.preemption.VictimTerminal;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.debug.DebugPage;
import org.flexlb.debug.DebugQuery;
import org.flexlb.debug.DebugRows;
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

/**
 * Canonical lifecycle and ownership registry for FlexLB requests.
 *
 * <p>Responsibilities:
 * <ul>
 *   <li>One canonical {@link RequestSlot} per request generation</li>
 *   <li>Admission permits, absolute deadlines, and terminal settlement</li>
 *   <li>Delivery claims and exact reservation ownership</li>
 *   <li>Batch enqueue and route-decision acknowledgement handling</li>
 *   <li>Preemption, cancellation, and shutdown reduction</li>
 * </ul>
 *
 * <p>Queue ordering and route selection belong to
 * {@link GlobalQueueCoordinator}; transport batching belongs to a concrete
 * delivery strategy. This class exposes only exact {@link RequestSlot}
 * capabilities and never performs endpoint selection or queue traversal.
 */
@Component
public class RequestRegistry {

    private static final Runnable NO_POST_LOCK_ACTION = () -> { };

    private final RequestCompletionPublisher completionPublisher;
    /** Sole semantic deadline/retention owner for the canonical slot directory. */
    private final ExpirationTimer expirationTimer;
    /** One-way lifecycle gate; terminal completions remain allowed after it closes. */
    private final AtomicBoolean shuttingDown = new AtomicBoolean();
    /**
     * Shutdown barrier for admission mutations which have crossed their slot
     * ownership boundary but have not yet published or transferred the exact
     * completion. The monitor is never held while acquiring a RequestSlot.
     */
    private final Object admissionQuiescenceMonitor = new Object();
    private int inFlightAdmissionMutations;
    /** Serializes duplicate detection and canonical publication. */
    private final Object registrationLock = new Object();
    private final BatchSchedulerReporter reporter;
    private final RequestSchedulerReporter requestReporter;
    /** The sole canonical owner for admission, delivery and terminal state. */
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
                Objects.requireNonNull(configService, "configService"),
                reporter);
        this.completionPublisher = new RequestCompletionPublisher(
                this, completionPublisherWorkers(configService));
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

    public void cancelForDeadline(RequestSlot exactSlot) {
        cancelRequest(exactSlot, 0L, CancelReason.DEADLINE_EXCEEDED);
    }

    void expireInactiveRequest(RequestSlot exactSlot, long nowMs) {
        if (exactSlot == null) {
            return;
        }
        TerminalAction expiration;
        synchronized (exactSlot) {
            if (!isCurrentSlot(exactSlot) || !exactSlot.ownsActiveGeneration()
                    || exactSlot.snapshot().state().isTerminal() || !exactSlot.requestInactive(nowMs)) {
                return;
            }
            String detail = "REQUEST_INACTIVE: no matching Engine request status before inactivity timeout";
            if (exactSlot.deferInactivityExpiryDuringAdmission(detail)) {
                return;
            }
            exactSlot.markCancellationRequested(CancelReason.DEADLINE_EXCEEDED, detail);
            expiration = beginExpiredRequestLocked(exactSlot, detail);
        }
        submitTerminal(expiration);
    }

    private void reconcileDecodeAcceptance(ScheduledRequest item) {
        if (item.decodeEp() != null && item.decodeEp().isReservationAccepted(item.decodeReservation())) {
            onDecodeAccepted(item.decodeEp(), item.decodeReservation());
        }
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

    private void applyEngineFact(long requestId,
                                java.util.function.Function<RequestSlot, RequestSlot.EngineObservation> reduction) {
        RequestSlot slot = requestSlot(requestId);
        if (slot == null) { return; }
        RequestSlot.EngineObservation effects;
        Runnable work;
        synchronized (slot) {
            if (!isCurrentSlot(slot)) { return; }
            effects = reduction.apply(slot);
            work = materializePostLockActionLocked(slot, effects.transition(), null);
        }
        try {
            cancelDecisionDeadline(effects.obsoleteDeadline());
        } finally {
            try { armDecisionDeadline(slot); }
            finally { runPostLock(work); }
        }
    }

    public void decisionExpired(RequestSlot.DecisionExpiry expiry) {
        if (expiry.needsConfirmation() && expiry.item() != null) {
            RequestSlot slot = entryFor(expiry.item());
            if (slot != null) {
                synchronized (slot) {
                    slot.markAwaitingConfirmation(
                            "decision lifetime expired; awaiting Engine confirmation");
                }
            }
        }
    }

    void projectPrefillRetirementItem(
            PrefillEndpoint retiredEndpoint,
            ScheduledRequest exactItem) {
        if (exactItem == null || exactItem.prefillEp() != retiredEndpoint) { return; }
        ScheduledRequest item = exactItem;
        RequestSlot slot = requestSlot(item.requestId());
        if (slot == null) {
            return;
        }
        String detail = "Prefill endpoint generation retired: "
                + retiredEndpoint.ipPort() + "#"
                + retiredEndpoint.getStatus().getGenerationId();
        TerminalAction action;
        synchronized (slot) {
            if (!isCurrentSlot(slot)) {
                return;
            }
            action = slot.beginPrefillRetirementTerminal(
                    retiredEndpoint,
                    item,
                    owner -> owner.fail(detail),
                    RequestRegistry.buildErrorResponse(
                            StrategyErrorType.DISPATCH_FAILED, detail));
        }
        submitTerminal(action);
    }

    void projectDecodeRetirementReservation(
            DecodeEndpoint retiredEndpoint,
            DecodeEndpoint.ReservationHandle reservation) {
        RequestSlot slot = requestSlot(reservation.requestId());
        if (slot == null) {
            return;
        }
        String detail = "Decode endpoint generation retired: generation="
                + reservation.endpointGenerationId();
        Runnable work;
        synchronized (slot) {
            if (!isCurrentSlot(slot)) {
                return;
            }
            work = materializePostLockActionLocked(
                    slot,
                    slot.reduceDecodeGenerationRetired(
                            retiredEndpoint, reservation, detail),
                    null);
        }
        runPostLock(work);
    }

    // ==================== Request submission ====================

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
                slot = new RequestSlot(completionPublisher, context.getRequestId());
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

    /**
     * Schedule request expiration as a reducer event. Attaching a timeout
     * directly to the public future would let the timer
     * permanently complete the frontend future while a priority Cancel owns
     * the request; a later authoritative CANCELED observation could then no
     * longer publish PRIORITY_PREEMPTED. FIFO and PRIORITY both arm this same
     * absolute-expiration timer.
     */
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

    // ==================== Exact inflight commit protocol ====================

    /** Bind one exact item before publishing it into its endpoint runtime. */
    boolean commitItemForPublication(ScheduledRequest item, BooleanSupplier publication) {
        return commitRoute(item, publication) == PlacementResult.Status.SUCCESS;
    }

    PlacementResult.Status commitRoute(
            ScheduledRequest item,
            BooleanSupplier publication) {
        Objects.requireNonNull(publication, "publication");
        if (shuttingDown.get() || item == null || item.future().isDone()) {
            return PlacementResult.Status.CLOSED;
        }
        RequestSlot slot = requestSlots.get(item.requestId());
        if (slot == null || !slot.ownsFuture(item.future())) {
            return PlacementResult.Status.CLOSED;
        }
        synchronized (slot) {
            if (!isCurrentSlot(slot)
                    || !slot.tryBindItemForPublication(item)) {
                return PlacementResult.Status.CLOSED;
            }
        }

        try {
            // Endpoint publication may acquire its queue lock. It must never
            // run while the exact RequestSlot monitor is held.
            if (publication.getAsBoolean()) {
                return PlacementResult.Status.SUCCESS;
            }
        } catch (RuntimeException | Error failure) {
            try {
                synchronized (slot) {
                    slot.rollbackItemPublication(item);
                }
            } catch (RuntimeException | Error resolutionFailure) {
                if (resolutionFailure != failure) {
                    failure.addSuppressed(resolutionFailure);
                }
            }
            throw failure;
        }

        synchronized (slot) {
            slot.rollbackItemPublication(item);
        }
        return PlacementResult.Status.BLOCKED;
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
                                (exact, failure) -> terminateAdmissionMutation(
                                        slot, exact, failure),
                                exact -> completeAdmissionMutation(slot, exact))
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

    /** Enter without retaining the monitor across RequestSlot ownership. */
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

    private void exitAdmissionMutationGate() {
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

    /** Wait uninterruptibly, then restore the shutdown thread's interrupt. */
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

    void completeAdmissionMutation(
            RequestSlot slot,
            AdmissionMutation exact) {
        try {
            RequestSlot.AdmissionMutationCompletion completion;
            synchronized (slot) {
                completion = slot.completeAdmissionMutation(exact);
            }
            if (!completion.owned()) {
                return;
            }
            if (completion.pendingTerminal() != null) {
                Runnable work;
                synchronized (slot) {
                    work = reduceDeferredTerminalFactLocked(
                            slot, completion.pendingTerminal());
                }
                runPostLock(work);
            } else if (completion.pendingRetirement() != null) {
                submitTerminal(completion.pendingRetirement());
            }
            // A retained delivery failure may have become stale after Engine acceptance.
            // Resume the already-selected cancellation even when that replay is a no-op.
            if (completion.cancellationToResume() != null) {
                resumeCancellationAfterAdmission(
                        slot, completion.cancellationToResume(), completion.inactivityExpired());
            }
        } finally {
            try {
                expirationTimer.attachInactivityDeadline(slot);
            } finally {
                exitAdmissionMutationGate();
            }
        }
    }

    void terminateAdmissionMutation(
            RequestSlot slot,
            AdmissionMutation exact,
            Response failure) {
        try {
            if (failure.isSuccess()) {
                throw new IllegalArgumentException(
                        "admission termination requires a failure response");
            }
            RequestSlot.AdmissionMutationCompletion completion;
            Runnable retainedWork = null;
            TerminalAction action = null;
            synchronized (slot) {
                completion = slot.claimAdmissionMutationTermination(exact);
                if (completion.pendingTerminal() != null) {
                    retainedWork = reduceDeferredTerminalFactLocked(
                            slot, completion.pendingTerminal());
                } else if (completion.pendingRetirement() == null) {
                    CancelReason pendingCancel =
                            completion.cancellationToResume();
                    Response terminalResponse = failure;
                    Function<RequestSlot, RequestState> transition;
                    if (pendingCancel == null) {
                        String detail = failure.getErrorMessage() == null
                                ? "eviction admission failed"
                                : failure.getErrorMessage();
                        transition = owner -> owner.fail(detail);
                    } else {
                        String detail = cancelDetail(pendingCancel);
                        terminalResponse = buildErrorResponse(
                                slot.cancellationErrorType(pendingCancel), detail);
                        transition = owner -> settleCancellationLifecycle(
                                owner, pendingCancel, detail);
                    }
                    action = beginTerminalLocked(
                            slot, false, false, transition, terminalResponse);
                    if (action == null) {
                        throw new IllegalStateException(
                                "failed to claim admission terminal for request "
                                        + slot.requestId());
                    }
                }
            }
            if (retainedWork != null) {
                runPostLock(retainedWork);
            } else if (completion.pendingRetirement() != null) {
                submitTerminal(completion.pendingRetirement());
            } else {
                submitTerminal(action);
            }
            if (completion.cancellationToResume() != null) {
                resumeCancellationAfterAdmission(
                        slot, completion.cancellationToResume(), completion.inactivityExpired());
            }
        } finally {
            try {
                expirationTimer.attachInactivityDeadline(slot);
            } finally {
                exitAdmissionMutationGate();
            }
        }
    }

    /**
     * Resume effects for a cancellation whose first cause was atomically
     * promoted while completing its admission mutation. Do not route this
     * through the public reducer: doing so would reopen first-cause election
     * after the mutation owner has already selected the canonical cause.
     */
    private void resumeCancellationAfterAdmission(
            RequestSlot entry,
            CancelReason reason,
            boolean inactivityExpired) {
        TerminalAction localCompletion = null;
        synchronized (entry) {
            if (!isCurrentSlot(entry) || !entry.ownsActiveGeneration()) {
                return;
            }
            CancelReason firstCause = entry.requireCancellationFirstCause();
            if (firstCause != reason) {
                throw new IllegalStateException(
                        "admission cancellation first cause changed for request " + entry.requestId());
            }
            String detail = inactivityExpired
                    ? "REQUEST_INACTIVE: no matching Engine request status before inactivity timeout"
                    : cancelDetail(firstCause);
            ScheduledRequest item = entry.activeItem();
            if (inactivityExpired || firstCause == CancelReason.DEADLINE_EXCEEDED
                    || entry.requestInactive(System.currentTimeMillis())) {
                localCompletion = beginExpiredRequestLocked(entry, detail);
            } else if (item == null || entry.canClaimLocalTerminal()) {
                localCompletion = beginTerminalLocked(
                        entry, item != null, item != null,
                        owner -> settleCancellationLifecycle(owner, firstCause, detail),
                        buildErrorResponse(entry.cancellationErrorType(firstCause), detail));
            }
        }
        submitTerminal(localCompletion);
    }

    /**
     * Terminate a yielded victim — one the engine never saw (prefill queue
     * eviction / decode reserved-only eviction, contract 5.3) — with the
     * retryable {@link StrategyErrorType#NO_AVAILABLE_WORKER}. Shares the
     * same exact-once release/tombstone chain as other local victim terminals.
     */
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

    /**
     * Shared victim terminal chain: rollback CAS, lifecycle fail, future
     * completion with the caller's terminal error type, tombstone. Each step
     * applies at most once regardless of repeats or terminal-path races.
     */
    private void finishVictim(ScheduledRequest victim, StrategyErrorType errorType, String detail) {
        RequestSlot entry = entryFor(victim);
        if (entry != null) {
            Runnable work;
            synchronized (entry) {
                work = reduceDeferredTerminalFactLocked(entry,
                        DeferredTerminal.failure(errorType, detail));
            }
            runPostLock(work);
        }
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

    public boolean tryApplyPreemptionPhase(
            PreemptionRegistration claim,
            PreemptionCancelPhase next) {
        if (next == null) {
            return false;
        }
        return tryReducePreemption(
                claim, false,
                entry -> entry.applyPreemptionPhase(claim, next));
    }

    public boolean tryReleasePreemption(PreemptionRegistration claim) {
        return tryReducePreemption(
                claim, false,
                entry -> entry.applyPreemptionRelease(claim));
    }

    public boolean trySettlePreemptionTombstone(
            PreemptionRegistration claim,
            String detail) {
        return tryReducePreemption(
                claim, true,
                entry -> entry.applyPreemptionTombstone(claim, detail));
    }

    private boolean tryReducePreemption(
            PreemptionRegistration claim,
            boolean cleanPrefillCounterpart,
            Function<RequestSlot, RequestSlot.PreemptionReduction> reducer) {
        if (claim == null) {
            return false;
        }
        RequestSlot entry = requestSlots.get(claim.requestId());
        if (entry == null) {
            return false;
        }
        Runnable work;
        synchronized (entry) {
            Runnable counterpartCleanup = cleanPrefillCounterpart
                    ? exactPrefillCounterpartCleanup(entry.activeItem())
                    : null;
            RequestSlot.PreemptionReduction reduction = reducer.apply(entry);
            work = materializePostLockActionLocked(
                    entry, reduction, counterpartCleanup);
        }
        runPostLock(work);
        return work != null;
    }

    Runnable materializePostLockActionLocked(
            RequestSlot entry,
            RequestSlot.PreemptionReduction reduction,
            Runnable priorityCounterpartCleanup) {
        if (!Thread.holdsLock(entry)) {
            throw new IllegalStateException(
                    "preemption reduction requires slot lock");
        }
        if (reduction.status()
                == RequestSlot.PreemptionReduction.Status.STALE) {
            return null;
        }
        if (reduction.status()
                == RequestSlot.PreemptionReduction.Status.NONE) {
            return NO_POST_LOCK_ACTION;
        }
        Runnable publication = materializeReplayLocked(
                entry, reduction.replay(), priorityCounterpartCleanup);
        if (publication == null) {
            throw new IllegalStateException(
                    "accepted preemption replay produced no publication for request "
                            + entry.requestId());
        }
        return replayPostLockAction(
                entry,
                publication,
                reduction.signal());
    }

    private Runnable materializeReplayLocked(
            RequestSlot entry,
            RequestSlot.PendingReplay replay,
            Runnable priorityCounterpartCleanup) {
        if (replay.terminal() != null) {
            DeferredTerminal exact = replay.terminal();
            if (exact.kind() == DeferredTerminal.Kind.PRIORITY) {
                return terminalPublication(
                        beginSettledPriorityTerminalLocked(
                                entry,
                                exact.detail(),
                                priorityCounterpartCleanup));
            }
            return applyOrdinaryTerminalLocked(entry, exact);
        }
        return deliveryPublication(
                entry,
                replay.item(),
                replay.confirmation(),
                replay.kind(),
                replay.batchId());
    }

    private Runnable replayPostLockAction(
            RequestSlot entry,
            Runnable publication,
            PreemptionRegistration terminalSignal) {
        return () -> {
            try {
                publication.run();
            } finally {
                if (terminalSignal != null) {
                    terminalSignal.signalTerminal(new VictimTerminal(entry.requestId()));
                }
            }
        };
    }

    private TerminalAction beginSettledPriorityTerminalLocked(
            RequestSlot entry,
            String detail,
            Runnable counterpartCleanup) {
        if (entry.hasCancellationFirstCause()) {
            CancelReason firstCause = entry.requireCancellationFirstCause();
            String cancellationDetail = cancelDetail(firstCause);
            return beginWorkerStatusTerminalLocked(
                    entry,
                    counterpartCleanup,
                    slot -> settleCancellationLifecycle(
                            slot, firstCause, cancellationDetail),
                    buildErrorResponse(
                            entry.cancellationErrorType(firstCause),
                            cancellationDetail));
        }
        return beginWorkerStatusTerminalLocked(
                entry,
                counterpartCleanup,
                slot -> slot.cancel(detail),
                buildErrorResponse(
                        StrategyErrorType.PRIORITY_PREEMPTED, detail));
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

    // ==================== External cancellation ====================

    /**
     * Record local cancellation for one exact request generation. Sent requests
     * remain tracked until Engine terminal evidence or their inactivity deadline.
     * Only the preemption coordinator may send cancellation to an Engine.
     */
    public RequestState cancelRequest(long requestId, long expectedBatchId, CancelReason reason) {
        return cancelRequest(requestSlots.get(requestId), expectedBatchId, reason);
    }

    private RequestState cancelRequest(RequestSlot entry, long expectedBatchId,
                                       CancelReason reason) {
        Objects.requireNonNull(reason, "reason");
        if (entry == null) { return null; }
        TerminalAction localCompletion = null;
        RequestState result;
        synchronized (entry) {
            if (!isCurrentSlot(entry)) {
                return matchingTerminalState(entry.requestId(), expectedBatchId);
            }
            RequestState current = entry.snapshot();
            if (!batchMatches(current, expectedBatchId)) {
                return null;
            }
            if (!entry.ownsActiveGeneration() || current.state().isTerminal()) {
                return current;
            }
            if (entry.hasCancellationFirstCause()
                    || (reason == CancelReason.DEADLINE_EXCEEDED
                        && current.deliveryClaimKind() != DeliveryClaimKind.NONE)) {
                return current;
            }
            String detail = cancelDetail(reason);
            if (entry.deferCancellationDuringAdmission(reason, detail)) {
                return entry.snapshot();
            }
            entry.markCancellationRequested(reason, detail);
            CancelReason firstCause = entry.requireCancellationFirstCause();
            ScheduledRequest item = entry.activeItem();
            if (item == null || entry.canClaimLocalTerminal()) {
                localCompletion = beginTerminalLocked(
                        entry, item != null, item != null,
                        owner -> settleCancellationLifecycle(owner, firstCause, detail),
                        buildErrorResponse(entry.cancellationErrorType(firstCause), detail));
            }
            result = entry.snapshot();
        }
        submitTerminal(localCompletion);
        return result;
    }

    /** Claim all exact local accounting at the end of this request's lease. */
    private TerminalAction beginExpiredRequestLocked(RequestSlot entry, String detail) {
        ScheduledRequest item = entry.activeItem();
        CancelReason firstCause = entry.requireCancellationFirstCause();
        return beginTerminalLocked(
                entry, true, false, false,
                item == null ? null : () -> expireEndpointAccounting(item),
                owner -> settleCancellationLifecycle(owner, firstCause, detail),
                buildErrorResponse(entry.cancellationErrorType(firstCause), detail));
    }

    private void expireEndpointAccounting(ScheduledRequest item) {
        Throwable failure = null;
        if (item.decodeEp() != null && item.decodeReservation() != null) {
            failure = runTerminalLeaf(failure,
                    () -> item.decodeEp().expireReservationExact(item.decodeReservation()));
        }
        if (item.prefillEp() != null) {
            failure = runTerminalLeaf(failure, () -> item.prefillEp().expireCommittedItem(item));
        }
        if (failure != null) {
            throw new IllegalStateException("request expiration cleanup failed: " + item.requestId(), failure);
        }
    }

    private RequestState matchingTerminalState(long requestId,
                                                            long expectedBatchId) {
        RequestSlot slot = requestSlots.get(requestId);
        RequestState terminal = slot == null
                ? null : slot.snapshot();
        return batchMatches(terminal, expectedBatchId) ? terminal : null;
    }

    private static CancelTarget cancelTarget(
            ScheduledRequest item) {
        ServerStatus prefill = item == null ? null : item.prefill();
        return prefill == null ? null
                : new CancelTarget(
                        prefill.getServerIp(), prefill.getGrpcPort());
    }

    /** Source endpoint accounting was already settled by its typed status fact. */
    private TerminalAction settleCancellationFromWorkerStatusLocked(
            RequestSlot entry,
            String proof,
            WorkerTerminalSource source) {
        return settleCancellationAfterEndpointSettlementLocked(
                entry,
                proof,
                workerStatusCounterpartCleanup(entry, source));
    }

    private TerminalAction settleCancellationAfterEndpointSettlementLocked(
            RequestSlot entry,
            String proof,
            Runnable counterpartCleanup) {
        CancelReason reason = entry.requireCancellationFirstCause();
        String detail = cancelDetail(reason) + "; " + proof;
        return beginWorkerStatusTerminalLocked(
                entry,
                counterpartCleanup,
                owner -> settleCancellationLifecycle(owner, reason, detail),
                buildErrorResponse(
                        entry.cancellationErrorType(reason), detail));
    }

    /** Called only after local rollback or authoritative engine settlement. */
    private static RequestState settleCancellationLifecycle(
            RequestSlot lifecycle,
            CancelReason reason,
            String detail) {
        return reason == CancelReason.DEADLINE_EXCEEDED
                ? lifecycle.timeout(detail)
                : lifecycle.cancel(detail);
    }

    private static String cancelDetail(CancelReason reason) {
        return reason == CancelReason.DEADLINE_EXCEEDED
                ? "request deadline exceeded"
                : "request cancelled by client";
    }

    /** Route a typed deferred terminal through the slot's opaque reducer. */
    private Runnable reduceDeferredTerminalFactLocked(
            RequestSlot entry,
            DeferredTerminal terminal) {
        ScheduledRequest item = entry.activeItem();
        if (item == null) {
            return null;
        }
        RequestSlot.PreemptionReduction reduction;
        if (terminal.kind() == DeferredTerminal.Kind.DELIVERY_REJECTED
                && item.decodeEp() != null
                && item.decodeReservation() != null) {
            reduction = entry.reduceDispatchRejected(
                    item.decodeEp(), item.decodeReservation(), item, terminal);
        } else {
            reduction = terminal.authoritativeWorker()
                    ? entry.reduceWorkerTerminal(
                            item, terminal)
                    : entry.reduceOrdinaryTerminal(
                            item, terminal);
        }
        return materializePostLockActionLocked(entry, reduction, null);
    }

    /** Apply an already-owned ordinary outcome. Called with {@code entry} locked. */
    private Runnable applyOrdinaryTerminalLocked(
            RequestSlot entry,
            DeferredTerminal terminal) {
        if (terminal.endpointAlreadyRetired()) {
            return applyDecodeSettledTerminalLocked(entry, terminal);
        }
        return switch (terminal.kind()) {
            case FAILURE -> applyFailureTerminalLocked(entry, terminal, true);
            case TIMEOUT -> applyTimeoutTerminalLocked(entry, terminal);
            case DELIVERY_FAILURE ->
                    applyFailureTerminalLocked(entry, terminal, true);
            case DELIVERY_REJECTED ->
                    applyDecodeSettledTerminalLocked(entry, terminal);
            case WORKER -> applyWorkerTerminalLocked(entry, terminal);
            case PRIORITY ->
                    throw new IllegalStateException(
                            "priority terminal requires its typed reducer");
            case DECODE_GENERATION_RETIRED ->
                    throw new IllegalStateException(
                            "retired Decode generation was not marked retired");
        };
    }

    private Runnable applyTimeoutTerminalLocked(
            RequestSlot entry,
            DeferredTerminal timeout) {
        return terminalPublication(beginTerminalLocked(
                entry,
                true,
                true,
                owner -> owner.timeout(timeout.detail()),
                buildErrorResponse(
                        entry.timeoutErrorType(), timeout.detail())));
    }

    private Runnable applyFailureTerminalLocked(
            RequestSlot entry,
            DeferredTerminal failure,
            boolean releaseDecode) {
        return terminalPublication(beginTerminalLocked(
                entry, true, releaseDecode,
                owner -> owner.fail(failure.detail()),
                buildErrorResponse(failure.errorType(), failure.detail())));
    }

    private Runnable applyDecodeSettledTerminalLocked(
            RequestSlot entry,
            DeferredTerminal terminal) {
        String terminalDetail;
        if (terminal.kind() == DeferredTerminal.Kind.DELIVERY_REJECTED
                || terminal.kind()
                    == DeferredTerminal.Kind.DECODE_GENERATION_RETIRED) {
            terminalDetail = terminal.detail();
        } else {
            throw new IllegalArgumentException(
                    "Decode-settled reducer requires a Decode terminal");
        }
        String detail = terminalDetail == null
                ? "Decode endpoint generation retired"
                : terminalDetail;
        if (entry.hasCancellationFirstCause()) {
            CancelReason firstCause = entry.requireCancellationFirstCause();
            String cancellationDetail = cancelDetail(firstCause)
                    + "; " + detail;
            return terminalPublication(beginTerminalLocked(
                    entry,
                    false,
                    true,
                    owner -> settleCancellationLifecycle(
                            owner, firstCause, cancellationDetail),
                    buildErrorResponse(
                            entry.cancellationErrorType(firstCause),
                            cancellationDetail)));
        }
        return terminalPublication(beginTerminalLocked(
                entry,
                false,
                true,
                owner -> owner.fail(detail),
                buildErrorResponse(
                        StrategyErrorType.DISPATCH_FAILED, detail)));
    }

    /** Endpoint resources are already settled; only RequestSlot/response remain. */
    private Runnable applyWorkerTerminalLocked(
            RequestSlot entry,
            DeferredTerminal terminal) {
        if (entry.hasCancellationFirstCause()) {
            String proof = terminal.workerSource()
                    == WorkerTerminalSource.PREFILL_BACKED
                            ? "Prefill terminal observed after cancellation"
                            : "Decode terminal observed after cancellation";
            return terminalPublication(
                    settleCancellationFromWorkerStatusLocked(
                            entry, proof, terminal.workerSource()));
        }
        Function<RequestSlot, RequestState> transition;
        Response response;
        if (terminal.workerSuccessful()) {
            transition = owner -> owner.complete("decode completed");
            ScheduledRequest item = entry.activeItem();
            response = buildSuccessResponse(
                    item, entry.snapshot().deliveryClaimKind());
        } else {
            String detail = "worker error code "
                    + terminal.workerErrorCode();
            transition = owner -> owner.fail(detail);
            response = buildErrorResponse(
                    StrategyErrorType.WORKER_EXECUTION_FAILED, detail);
        }
        return terminalPublication(beginWorkerStatusTerminalLocked(
                entry,
                workerStatusCounterpartCleanup(
                        entry, terminal.workerSource()),
                transition,
                response));
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

    /** Age in milliseconds of the oldest live request, or zero when none remain. */
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

    /** Per-slot immutable sample. Map traversal is weakly consistent, not a global cut. */
    public DebugPage debugSnapshot(DebugQuery query) {
        DebugRows rows = new DebugRows(query);
        if (query.requestId() != null) {
            RequestSlot slot = requestSlots.get(query.requestId());
            if (slot != null && rows.visit()) {
                synchronized (slot) {
                    if (requestSlots.get(query.requestId()) != slot) {
                        return DebugPage.unavailable("per_entry", "generation_changed", System.currentTimeMillis());
                    }
                    rows.add(slot.debugSnapshot());
                }
            }
        } else {
            for (Map.Entry<Long, RequestSlot> candidate : requestSlots.entrySet()) {
                if (!rows.visit()) {
                    break;
                }
                RequestSlot slot = candidate.getValue();
                synchronized (slot) {
                    if (requestSlots.get(candidate.getKey()) == slot) {
                        rows.add(slot.debugSnapshot());
                    }
                }
            }
        }
        return rows.finish("per_entry", Map.of("scope", "retained_scheduler_slots",
                "order", "unspecified", "includes_tombstones", true));
    }

    /**
     * Weakly-consistent immutable view of all scheduler-owned live request
     * lifecycles. The requestSlots map is authoritative; no diagnostic-only
     * shadow queue is maintained.
     */
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

    /** Whether scheduler lifecycle still owns endpoint accounting for this id. */
    public boolean ownsRequestGeneration(long requestId) {
        RequestSlot slot = requestSlots.get(requestId);
        if (slot == null) {
            return false;
        }
        synchronized (slot) {
            return isCurrentSlot(slot) && slot.isLiveGeneration();
        }
    }

    // ==================== Queue lifecycle callbacks ====================

    public void onQueuedItemExpired(ScheduledRequest exactItem) {
        ScheduledRequest head = exactItem;
        if (entryFor(head) != null) {
            // The batcher and the request timer may observe the same absolute
            // expiration concurrently. Both must enter the cancellation
            // reducer so first-cause ownership and the existing external
            // timeout classification cannot depend on which thread wins.
            cancelRequest(
                    head.requestId(), 0L,
                    CancelReason.DEADLINE_EXCEEDED);
        }
    }

    public void onQueueOfferFailure(
            ScheduledRequest exactItem,
            Throwable error) {
        ScheduledRequest item = exactItem;
        String failureDetail = error == null ? "endpoint publication failed" : error.getMessage();
        RequestSlot entry = entryFor(item);
        if (entry != null) {
            Runnable work;
            synchronized (entry) {
                work = reduceDeferredTerminalFactLocked(entry,
                        DeferredTerminal.failure(
                                StrategyErrorType.DISPATCH_FAILED,
                                "Worker scheduling queue rejected request: "
                                        + failureDetail));
            }
            runPostLock(work);
        }
    }

    public void onPreparedDeliveryFailure(
            ScheduledRequest exactItem,
            Throwable error) {
        failPrepared(exactItem, error);
    }

    // ==================== Delivery pipeline ====================

    /** Caller holds the exact RequestSlot. */
    private boolean ownsPreparedDelivery(RequestSlot entry, ScheduledRequest item) {
        RequestState snapshot = entry.snapshot();
        return entry.ownsActiveItem(item)
                && entry.isOpen()
                && entry.canClaimDelivery()
                && snapshot.state() == RequestState.Phase.QUEUED
                && snapshot.deliveryClaimKind() == DeliveryClaimKind.NONE;
    }

    public <T> Optional<T> prepareIfOwned(
            ScheduledRequest exactItem,
            Supplier<T> preparation) {
        ScheduledRequest item = exactItem;
        RequestSlot entry = entryFor(item);
        if (entry == null) {
            return Optional.empty();
        }
        synchronized (entry) {
            if (!ownsPreparedDelivery(entry, item)) {
                return Optional.empty();
            }
            return Optional.of(preparation.get());
        }
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

    private DeliveryClaim tryClaimForDelivery(
            ScheduledRequest exactItem,
            DeliveryClaimKind kind,
            long correlationId,
            BooleanSupplier endpointHandoff) {
        ScheduledRequest item = exactItem;
        RequestSlot entry = entryFor(item);
        if (entry == null) {
            return null;
        }
        synchronized (entry) {
            if (!ownsPreparedDelivery(entry, item)) {
                return null;
            }
            DeliveryClaim claim = new DeliveryClaim(
                    this, entry, item, kind, correlationId);
            if (!endpointHandoff.getAsBoolean()) {
                throw new IllegalStateException(
                        "endpoint ownership was lost while the exact"
                                + " RequestSlot was locked request_id="
                                + item.requestId());
            }
            switch (kind) {
                case BATCH_ENQUEUE -> {
                    entry.startBatchEnqueue(correlationId);
                    entry.markBatchEnqueueStarted();
                }
                case ROUTE_DECISION ->
                        entry.startRouteDecisionDelivery();
                case NONE -> throw new IllegalArgumentException(
                        "delivery claim kind cannot be NONE");
            }
            return claim;
        }
    }

    /** Begin Engine-facing delivery with one exact prediction and acceptance observation. */
    public void beginDelivery(DeliveryClaim claim, WorkSnapshot precedingWork, long unstartedWorkMs) {
        DeliveryClaim exact = exactClaim(claim);
        if (exact == null) {
            throw new IllegalArgumentException("delivery claim was not created by this scheduler");
        }
        Objects.requireNonNull(precedingWork, "precedingWork");
        if (unstartedWorkMs < 0L) {
            throw new IllegalArgumentException("unstarted work must be non-negative");
        }
        reconcileDecodeAcceptance(exact.item);
        synchronized (exact.slot) {
            if (!ownsDeliveryClaim(exact)) { return; }
            exact.slot.startDecisionTracking(precedingWork, unstartedWorkMs, System.currentTimeMillis());
        }
        armDecisionDeadline(exact.slot);
    }

    /** Publish a route after its shared delivery lifecycle has begun. */
    public void beginRouteDelivery(DeliveryClaim claim, WorkSnapshot precedingWork, long unstartedWorkMs) {
        DeliveryClaim exact = exactClaim(claim);
        if (exact == null || exact.kind != DeliveryClaimKind.ROUTE_DECISION) {
            throw new IllegalArgumentException("route delivery requires an exact route claim");
        }
        beginDelivery(exact, precedingWork, unstartedWorkMs);
        complete(exact, DeliveryResult.delivered());
    }

    private DeliveryClaim exactClaim(DeliveryClaim claim) {
        return claim != null && claim.owner == this ? claim : null;
    }

    /** Caller holds {@code exact.slot}. */
    private boolean ownsDeliveryClaim(DeliveryClaim exact) {
        return exact.slot.ownsDeliveryClaim(
                exact.item, exact.kind, exact.correlationId);
    }

    /** Opaque proof of the exact slot point-of-no-return. */
    public static final class DeliveryClaim {
        private final RequestRegistry owner;
        private final RequestSlot slot;
        private final ScheduledRequest item;
        private final DeliveryClaimKind kind;
        private final long correlationId;
        private boolean completed;

        private DeliveryClaim(
                RequestRegistry owner,
                RequestSlot slot,
                ScheduledRequest item,
                DeliveryClaimKind kind,
                long correlationId) {
            this.owner = owner;
            this.slot = slot;
            this.item = item;
            this.kind = kind;
            this.correlationId = correlationId;
        }

        public ScheduledRequest item() {
            return item;
        }
    }

    // ==================== Delivery outcomes ====================

    /** Called with {@code entry} locked. */
    private Runnable confirmRouteDecisionLocked(
            RequestSlot entry,
            ScheduledRequest item) {
        if (!entry.ownsDeliveryClaim(
                item, DeliveryClaimKind.ROUTE_DECISION, 0L)) {
            return null;
        }
        return materializePostLockActionLocked(
                entry,
                entry.reduceDeliveryConfirmed(0L),
                null);
    }

    public void complete(
            DeliveryClaim claim,
            DeliveryResult completion) {
        DeliveryClaim exact = exactClaim(claim);
        if (exact == null) {
            throw new IllegalArgumentException(
                    "delivery claim was not created by this scheduler");
        }
        Runnable work = null;
        synchronized (exact.slot) {
            // WorkerStatus may settle this generation before the RPC callback arrives.
            // Its terminal proof wins; an old transport outcome cannot reopen ownership.
            if (!ownsDeliveryClaim(exact)) {
                return;
            }
            if (exact.completed) {
                throw new IllegalStateException(
                        "delivery claim is already completed: request_id="
                                + exact.item.requestId());
            }
            exact.completed = true;
            if (completion.status()
                    == DeliveryResult.Status.DELIVERED) {
                work = switch (exact.kind) {
                    case BATCH_ENQUEUE -> confirmBatchEnqueueLocked(
                            exact.slot, exact.item);
                    case ROUTE_DECISION -> confirmRouteDecisionLocked(
                            exact.slot, exact.item);
                    case NONE -> throw new IllegalStateException(
                            "delivery claim kind cannot be NONE");
                };
            } else if (completion.status()
                    == DeliveryResult.Status.FAILED) {
                String detail = "Delivery failed: "
                        + detailOf(completion.cause());
                if (exact.slot.decodeOwnsRequest()) {
                    work = materializePostLockActionLocked(
                            exact.slot,
                            exact.slot.reduceDeliveryConfirmed(
                                    exact.correlationId),
                            null);
                } else {
                    DecodeEndpoint decode = exact.item.decodeEp();
                    DecodeEndpoint.ReservationHandle reservation =
                            exact.item.decodeReservation();
                    DecodeEndpoint.DispatchRejectionSettlement settlement =
                            decode == null || reservation == null
                                    ? DecodeEndpoint.DispatchRejectionSettlement.RELEASED
                                    : decode.settleDefiniteDispatchRejection(
                                            reservation);
                    switch (settlement) {
                        case RELEASED -> work = reduceDeferredTerminalFactLocked(
                                exact.slot,
                                DeferredTerminal.deliveryRejected(detail));
                        case ENGINE_ACCEPTED -> work = materializePostLockActionLocked(
                                exact.slot,
                                exact.slot.reduceDeliveryConfirmed(
                                        exact.correlationId),
                                null);
                        case CONFLICT -> exact.slot.markAwaitingConfirmation(detail);
                        case STALE -> work = null;
                    }
                }
            } else if (exact.slot.decodeOwnsRequest()) {
                work = materializePostLockActionLocked(
                        exact.slot,
                        exact.slot.reduceDeliveryConfirmed(
                                exact.correlationId),
                        null);
            } else {
                String detail = switch (completion.status()) {
                    case TIMED_OUT ->
                            "Delivery timed out: "
                                    + detailOf(completion.cause());
                    case UNCERTAIN ->
                            "Delivery outcome uncertain: "
                                    + detailOf(completion.cause());
                    case DELIVERED ->
                            throw new IllegalStateException(
                                    "delivered outcome was already handled");
                    case FAILED ->
                            throw new IllegalStateException(
                                    "failed outcome was already handled");
                };
                exact.slot.markAwaitingConfirmation(detail);
            }
        }
        runPostLock(work);
    }

    /** Called with {@code entry} locked. */
    private Runnable confirmBatchEnqueueLocked(
            RequestSlot entry,
            ScheduledRequest item) {
        RequestState current = entry.snapshot();
        long batchId = current.batchId();
        if (!entry.ownsDeliveryClaim(
                item, DeliveryClaimKind.BATCH_ENQUEUE, batchId)) {
            Logger.debug("Ignoring EnqueueBatch ACK without a batch claim request_id={}",
                    item.requestId());
            return null;
        }
        item.ctx().setAckAtMs(System.currentTimeMillis());
        item.ctx().setAckAtNanos(System.nanoTime());
        return materializePostLockActionLocked(
                entry,
                entry.reduceDeliveryConfirmed(batchId),
                null);
    }

    /**
     * Confirm a delivery after its ownership decision is final.
     * The returned publication must be executed only after every scheduler lock
     * has been released: {@code CompletableFuture.complete} runs arbitrary user
     * continuations synchronously on the completing thread.
     */
    private Runnable deliveryPublication(
            RequestSlot entry,
            ScheduledRequest item,
            RequestSlot.DeliveryConfirmation confirmation,
            DeliveryClaimKind deliveryKind,
            long batchId) {
        Response response = buildSuccessResponse(
                item, deliveryKind);
        return () -> publishDelivery(
                entry, item, response, confirmation, deliveryKind);
    }

    /** Claim the canonical slot and move every local cleanup capability once. */
    static TerminalAction beginTerminalLocked(
            RequestSlot entry,
            boolean releaseDecode,
            boolean releasePrefill,
            Function<RequestSlot, RequestState> transition,
            Response response) {
        return beginTerminalLocked(
                entry, true, releaseDecode, releasePrefill, null,
                transition, response);
    }

    /** Endpoint status has already settled source queue/ledger ownership. */
    static TerminalAction beginWorkerStatusTerminalLocked(
            RequestSlot entry,
            Runnable counterpartCleanup,
            Function<RequestSlot, RequestState> transition,
            Response response) {
        return beginTerminalLocked(
                entry, false, false, false, counterpartCleanup,
                transition, response);
    }

    private static TerminalAction beginTerminalLocked(
            RequestSlot entry,
            boolean removePrefillQueue,
            boolean releaseDecode,
            boolean releasePrefill,
            Runnable counterpartCleanup,
            Function<RequestSlot, RequestState> transition,
            Response response) {
        return entry.beginTerminalizing(
                removePrefillQueue,
                releaseDecode,
                releasePrefill,
                counterpartCleanup,
                transition,
                response);
    }

    /** Run exact local leaves, then and only then publish a terminal tombstone. */
    private RequestSlot.PublicationPermit finishTerminal(
            TerminalAction action) {
        RequestSlot entry = action.slot();
        ScheduledRequest item = action.item();
        Throwable cleanupFailure = null;
        cleanupFailure = runTerminalLeaf(
                cleanupFailure,
                action.terminalResources() == null
                        ? null : () -> expirationTimer.release(
                                action.terminalResources()));
        cleanupFailure = runTerminalLeaf(
                cleanupFailure,
                !action.removePrefillQueue()
                                || item == null || item.prefillEp() == null
                        ? null : () -> item.prefillEp().removeQueued(
                                item, action.queueReason()));
        cleanupFailure = runTerminalLeaf(
                cleanupFailure,
                action.releaseDecode() && item != null
                        ? () -> rollback(item) : null);
        cleanupFailure = runTerminalLeaf(
                cleanupFailure,
                action.releasePrefill() && item != null
                        ? () -> releasePrefillAccounting(item) : null);
        cleanupFailure = runTerminalLeaf(cleanupFailure, action.counterpartCleanup());
        cleanupFailure = runTerminalLeaf(
                cleanupFailure,
                action.preemption() == null
                        ? null : () -> action.preemption().signalTerminal(new VictimTerminal(entry.requestId())));

        TombstoneResult tombstone;
        synchronized (entry) {
            tombstone = entry.finishTombstone(action);
        }
        Throwable terminalFailure = tombstone.transitionFailure() == null
                ? cleanupFailure
                : appendFailure(cleanupFailure, tombstone.transitionFailure());
        if (terminalFailure != null) {
            Logger.error("Terminal cleanup isolated after canonical claim: request_id={}",
                    entry.requestId(), terminalFailure);
        }
        return tombstone.terminal() == null
                ? null : tombstone.publication();
    }

    private static Throwable runTerminalLeaf(Throwable first, Runnable leaf) {
        if (leaf == null) {
            return first;
        }
        try {
            leaf.run();
            return first;
        } catch (Throwable failure) {
            return appendFailure(first, failure);
        }
    }

    private Runnable terminalPublication(TerminalAction action) {
        return action == null ? null : () -> submitTerminal(action);
    }

    void runPostLock(Runnable action) {
        if (action == null) {
            return;
        }
        action.run();
    }

    void submitTerminal(TerminalAction action) {
        if (action == null) {
            return;
        }
        RequestSlot.PublicationPermit permit = finishTerminal(action);
        if (permit != null && action.response() != null) {
            completionPublisher.submitTerminalResponse(
                    permit, action.response());
        }
    }

    private void publishDelivery(
            RequestSlot slot,
            ScheduledRequest item,
            Response response,
            RequestSlot.DeliveryConfirmation confirmation,
            DeliveryClaimKind deliveryKind) {
        Throwable preparationFailure = null;
        preparationFailure = runTerminalLeaf(
                preparationFailure,
                confirmation.requestDeadline() == null
                        ? null : () -> expirationTimer.cancel(
                                confirmation.requestDeadline()));
        if (deliveryKind == DeliveryClaimKind.BATCH_ENQUEUE
                && item.ctx().getAckAtMs() > 0L && confirmation.batchEnqueueStartedAtMs() > 0L) {
            long latencyMs = Math.max(
                    0L,
                    item.ctx().getAckAtMs() - confirmation.batchEnqueueStartedAtMs());
            preparationFailure = runTerminalLeaf(
                    preparationFailure,
                    () -> reporter.reportDispatchAckTimeMs(
                            RoleType.PREFILL.name(),
                            item.prefillEp() == null
                                    ? ""
                                    : item.prefillEp().getIp(),
                            latencyMs));
        }
        if (preparationFailure != null) {
            Logger.error(
                    "Delivery publication preparation isolated: request_id={}",
                    item.requestId(),
                    preparationFailure);
        }
        completionPublisher.submitDeliveryResponse(
                confirmation.publication(), response);
    }

    private void armDecisionDeadline(RequestSlot entry) {
        java.util.OptionalLong deadline;
        synchronized (entry) {
            deadline = entry.decisionDeadlineAtMs();
        }
        if (deadline.isPresent()) {
            expirationTimer.registerDecisionDeadline(
                    entry, deadline.getAsLong());
        }
    }

    void cancelDecisionDeadline(
            ExpirationTimer.DecisionDeadline cleanup) {
        if (cleanup == null) {
            return;
        }
        try {
            expirationTimer.release(cleanup);
        } catch (Throwable failure) {
            Logger.error("Admission cleanup isolated", failure);
        }
    }

    private Response buildSuccessResponse(
            ScheduledRequest item,
            DeliveryClaimKind deliveryKind) {
        Response success = copyResponse(item.routeResponse());
        success.setSuccess(true);
        success.setCode(200);
        success.setEnqueuedByMaster(
                deliveryKind == DeliveryClaimKind.BATCH_ENQUEUE);
        return success;
    }

    public void failPrepared(ScheduledRequest exactItem, Throwable cause) {
        ScheduledRequest item = exactItem;
        RequestSlot entry = entryFor(item);
        if (entry == null) {
            return;
        }
        Runnable work = null;
        try {
            synchronized (entry) {
                if (!ownsPreparedDelivery(entry, item)) {
                    return;
                }
                work = reduceDeferredTerminalFactLocked(
                        entry,
                        DeferredTerminal.deliveryFailure(
                                StrategyErrorType.DISPATCH_FAILED,
                                "Delivery preparation failed: "
                                        + detailOf(cause)));
            }
            runPostLock(work);
        } catch (Throwable reductionFailure) {
            if (cause != null && cause != reductionFailure) {
                reductionFailure.addSuppressed(cause);
            }
            Logger.error(
                    "Prepared delivery failure reduction failed request_id={}",
                    item.requestId(), reductionFailure);
        }
    }

    private static String detailOf(Throwable cause) {
        if (cause == null) {
            return "unknown delivery failure";
        }
        String message = cause.getMessage();
        return message == null || message.isBlank()
                ? cause.getClass().getSimpleName() : message;
    }

    // ==================== Internal: resource rollback ====================

    /** Rollback using endpoint references already held by the item (no registry lookup). */
    private void rollback(ScheduledRequest item) {
        DecodeEndpoint decodeEp = item.decodeEp();
        DecodeEndpoint.ReservationHandle reservation =
                item.decodeReservation();
        if (decodeEp != null && reservation != null) {
            decodeEp.releaseReservationExact(reservation);
        }
    }

    /**
     * Exact opposite-role cleanup for a source endpoint which has already
     * settled itself. Both endpoint methods are total conditional operations;
     * Engine/protocol ownership remains with its canonical owner.
     */
    private static Runnable workerStatusCounterpartCleanup(
            RequestSlot entry,
            WorkerTerminalSource source) {
        ScheduledRequest item = entry.activeItem();
        if (item == null) {
            return null;
        }
        if (source == WorkerTerminalSource.PREFILL_BACKED) {
            DecodeEndpoint decode = item.decodeEp();
            DecodeEndpoint.ReservationHandle reservation =
                    item.decodeReservation();
            return decode == null || reservation == null
                    ? null
                    : () -> decode.releaseLocalShadowIfExact(reservation);
        }
        if (entry.snapshot().deliveryClaimKind()
                != DeliveryClaimKind.ROUTE_DECISION) {
            return null;
        }
        return exactPrefillCounterpartCleanup(item);
    }

    private static Runnable exactPrefillCounterpartCleanup(ScheduledRequest item) {
        PrefillEndpoint prefill = item.prefillEp();
        return prefill == null
                ? null : () -> prefill.releaseCommittedItem(item);
    }

    // ==================== Internal: requestSlots queries ====================

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

    /**
     * Settle the canonical Prefill owner by exact committed item identity.
     *
     * <p>The Registry already knows whether this request is an individual or
     * a batch member. Consulting RequestState here creates a commit window
     * in which Registry is committed but lifecycle delivery has not started.
     */
    private void releasePrefillAccounting(ScheduledRequest item) {
        PrefillEndpoint prefillEp = item.prefillEp();
        if (prefillEp == null) {
            return;
        }
        if (prefillEp.releaseCommittedItem(item)) {
            Logger.debug("FlexLB release canonical Prefill accounting: request_id={} engine={}",
                    item.requestId(), prefillEp.getIp());
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

    RequestSlot.PublicationPermit publishExternalResponse(
            RequestSlot slot, Response response) {
        String detail = response != null && response.getErrorMessage() != null
                ? response.getErrorMessage() : "external future completion";
        Function<RequestSlot, RequestState> transition =
                response != null && !response.isSuccess()
                        ? owner -> owner.fail(detail)
                        : owner -> owner.complete(detail);
        TerminalAction action = claimExternalLocalTerminal(slot, transition);
        return action == null ? null : finishTerminal(action);
    }

    /** Apply terminal cleanup synchronously and publish the response off the decision thread. */
    boolean publishDecisionResponseAsync(
            long requestId,
            CompletableFuture<Response> future,
            Response response) {
        RequestSlot slot = requestSlots.get(requestId);
        if (slot == null || !slot.ownsFuture(future)) {
            return false;
        }
        RequestSlot.PublicationPermit permit = publishExternalResponse(
                slot, response);
        if (permit == null) {
            return false;
        }
        try {
            completionPublisher.submitTerminalResponse(permit, response);
            return true;
        } catch (RuntimeException | Error publicationFailure) {
            permit.abortClaimedPublication();
            throw publicationFailure;
        }
    }

    RequestSlot.PublicationPermit publishExternalFailure(
            RequestSlot slot, Throwable error) {
        Objects.requireNonNull(error, "error");
        String detail = "external future failure"
                + (error.getMessage() == null ? "" : ": " + error.getMessage());
        TerminalAction action = claimExternalLocalTerminal(
                slot, owner -> owner.fail(detail));
        return action == null ? null : finishTerminal(action);
    }

    RequestSlot.PublicationPermit publishExternalCancellation(
            RequestSlot slot) {
        String detail = cancelDetail(CancelReason.CLIENT_CANCELLED);
        TerminalAction action = claimExternalLocalTerminal(
                slot, owner -> owner.cancel(detail));
        return action == null ? null : finishTerminal(action);
    }

    /**
     * Claim only a locally reversible exact slot. The returned action is a
     * one-shot capability; cleanup and tombstone publication happen
     * synchronously before any public CompletableFuture state becomes visible.
     */
    private TerminalAction claimExternalLocalTerminal(
            RequestSlot slot,
            Function<RequestSlot, RequestState> transition) {
        synchronized (slot) {
            if (!isCurrentSlot(slot) || !slot.canClaimLocalTerminal()) {
                return null;
            }
            return slot.beginExternalTerminalizing(transition);
        }
    }

    static Response buildErrorResponse(StrategyErrorType errorType,
                                               String message) {
        Response errorResp = Response.error(errorType);
        errorResp.setErrorMessage(errorType.buildErrorMessage(message));
        return errorResp;
    }

    static Throwable appendFailure(Throwable first, Throwable next) {
        if (first == null) {
            return next;
        }
        if (first != next) {
            first.addSuppressed(next);
        }
        return first;
    }

    private static boolean batchMatches(RequestState snapshot,
                                        long expectedBatchId) {
        if (snapshot == null) {
            return false;
        }
        return expectedBatchId == 0 || snapshot.batchId() == expectedBatchId;
    }

    // ==================== Internal: static utilities ====================

    private static Response copyResponse(Response src) {
        Response response = new Response();
        response.setServerStatus(copyServerList(src.getServerStatus()));
        response.setSuccess(src.isSuccess());
        response.setCode(src.getCode());
        response.setErrorMessage(src.getErrorMessage());
        response.setRealMasterHost(src.getRealMasterHost());
        response.setQueueLength(src.getQueueLength());
        response.setEnqueuedByMaster(src.isEnqueuedByMaster());
        response.setAdmissionRejectReason(src.getAdmissionRejectReason());
        return response;
    }

    private static List<ServerStatus> copyServerList(List<ServerStatus> src) {
        if (src == null) {
            return null;
        }
        List<ServerStatus> result = new ArrayList<>(src.size());
        for (ServerStatus serverStatus : src) {
            result.add(copyOf(serverStatus));
        }
        return result;
    }

    /** Defensive copy at the scheduler's exact queue-admission boundary. */
    static ServerStatus copyOf(ServerStatus src) {
        if (src == null) {
            return null;
        }
        ServerStatus status = new ServerStatus();
        status.setRole(src.getRole());
        status.setServerIp(src.getServerIp());
        status.setHttpPort(src.getHttpPort());
        status.setGrpcPort(src.getGrpcPort());
        status.setDpRank(src.getDpRank());
        status.setPrefillTime(src.getPrefillTime());
        status.setGroup(src.getGroup());
        status.setDebugInfo(copyOf(src.getDebugInfo()));
        status.setRequestId(src.getRequestId());
        status.setSuccess(src.isSuccess());
        status.setCode(src.getCode());
        status.setMessage(src.getMessage());
        return status;
    }

    private static DebugInfo copyOf(DebugInfo src) {
        if (src == null) {
            return null;
        }
        DebugInfo info = new DebugInfo();
        info.setRunningBatchSize(src.getRunningBatchSize());
        info.setQueueSize(src.getQueueSize());
        info.setWaitingTimeMs(src.getWaitingTimeMs());
        info.setAvailableKvCacheLen(src.getAvailableKvCacheLen());
        info.setEstimateTtftMs(src.getEstimateTtftMs());
        info.setEstimateTpotMs(src.getEstimateTpotMs());
        info.setHitCacheLen(src.getHitCacheLen());
        return info;
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

    /** Complete locally reversible requests before closing the publisher. */
    private void completeOutstandingRequestsForShutdown() {
        String detail = "request scheduler is shutting down";
        List<TerminalAction> publications = new ArrayList<>();
        // Registered requests are authoritative even when their caller did
        // not originate from submit() (for example an eviction admission
        // integration). Do not make shutdown publication depend on the
        // presence or concrete type of the generation gate.
        for (RequestSlot entry : requestSlots.values()) {
            synchronized (entry) {
                if (!isCurrentSlot(entry)
                        || !entry.canClaimLocalTerminal()) {
                    continue;
                }
                ScheduledRequest item = entry.activeItem();
                TerminalAction publication = beginTerminalLocked(
                        entry, item != null, item != null,
                        owner -> owner.fail(detail),
                        buildErrorResponse(
                                StrategyErrorType.DISPATCH_FAILED,
                                detail));
                if (publication != null) {
                    publications.add(publication);
                }
            }
        }
        for (TerminalAction publication : publications) {
            submitTerminal(publication);
        }
    }

}
