package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.balance.preemption.VictimTerminal;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.DebugInfo;
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
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiConsumer;
import java.util.function.BooleanSupplier;
import java.util.function.Function;
import java.util.function.LongPredicate;
import java.util.function.Supplier;

/**
 * Coordinates request scheduling for FlexLB disaggregated inference.
 *
 * <p>Responsibilities:
 * <ul>
 *   <li>Request admission and routing</li>
 *   <li>Inflight lifecycle management (requestSlots map, TTL cleanup)</li>
 *   <li>Priority decision-group coordination through {@link WorkerBatcher}</li>
 *   <li>Delivery-independent lifecycle and resource ownership</li>
 *   <li>Batch enqueue or caller-owned route-decision delivery</li>
 *   <li>Exact reservation release on failure or completion</li>
 * </ul>
 *
 * <p>Mode-specific admission and transport ownership live in a concrete
 * delivery strategy. This class exposes only exact RequestSlot capabilities;
 * it never owns dispatcher permits, batch ids, or route handoff resources.
 */
@Component
public class RequestRegistry {

    private static final long DEFAULT_CANCEL_ACK_TIMEOUT_MS = 50L;
    static final int OUTSTANDING_ADMISSION_CLOSED = -1;
    private static final int DECODE_ACCEPTANCE_CLOSED = -1;
    private static final Runnable NO_POST_LOCK_ACTION = () -> { };

    private final RequestCompletionPublisher completionPublisher;
    /** Sole semantic deadline/retention owner for the canonical slot directory. */
    private final ExpirationTimer expirationTimer;
    private final EngineCancelChannel engineCancelChannel;
    /** One-way lifecycle gate; terminal completions remain allowed after it closes. */
    private final AtomicBoolean shuttingDown = new AtomicBoolean();
    /**
     * Shutdown barrier for admission mutations which have crossed their slot
     * ownership boundary but have not yet published or transferred the exact
     * completion. The monitor is never held while acquiring a RequestSlot.
     */
    private final Object admissionQuiescenceMonitor = new Object();
    private int inFlightAdmissionMutations;
    /**
     * Exact cluster-wide QUEUE ownership bound. Unlike {@code requestSlots.size()},
     * this counter includes admissions which have not reached registration yet.
     * The CAS increment is the capacity linearization point for every submit.
     */
    private final AtomicInteger outstandingRequestCount = new AtomicInteger();
    /** QUEUE requests whose Decode-acceptance guard remains active. */
    private final AtomicInteger decodeAcceptanceCount = new AtomicInteger();
    private final BatchSchedulerReporter reporter;
    private final RequestSchedulerReporter requestReporter;
    /** The sole canonical owner for admission, delivery, fence and terminal state. */
    private final ConcurrentMap<Long, RequestSlot> requestSlots =
            new ConcurrentHashMap<>();
    @Autowired
    public RequestRegistry(ConfigService configService,
                            BatchSchedulerReporter reporter,
                            RequestSchedulerReporter requestReporter,
                            EngineCancelChannel engineCancelChannel) {
        this.reporter = Objects.requireNonNull(reporter, "reporter");
        this.requestReporter = Objects.requireNonNull(requestReporter);
        this.expirationTimer = new ExpirationTimer(
                this,
                Objects.requireNonNull(configService, "configService"),
                reporter);
        this.engineCancelChannel = Objects.requireNonNull(
                engineCancelChannel, "engineCancelChannel");
        this.completionPublisher = new RequestCompletionPublisher(this);
    }

    /** Reserve one global request slot without a check-then-act window. */
    private boolean tryAcquireOutstandingPermit(int limit) {
        while (true) {
            int current = outstandingRequestCount.get();
            if (current == OUTSTANDING_ADMISSION_CLOSED
                    || current == Integer.MAX_VALUE
                    || (limit > 0 && current >= limit)) {
                return false;
            }
            if (outstandingRequestCount.compareAndSet(current, current + 1)) {
                return true;
            }
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
        cancelRequest(
                exactSlot.requestId(), 0L,
                CancelReason.DEADLINE_EXCEEDED);
    }

    public void acceptanceExpired(RequestSlot.AcceptanceExpiry expiry) {
        releaseAdmissionCleanup(expiry.cleanup());
        if (expiry.needsFence() && expiry.item() != null) {
            fenceAfterDeliveryTimeout(
                    expiry.item(),
                    "post_delivery_acceptance_timeout");
        }
    }

    public boolean reduceStale(
            RequestSlot exactSlot,
            long nowMs,
            long staleTtlMs) {
        return reduceStaleSlot(exactSlot, nowMs, staleTtlMs);
    }

    // ==================== Request submission ====================

    CompletableFuture<Response> register(
            BalanceContext context,
            int maxOutstanding) {
        if (context == null || context.getRequest() == null) {
            return CompletableFuture.completedFuture(buildErrorResponse(
                    StrategyErrorType.INVALID_REQUEST, null));
        }
        if (shuttingDown.get()) {
            return CompletableFuture.completedFuture(buildErrorResponse(
                    StrategyErrorType.BATCH_DISPATCH_FAILED,
                    "request scheduler is shutting down"));
        }

        RequestSlot slot = new RequestSlot(
                completionPublisher,
                context.getRequestId());
        RequestFuture future = slot.future();
        boolean registered = false;
        try {
            context.setEnqueueTime(System.currentTimeMillis());
            RequestSlot prior =
                    requestSlots.putIfAbsent(context.getRequestId(), slot);
            if (prior != null) {
                return CompletableFuture.completedFuture(buildErrorResponse(
                        StrategyErrorType.INVALID_REQUEST,
                        "duplicate request_id: " + context.getRequestId()));
            }
            registered = true;
            synchronized (slot) {
                slot.configureDeadlineError(
                        StrategyErrorType.BATCH_SLO_EXPIRED);
            }
            if (!tryAcquireOutstandingPermit(maxOutstanding)) {
                completeError(
                        future,
                        shuttingDown.get()
                                || outstandingRequestCount.get()
                                        == OUTSTANDING_ADMISSION_CLOSED
                                ? StrategyErrorType.BATCH_DISPATCH_FAILED
                                : StrategyErrorType.QUEUE_FULL,
                        shuttingDown.get()
                                ? "request scheduler is shutting down" : null);
                return future;
            }
            synchronized (slot) {
                if (!slot.bindOutstandingPermit(outstandingRequestCount)) {
                    return future;
                }
            }
            if (shuttingDown.get()) {
                completeError(
                        future,
                        StrategyErrorType.BATCH_DISPATCH_FAILED,
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
                        future,
                        StrategyErrorType.BATCH_DISPATCH_FAILED,
                        detail);
                return future;
            }
            return CompletableFuture.completedFuture(buildErrorResponse(
                    StrategyErrorType.BATCH_DISPATCH_FAILED, detail));
        }
    }

    /**
     * Schedule request expiration as a reducer event. Directly attaching
     * Attaching a timeout directly to the public future would let the timer
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
        expirationTimer.attachRequestDeadline(
                slot, context.getRequestExpiresAtMs());
    }

    // ==================== Exact inflight commit protocol ====================

    /**
     * Register a priority-admitted item into the shared requestSlots tracking so
     * dispatch, completion, expiration, and rollback behave identically.
     * Mirrors the duplicate-request check in {@link #submit}.
     */
    boolean commitItemForPublication(
            ScheduledRequest item,
            boolean priorityAdmission,
            BooleanSupplier publication) {
        Objects.requireNonNull(publication, "publication");
        if (shuttingDown.get() || item == null || item.future().isDone()) {
            return false;
        }
        RequestSlot slot = requestSlots.get(item.requestId());
        if (slot == null || !slot.ownsFuture(item.future())) {
            return false;
        }
        synchronized (slot) {
            if (!isCurrentSlot(slot)
                    || !slot.tryBindItemForPublication(
                            item, priorityAdmission)) {
                return false;
            }
        }

        try {
            // Endpoint publication may acquire its queue lock. It must never
            // run while the exact RequestSlot monitor is held.
            if (publication.getAsBoolean()) {
                return true;
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
        return false;
    }

    PlacementResult.Status commitRoute(
            ScheduledRequest item,
            boolean priorityAdmission,
            int acceptanceLimit,
            long acceptanceTimeoutMs,
            BooleanSupplier publication) {
        Objects.requireNonNull(publication, "publication");
        if (item == null || item.decodeEp() == null) {
            return commitItemForPublication(item, priorityAdmission, publication)
                    ? PlacementResult.Status.SUCCESS
                    : PlacementResult.Status.CLOSED;
        }
        if (acceptanceLimit < 0 || acceptanceTimeoutMs < 0L) {
            throw new IllegalArgumentException(
                    "Decode acceptance limits must be non-negative");
        }
        if (shuttingDown.get() || item.future().isDone()) {
            return PlacementResult.Status.CLOSED;
        }
        RequestSlot slot = requestSlots.get(item.requestId());
        if (slot == null || !slot.ownsFuture(item.future())) {
            return PlacementResult.Status.CLOSED;
        }

        boolean permitAcquired = false;
        try {
            RequestSlot.AdmissionCleanup immediate;
            synchronized (slot) {
                if (!isCurrentSlot(slot) || !slot.isOpen()) {
                    return PlacementResult.Status.CLOSED;
                }
                if (!tryAcquireDecodeAcceptancePermit(acceptanceLimit)) {
                    return PlacementResult.Status.LIMIT_REACHED;
                }
                permitAcquired = true;
                if (!slot.tryBindItemForPublication(
                        item, priorityAdmission)) {
                    return PlacementResult.Status.CLOSED;
                }
                immediate = slot.bindAdmissionResources(
                        this::releaseDecodeAcceptancePermit,
                        acceptanceTimeoutMs);
                permitAcquired = false;
                if (immediate != null) {
                    slot.rollbackItemPublication(item);
                }
            }
            if (immediate != null) {
                releaseAdmissionCleanup(immediate);
                return PlacementResult.Status.CLOSED;
            }
            if (publication.getAsBoolean()) {
                return PlacementResult.Status.SUCCESS;
            }
            releaseAdmissionCleanup(
                    rollbackAdmissionPublication(slot, item));
            return PlacementResult.Status.BLOCKED;
        } catch (RuntimeException | Error failure) {
            RequestSlot.AdmissionCleanup cleanup = null;
            try {
                synchronized (slot) {
                    if (slot.activeItem() == item) {
                        cleanup = slot.rollbackAdmissionPublication(item);
                    }
                }
            } catch (RuntimeException | Error rollbackFailure) {
                if (rollbackFailure != failure) {
                    failure.addSuppressed(rollbackFailure);
                }
            }
            releaseAdmissionCleanup(cleanup);
            throw failure;
        } finally {
            if (permitAcquired) {
                releaseDecodeAcceptancePermit();
            }
        }
    }

    private RequestSlot.AdmissionCleanup rollbackAdmissionPublication(
            RequestSlot slot, ScheduledRequest item) {
        synchronized (slot) {
            return slot.rollbackAdmissionPublication(item);
        }
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
            if (completion.tombstonedFence() != null) {
                resumeEngineFenceTombstoned(
                        slot, completion.tombstonedFence());
            } else if (completion.pendingTerminal() != null) {
                Runnable work;
                synchronized (slot) {
                    work = reduceDeferredTerminalFactLocked(
                            slot, completion.pendingTerminal());
                }
                runPostLock(work);
            } else if (completion.pendingRetirement() != null) {
                submitTerminal(completion.pendingRetirement());
            } else if (completion.cancellationToResume() != null) {
                resumeCancellationAfterAdmission(
                        slot, completion.cancellationToResume());
            }
        } finally {
            exitAdmissionMutationGate();
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
                if (completion.tombstonedFence() == null
                        && completion.pendingTerminal() != null) {
                    retainedWork = reduceDeferredTerminalFactLocked(
                            slot, completion.pendingTerminal());
                } else if (completion.tombstonedFence() == null
                        && completion.pendingRetirement() == null) {
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
            if (completion.tombstonedFence() != null) {
                resumeEngineFenceTombstoned(
                        slot, completion.tombstonedFence());
            } else if (retainedWork != null) {
                runPostLock(retainedWork);
            } else if (completion.pendingRetirement() != null) {
                submitTerminal(completion.pendingRetirement());
            } else {
                submitTerminal(action);
            }
        } finally {
            exitAdmissionMutationGate();
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
            CancelReason reason) {
        RequestSlot.FenceReduction fenceReduction = null;
        TerminalAction localCompletion = null;
        synchronized (entry) {
            if (!isCurrentSlot(entry)
                    || entry.snapshot().state().isTerminal()) {
                return;
            }
            CancelReason firstCause = entry.requireCancellationFirstCause();
            if (firstCause != reason) {
                throw new IllegalStateException(
                        "admission cancellation first cause changed for request "
                                + entry.requestId());
            }
            String detail = cancelDetail(firstCause);
            RequestState current = entry.snapshot();
            ScheduledRequest item = entry.activeItem();
            if (item == null) {
                localCompletion = beginTerminalLocked(
                        entry,
                        false,
                        false,
                        owner -> settleCancellationLifecycle(
                                owner, firstCause, detail),
                        buildErrorResponse(
                                entry.cancellationErrorType(firstCause),
                                detail));
            } else if (entry.canClaimLocalTerminal()
                    && isLocallyReversible(current)) {
                localCompletion = beginTerminalLocked(
                        entry,
                        true,
                        true,
                        owner -> settleCancellationLifecycle(
                                owner, firstCause, detail),
                        buildErrorResponse(
                                entry.cancellationErrorType(firstCause),
                                detail));
            } else {
                fenceReduction = entry.requestCancellationFence(detail);
            }
        }
        submitTerminal(localCompletion);
        consumeFenceStart(entry, fenceReduction, false);
    }

    int decodeAcceptanceCount() {
        return Math.max(0, decodeAcceptanceCount.get());
    }

    private boolean tryAcquireDecodeAcceptancePermit(int limit) {
        while (true) {
            int current = decodeAcceptanceCount.get();
            if (current == DECODE_ACCEPTANCE_CLOSED
                    || current == Integer.MAX_VALUE
                    || (limit > 0 && current >= limit)) {
                return false;
            }
            if (decodeAcceptanceCount.compareAndSet(current, current + 1)) {
                return true;
            }
        }
    }

    private void releaseDecodeAcceptancePermit() {
        while (true) {
            int current = decodeAcceptanceCount.get();
            if (current == DECODE_ACCEPTANCE_CLOSED) {
                return;
            }
            if (current <= 0) {
                throw new IllegalStateException(
                        "Decode acceptance permit counter underflow");
            }
            if (decodeAcceptanceCount.compareAndSet(current, current - 1)) {
                return;
            }
        }
    }

    private void fenceAfterDeliveryTimeout(ScheduledRequest item, String detail) {
        RequestSlot entry = entryFor(item);
        if (entry == null) {
            return;
        }
        RequestSlot.FenceReduction reduction;
        synchronized (entry) {
            if (!entry.ownsActiveItem(item)) {
                return;
            }
            reduction = entry.requestDeliveryFence(detail);
        }
        consumeFenceStart(entry, reduction, false);
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
        if (reduction.status()
                == RequestSlot.PreemptionReduction.Status.START_FENCE) {
            return () -> startFence(
                    entry, reduction.fence(), reduction.target());
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
                reduction.signal(),
                reduction.prefillOnlyCleanup());
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
            PreemptionRegistration terminalSignal,
            RequestSlot.ExactPrefillOnlyCleanup prefillOnlyCleanup) {
        return () -> {
            Throwable isolatedFailure = null;
            if (prefillOnlyCleanup != null) {
                try {
                    prefillOnlyCleanup.release();
                } catch (Throwable cleanupFailure) {
                    isolatedFailure = cleanupFailure;
                }
            }
            try {
                publication.run();
            } finally {
                if (terminalSignal != null) {
                    terminalSignal.signalTerminal(
                            new VictimTerminal(entry.requestId()));
                }
            }
            if (isolatedFailure != null) {
                Logger.error(
                        "Preemption cleanup isolated: request_id={}",
                        entry.requestId(), isolatedFailure);
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
     * Cancel one request generation owned by this scheduler.
     *
     * <p>This method is the only reducer for the frontend-facing Cancel RPC.
     * Requests which have not crossed an external delivery boundary are
     * released locally.  Once an EnqueueBatch send has started, or a route
     * decision may have been published, the existing request-scoped Engine
     * fence owns reconciliation and resources remain charged until an
     * authoritative terminal is observed.</p>
     *
     * @return the current lifecycle for the matching request generation, or
     *         {@code null} when the request is unknown or {@code batchId}
     *         addresses a different generation
     */
    public RequestState cancelRequest(long requestId,
                                                   long expectedBatchId,
                                                   CancelReason reason) {
        Objects.requireNonNull(reason, "reason");
        RequestSlot entry = requestSlots.get(requestId);
        if (entry == null) {
            return null;
        }
        RequestSlot.FenceReduction fenceReduction = null;
        TerminalAction localCompletion = null;
        RequestState result;
        synchronized (entry) {
            if (!isCurrentSlot(entry)) {
                return matchingTerminalState(requestId, expectedBatchId);
            }
            RequestState current = entry.snapshot();
            if (!batchMatches(current, expectedBatchId)) {
                return null;
            }
            if (current.state().isTerminal()
                    || entry.hasCancellationFirstCause()) {
                return current;
            }
            if (reason == CancelReason.DEADLINE_EXCEEDED
                    && current.deliveryClaimKind()
                        != DeliveryClaimKind.NONE) {
                return current;
            }

            String detail = cancelDetail(reason);
            if (entry.deferCancellationDuringAdmission(reason, detail)) {
                return entry.snapshot();
            }
            ScheduledRequest item = entry.activeItem();
            if (item == null) {
                entry.rememberCancellation(reason, detail);
                localCompletion = beginTerminalLocked(
                        entry,
                        false,
                        false,
                        owner -> settleCancellationLifecycle(
                                owner, reason, detail),
                        buildErrorResponse(
                                entry.cancellationErrorType(reason), detail));
            } else {
                entry.rememberCancellation(reason, detail);
                if (entry.canClaimLocalTerminal()
                        && isLocallyReversible(current)) {
                    localCompletion = beginTerminalLocked(
                            entry,
                            true,
                            true,
                            owner -> settleCancellationLifecycle(
                                    owner, reason, detail),
                            buildErrorResponse(
                                    entry.cancellationErrorType(reason),
                                    detail));
                } else {
                    fenceReduction = entry.requestCancellationFence(detail);
                }
            }
            result = entry.snapshot();
        }
        submitTerminal(localCompletion);
        consumeFenceStart(entry, fenceReduction, false);
        return result;
    }

    private RequestState matchingTerminalState(long requestId,
                                                            long expectedBatchId) {
        RequestSlot slot = requestSlots.get(requestId);
        RequestState terminal = slot == null
                ? null : slot.snapshot();
        return batchMatches(terminal, expectedBatchId) ? terminal : null;
    }

    /** Called with both delivery and entry ownership held. */
    private static boolean isLocallyReversible(
            RequestState snapshot) {
        // DeliveryClaimKind is the single point-of-no-return. Endpoint capacity
        // and this claim commit in one delivery transaction, so consulting a
        // timestamp or later ACK would recreate a second ownership mirror.
        return snapshot.deliveryClaimKind() == DeliveryClaimKind.NONE;
    }

    private static CancelTarget cancelTarget(
            ScheduledRequest item) {
        ServerStatus prefill = item == null ? null : item.prefill();
        return prefill == null ? null
                : new CancelTarget(
                        prefill.getServerIp(), prefill.getGrpcPort());
    }

    private void startFence(
            RequestSlot entry,
            RequestSlot.EngineFenceRegistration fence,
            CancelTarget target) {
        if (fence == null) {
            return;
        }
        if (target == null) {
            synchronized (entry) {
                RequestSlot.FenceReduction reduction = entry.applyFenceUpdate(
                        fence, RequestSlot.FenceUpdate.AWAIT_TERMINAL);
                if (reduction.status()
                            != RequestSlot.FenceReduction.Status.NONE
                        && reduction.status()
                            != RequestSlot.FenceReduction.Status.STALE) {
                    throw new IllegalStateException(
                            "await-terminal produced an invalid fence effect: "
                                    + reduction.getClass().getSimpleName());
                }
            }
            return;
        }
        startEngineFenceCancel(
                entry, fence, target, DEFAULT_CANCEL_ACK_TIMEOUT_MS);
    }

    /** Invoke Cancel once after the exact fence crosses its send boundary. */
    private void startEngineFenceCancel(
            RequestSlot slot,
            RequestSlot.EngineFenceRegistration fence,
            CancelTarget target,
            long timeoutMs) {
        synchronized (slot) {
            RequestSlot.FenceReduction reduction = slot.applyFenceUpdate(
                    fence, RequestSlot.FenceUpdate.CANCEL_STARTED);
            if (reduction.status()
                    == RequestSlot.FenceReduction.Status.STALE) {
                return;
            }
            requireNoFenceEffect(reduction, "Cancel start");
        }

        try {
            CompletableFuture<EngineCancelChannel.CancelAck> outcome =
                    engineCancelChannel.cancel(
                            target, slot.requestId(), timeoutMs);
            if (outcome == null) {
                awaitAuthoritativeTerminal(slot, fence);
                return;
            }
            outcome.whenComplete((ack, failure) -> {
                try {
                    if (failure == null
                            && ack == EngineCancelChannel.CancelAck.TOMBSTONED) {
                        resumeEngineFenceTombstoned(slot, fence);
                    } else {
                        awaitAuthoritativeTerminal(slot, fence);
                    }
                } catch (Throwable invariantFailure) {
                    Logger.error(
                            "Engine fence completion invariant failed: request_id={}",
                            slot.requestId(), invariantFailure);
                }
            });
        } catch (RuntimeException | Error invocationFailure) {
            // A synchronous transport failure cannot prove that Cancel did not
            // reach the engine. Retain the fence until endpoint evidence wins.
            awaitAuthoritativeTerminal(slot, fence);
        }
    }

    /** Consume an exact TOMBSTONED proof or retain it for admission replay. */
    private void resumeEngineFenceTombstoned(
            RequestSlot slot,
            RequestSlot.EngineFenceRegistration fence) {
        RequestSlot.FenceReduction reduction;
        synchronized (slot) {
            reduction = slot.applyFenceUpdate(
                    fence, RequestSlot.FenceUpdate.TOMBSTONED);
        }
        if (reduction.status() == RequestSlot.FenceReduction.Status.DEFERRED
                || reduction.status()
                        == RequestSlot.FenceReduction.Status.STALE) {
            return;
        }
        if (reduction.status()
                != RequestSlot.FenceReduction.Status.TERMINAL_PROOF) {
            throw new IllegalStateException(
                    "Engine fence TOMBSTONED produced an invalid effect: "
                            + reduction.status());
        }

        boolean terminalized = onEngineFenceTombstoned(
                slot, reduction.proof());
        synchronized (slot) {
            RequestSlot.FenceReduction actual = slot.applyFenceUpdate(
                    fence, RequestSlot.FenceUpdate.TOMBSTONED);
            if (terminalized ? !slot.isTombstone()
                    : actual.status()
                            != RequestSlot.FenceReduction.Status.STALE) {
                throw new IllegalStateException(
                        "Engine fence TOMBSTONED proof was not consumed: request_id="
                                + slot.requestId());
            }
        }
    }

    private static void awaitAuthoritativeTerminal(
            RequestSlot slot,
            RequestSlot.EngineFenceRegistration fence) {
        synchronized (slot) {
            RequestSlot.FenceReduction reduction = slot.applyFenceUpdate(
                    fence, RequestSlot.FenceUpdate.AWAIT_TERMINAL);
            if (reduction.status()
                    != RequestSlot.FenceReduction.Status.STALE) {
                requireNoFenceEffect(reduction, "await terminal");
            }
        }
    }

    private static void requireNoFenceEffect(
            RequestSlot.FenceReduction reduction,
            String operation) {
        if (reduction.status() != RequestSlot.FenceReduction.Status.NONE) {
            throw new IllegalStateException(
                    operation + " produced an invalid Engine fence effect: "
                            + reduction.status());
        }
    }

    private void consumeFenceStart(
            RequestSlot entry,
            RequestSlot.FenceReduction reduction,
            boolean requireOwnedEffect) {
        if (reduction == null) {
            return;
        }
        if (reduction.status()
                == RequestSlot.FenceReduction.Status.START) {
            startFence(entry, reduction.fence(), reduction.target());
            return;
        }
        if (reduction.status()
                == RequestSlot.FenceReduction.Status.NONE) {
            return;
        }
        if (reduction.status()
                    == RequestSlot.FenceReduction.Status.STALE
                && !requireOwnedEffect) {
            return;
        }
        throw new IllegalStateException(
                "fence request produced an invalid effect: "
                        + reduction.getClass().getSimpleName());
    }

    private boolean onEngineFenceTombstoned(
                    RequestSlot entry,
                    RequestSlot.FenceTerminalProof proof) {
        TerminalAction action;
        PreemptionRegistration transferred;
        Throwable isolatedFailure;
        synchronized (entry) {
            ScheduledRequest item = entry.activeItem();
            DecodeEndpoint endpoint = item == null ? null : item.decodeEp();
            transferred = proof.transferred();
            isolatedFailure = null;
            if (endpoint != null && transferred != null) {
                try {
                    endpoint.settleEngineFenceClaim(
                            transferred.attemptToken(),
                            item.decodeReservation());
                } catch (Throwable claimFailure) {
                    isolatedFailure = claimFailure;
                }
            }
            DecodeEndpoint.AuthoritativeTerminalProof decodeProof =
                    proof.decodeProof();
            if (endpoint != null && decodeProof != null) {
                try {
                    endpoint.settleAuthoritativeTerminal(decodeProof);
                } catch (Throwable proofFailure) {
                    isolatedFailure = appendFailure(
                            isolatedFailure, proofFailure);
                }
            }
            String proofDetail = proof.detail()
                    + "; engine reported TOMBSTONED";
            action = entry.hasCancellationFirstCause()
                    ? settleCancellationAfterEndpointSettlementLocked(
                            entry,
                            proofDetail,
                            item == null ? null
                                    : exactPrefillCounterpartCleanup(item))
                    : timeoutAfterEndpointSettlementLocked(
                            entry,
                            proofDetail,
                            item == null ? null
                                    : exactPrefillCounterpartCleanup(item));
            if (action == null) {
                return false;
            }
        }
        try {
            submitTerminal(action);
        } catch (Throwable publicationFailure) {
            isolatedFailure = appendFailure(
                    isolatedFailure, publicationFailure);
        }
        if (transferred != null) {
            try {
                transferred.signalTerminal(
                        new VictimTerminal(entry.requestId()));
            } catch (Throwable signalFailure) {
                isolatedFailure = appendFailure(
                        isolatedFailure, signalFailure);
            }
        }
        if (isolatedFailure != null) {
            Logger.error(
                    "Engine fence TOMBSTONED settlement isolated: request_id={}",
                    entry.requestId(), isolatedFailure);
        }
        return true;
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
            case FAILURE -> applyFailureTerminalLocked(entry, terminal, false);
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
                        StrategyErrorType.BATCH_DISPATCH_FAILED, detail)));
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

    private boolean reduceStaleSlot(
            RequestSlot exactSlot,
            long nowMs,
            long staleTtlMs) {
        TerminalAction direct = null;
        Runnable work = null;
        boolean terminalized = false;
        synchronized (exactSlot) {
            if (!isCurrentSlot(exactSlot)
                    || !exactSlot.isLiveGeneration()
                    || nowMs - exactSlot.lastWorkerStatusAtMs() <= staleTtlMs
                    || exactSlot.hasCancellationFirstCause()) {
                return false;
            }
            String detail = "inflight inactive TTL expired";
            if (exactSlot.activeItem() == null) {
                direct = beginTerminalLocked(
                        exactSlot,
                        false,
                        false,
                        owner -> owner.timeout(detail),
                        buildErrorResponse(
                                exactSlot.timeoutErrorType(), detail));
            } else {
                RequestSlot.PreemptionReduction reduction =
                        exactSlot.reduceOrdinaryTerminal(
                                exactSlot.activeItem(),
                                DeferredTerminal.timeout(detail));
                terminalized = reduction.status()
                        == RequestSlot.PreemptionReduction.Status.REPLAY;
                work = materializePostLockActionLocked(
                        exactSlot, reduction, null);
            }
        }
        submitTerminal(direct);
        runPostLock(work);
        return direct != null || terminalized;
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
        String failureDetail = error == null ? "queue full" : error.getMessage();
        RequestSlot entry = entryFor(item);
        if (entry != null) {
            Runnable work;
            synchronized (entry) {
                work = reduceDeferredTerminalFactLocked(entry,
                        DeferredTerminal.failure(
                                StrategyErrorType.BATCH_DISPATCH_FAILED,
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
        RequestSlot.FenceReduction fenceReduction = null;
        synchronized (exact.slot) {
            if (exact.completed || !ownsDeliveryClaim(exact)) {
                throw new IllegalStateException(
                        "delivery claim is stale or already completed: request_id="
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
                        case CONFLICT -> fenceReduction =
                                exact.slot.requestDeliveryFence(detail);
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
                fenceReduction = exact.slot.requestDeliveryFence(detail);
            }
        }
        runPostLock(work);
        consumeFenceStart(exact.slot, fenceReduction, true);
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
        long nowMs = System.currentTimeMillis();
        item.ctx().setAckAtMs(nowMs);
        item.ctx().setAckAtNanos(System.nanoTime());
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
        cleanupFailure = runTerminalLeaf(
                cleanupFailure,
                action.fence() == null
                        ? null : action.fence()::release);
        cleanupFailure = runTerminalLeaf(
                cleanupFailure, action.counterpartCleanup());
        cleanupFailure = runTerminalLeaf(
                cleanupFailure,
                action.publication() == null
                        ? entry::releaseOutstandingPermit : null);

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
        preparationFailure = runTerminalLeaf(
                preparationFailure,
                confirmation.admissionCleanup() == null
                        ? null : () -> expirationTimer.release(
                                confirmation.admissionCleanup()));
        preparationFailure = runTerminalLeaf(
                preparationFailure,
                confirmation.armAcceptanceDeadline()
                        ? () -> armAcceptanceDeadline(slot)
                        : null);
        if (deliveryKind == DeliveryClaimKind.BATCH_ENQUEUE
                && confirmation.batchEnqueueStartedAtMs() > 0L) {
            long latencyMs = Math.max(
                    0L,
                    System.currentTimeMillis()
                            - confirmation.batchEnqueueStartedAtMs());
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

    private void armAcceptanceDeadline(RequestSlot entry) {
        java.util.OptionalLong delay;
        synchronized (entry) {
            delay = entry.acceptanceDeadlineDelayMs();
        }
        if (delay.isPresent()) {
            expirationTimer.registerAcceptanceDeadline(
                    entry, delay.getAsLong());
        }
    }

    void releaseAdmissionCleanup(
            RequestSlot.AdmissionCleanup cleanup) {
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
        replaceDecodeStatus(success, item.decode());
        success.setSuccess(true);
        success.setCode(200);
        success.setEnqueuedByMaster(
                deliveryKind == DeliveryClaimKind.BATCH_ENQUEUE);
        // This method is called while the exact RequestSlot is locked.  Do not
        // traverse and lock every other slot here: concurrent batch completions
        // would each hold their own slot and wait on one another.  The admission
        // permit counter is the lock-free, cluster-wide outstanding snapshot.
        success.setQueueLength(Math.max(0, outstandingRequestCount.get()));
        return success;
    }

    private static void replaceDecodeStatus(
            Response response,
            ServerStatus currentDecode) {
        if (currentDecode == null || response.getServerStatus() == null) {
            return;
        }
        List<ServerStatus> statuses = response.getServerStatus();
        for (int index = 0; index < statuses.size(); index++) {
            ServerStatus status = statuses.get(index);
            if (status != null && status.getRole() == RoleType.DECODE) {
                statuses.set(index, copyOf(currentDecode));
                return;
            }
        }
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
                                StrategyErrorType.BATCH_DISPATCH_FAILED,
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

    private TerminalAction timeoutAfterEndpointSettlementLocked(
            RequestSlot entry,
            String detail,
            Runnable counterpartCleanup) {
        return beginWorkerStatusTerminalLocked(
                entry,
                counterpartCleanup,
                owner -> owner.timeout(detail),
                buildErrorResponse(entry.timeoutErrorType(), detail));
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
        outstandingRequestCount.getAndSet(OUTSTANDING_ADMISSION_CLOSED);
        decodeAcceptanceCount.getAndSet(DECODE_ACCEPTANCE_CLOSED);
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

    /**
     * Complete only locally reversible requests during shutdown. A request
     * behind a real asynchronous Engine fence is deliberately not published:
     * shutdown is not an authoritative Engine-terminal proof.
     */
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
                                StrategyErrorType.BATCH_DISPATCH_FAILED,
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
