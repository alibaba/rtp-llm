package org.flexlb.balance.scheduler;

import org.flexlb.balance.admission.AdmissionFailure;
import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.balance.endpoint.EndpointEventSink;
import org.flexlb.balance.delivery.SlotDeliveryPort;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.balance.eviction.DecodePreemptionCoordinator.PreemptionUpdate;
import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.balance.preemption.PreemptionClaim;
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
public class RequestRegistry implements SlotDeliveryPort {

    private static final long DEFAULT_CANCEL_ACK_TIMEOUT_MS = 50L;
    static final int OUTSTANDING_ADMISSION_CLOSED = -1;
    private static final int DECODE_ACCEPTANCE_CLOSED = -1;

    enum RouteCommitResult {
        PUBLISHED,
        REQUEST_CLOSED,
        ACCEPTANCE_LIMIT_REACHED,
        PUBLICATION_REJECTED
    }

    private final RequestCompletionPublisher completionPublisher;
    /** Sole semantic deadline/retention owner for the canonical slot directory. */
    private final ExpirationTimer expirationTimer;
    private final EngineFenceCoordinator engineFenceCoordinator;
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
        Objects.requireNonNull(engineCancelChannel, "engineCancelChannel");
        this.completionPublisher = new RequestCompletionPublisher(this);
        this.engineFenceCoordinator = new EngineFenceCoordinator(
                engineCancelChannel, this::onEngineFenceTombstoned);
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

    public boolean isCurrent(RequestSlot exactSlot) {
        return isCurrentSlot(exactSlot);
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
        if (expiry.consumed() && expiry.needsFence()
                && expiry.item() != null) {
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

    RouteCommitResult commitRoute(
            ScheduledRequest item,
            boolean priorityAdmission,
            int acceptanceLimit,
            long acceptanceTimeoutMs,
            BooleanSupplier publication) {
        Objects.requireNonNull(publication, "publication");
        if (item == null || item.decodeEp() == null) {
            return commitItemForPublication(item, priorityAdmission, publication)
                    ? RouteCommitResult.PUBLISHED
                    : RouteCommitResult.REQUEST_CLOSED;
        }
        if (acceptanceLimit < 0 || acceptanceTimeoutMs < 0L) {
            throw new IllegalArgumentException(
                    "Decode acceptance limits must be non-negative");
        }
        if (shuttingDown.get() || item.future().isDone()) {
            return RouteCommitResult.REQUEST_CLOSED;
        }
        RequestSlot slot = requestSlots.get(item.requestId());
        if (slot == null || !slot.ownsFuture(item.future())) {
            return RouteCommitResult.REQUEST_CLOSED;
        }

        boolean permitAcquired = false;
        try {
            RequestSlot.AdmissionCleanup immediate;
            synchronized (slot) {
                if (!isCurrentSlot(slot) || !slot.isOpen()) {
                    return RouteCommitResult.REQUEST_CLOSED;
                }
                if (!tryAcquireDecodeAcceptancePermit(acceptanceLimit)) {
                    return RouteCommitResult.ACCEPTANCE_LIMIT_REACHED;
                }
                permitAcquired = true;
                if (!slot.tryBindItemForPublication(
                        item, priorityAdmission)) {
                    return RouteCommitResult.REQUEST_CLOSED;
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
                return RouteCommitResult.REQUEST_CLOSED;
            }
            if (publication.getAsBoolean()) {
                return RouteCommitResult.PUBLISHED;
            }
            releaseAdmissionCleanup(
                    rollbackAdmissionPublication(slot, item));
            return RouteCommitResult.PUBLICATION_REJECTED;
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

    AdmissionScope beginAdmission(
            long requestId, CompletableFuture<?> future) {
        AdmissionMutation exact = claimAdmissionMutation(requestId, future);
        return exact == null ? null : new AdmissionScope(exact);
    }

    static final class AdmissionScope implements AutoCloseable {
        private final AdmissionMutation exact;

        private AdmissionScope(AdmissionMutation exact) {
            this.exact = exact;
        }

        @Override
        public void close() {
            exact.close();
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
                engineFenceCoordinator.resumeTombstoned(
                        slot, completion.tombstonedFence());
            } else if (completion.pendingTerminal() != null) {
                PreemptionWork work;
                synchronized (slot) {
                    work = reduceDeferredTerminalFactLocked(
                            slot, completion.pendingTerminal());
                }
                consumePreemptionWork(slot, work);
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
            PreemptionWork retainedWork = null;
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
                    Function<RequestSlot, RequestState.Snapshot> transition;
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
                engineFenceCoordinator.resumeTombstoned(
                        slot, completion.tombstonedFence());
            } else if (retainedWork != null) {
                consumePreemptionWork(slot, retainedWork);
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
            RequestState.Snapshot current = entry.snapshot();
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
            PreemptionWork work;
            synchronized (entry) {
                work = reduceDeferredTerminalFactLocked(entry,
                        DeferredTerminal.failure(errorType, detail));
            }
            consumePreemptionWork(entry, work);
        }
    }

    public Optional<PreemptionClaim> tryClaim(
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

    public boolean tryApplyUpdate(
            PreemptionClaim claim,
            PreemptionUpdate update) {
        if (claim == null || update == null) {
            return false;
        }
        RequestSlot entry = requestSlots.get(claim.requestId());
        if (entry == null) {
            return false;
        }
        PreemptionWork work;
        synchronized (entry) {
            Runnable counterpartCleanup =
                    update instanceof PreemptionUpdate.Tombstoned
                    ? exactPrefillCounterpartCleanup(entry.activeItem())
                    : null;
            RequestSlot.PreemptionReduction reduction =
                    entry.applyPreemptionUpdate(claim, update);
            work = materializePreemptionReductionLocked(
                    entry, reduction, counterpartCleanup);
        }
        consumePreemptionWork(entry, work);
        return work.accepted();
    }

    PreemptionWork materializePreemptionWorkLocked(
            RequestSlot entry,
            RequestSlot.PreemptionReduction reduction,
            Runnable priorityCounterpartCleanup) {
        if (!Thread.holdsLock(entry)) {
            throw new IllegalStateException(
                    "preemption reduction requires slot lock");
        }
        return materializePreemptionReductionLocked(
                entry, reduction, priorityCounterpartCleanup);
    }

    private PreemptionWork materializePreemptionReductionLocked(
            RequestSlot entry,
            RequestSlot.PreemptionReduction reduction,
            Runnable priorityCounterpartCleanup) {
        if (reduction.status()
                == RequestSlot.PreemptionReduction.Status.STALE) {
            return PreemptionWork.STALE;
        }
        if (reduction.status()
                == RequestSlot.PreemptionReduction.Status.NONE) {
            return PreemptionWork.ACCEPTED;
        }
        if (reduction.status()
                == RequestSlot.PreemptionReduction.Status.START_FENCE) {
            return new PreemptionWork(
                    true, null, reduction.fence(), reduction.target(),
                    null, null);
        }
        PublicationWork publication = materializeReplayLocked(
                entry, reduction.payload(), priorityCounterpartCleanup);
        if (publication == null) {
            throw new IllegalStateException(
                    "accepted preemption replay produced no publication for request "
                            + entry.requestId());
        }
        return new PreemptionWork(
                true,
                publication,
                null,
                null,
                reduction.signal(),
                reduction.prefillOnlyCleanup());
    }

    private PublicationWork materializeReplayLocked(
            RequestSlot entry,
            RequestSlot.ReplayPayload payload,
            Runnable priorityCounterpartCleanup) {
        if (payload instanceof RequestSlot.ReplayPayload.Terminal terminal) {
            DeferredTerminal exact = terminal.exact();
            if (exact.kind() == DeferredTerminal.Kind.PRIORITY) {
                return PublicationWork.terminal(
                        beginSettledPriorityTerminalLocked(
                                entry,
                                exact.detail(),
                                priorityCounterpartCleanup));
            }
            return applyOrdinaryTerminalLocked(entry, exact);
        }
        RequestSlot.ReplayPayload.Delivery delivery =
                (RequestSlot.ReplayPayload.Delivery) payload;
        return deliveryPublication(
                entry,
                delivery.item(),
                delivery.exact(),
                delivery.kind(),
                delivery.batchId());
    }

    void consumePreemptionWork(
            RequestSlot entry,
            PreemptionWork work) {
        if (work == null || !work.accepted()) {
            return;
        }
        Throwable isolatedFailure = null;
        if (work.prefillOnlyCleanup() != null) {
            try {
                work.prefillOnlyCleanup().release();
            } catch (Throwable cleanupFailure) {
                isolatedFailure = cleanupFailure;
            }
        }
        if (work.publication() != null) {
            try {
                submitPublication(work.publication());
            } finally {
                if (work.terminalSignal() != null) {
                    work.terminalSignal().signalTerminal(
                            new VictimTerminal(entry.requestId()));
                }
            }
        }
        if (work.fence() != null) {
            startFence(entry, work.fence(), work.target());
        }
        if (isolatedFailure != null) {
            Logger.error(
                    "Preemption cleanup isolated: request_id={}",
                    entry.requestId(), isolatedFailure);
        }
    }

    record PreemptionWork(
            boolean accepted,
            PublicationWork publication,
            RequestSlot.FenceHandle fence,
            CancelTarget target,
            PreemptionRegistration terminalSignal,
            RequestSlot.PrefillOnlyCleanup prefillOnlyCleanup) {
        private static final PreemptionWork STALE =
                new PreemptionWork(false, null, null, null, null, null);
        private static final PreemptionWork ACCEPTED =
                new PreemptionWork(true, null, null, null, null, null);
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
    public RequestState.Snapshot cancelRequest(long requestId,
                                                   long expectedBatchId,
                                                   CancelReason reason) {
        Objects.requireNonNull(reason, "reason");
        RequestSlot entry = requestSlots.get(requestId);
        if (entry == null) {
            return null;
        }
        RequestSlot.FenceReduction fenceReduction = null;
        TerminalAction localCompletion = null;
        RequestState.Snapshot result;
        synchronized (entry) {
            if (!isCurrentSlot(entry)) {
                return matchingTerminalState(requestId, expectedBatchId);
            }
            RequestState.Snapshot current = entry.snapshot();
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

    private RequestState.Snapshot matchingTerminalState(long requestId,
                                                            long expectedBatchId) {
        RequestSlot slot = requestSlots.get(requestId);
        RequestState.Snapshot terminal = slot == null
                ? null : slot.snapshot();
        return batchMatches(terminal, expectedBatchId) ? terminal : null;
    }

    /** Called with both delivery and entry ownership held. */
    private static boolean isLocallyReversible(
            RequestState.Snapshot snapshot) {
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
            RequestSlot.FenceHandle fence,
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
        engineFenceCoordinator.start(
                entry, fence, target, DEFAULT_CANCEL_ACK_TIMEOUT_MS);
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

    /** Caller holds the exact slot; TOMBSTONED settles every Decode fence leaf. */
    private FenceEndpointSettlement settleFenceEndpointTerminalLocked(
            RequestSlot entry,
            RequestSlot.FenceTerminalProof proof) {
        if (!Thread.holdsLock(entry)) {
            throw new IllegalStateException(
                    "Engine fence terminal settlement requires slot lock");
        }
        ScheduledRequest item = entry.activeItem();
        DecodeEndpoint endpoint = item == null ? null : item.decodeEp();
        PreemptionRegistration transferred = proof.transferred();
        Throwable failure = null;
        if (endpoint != null && transferred != null) {
            try {
                endpoint.settleEngineFenceClaim(
                        transferred.attemptToken(),
                        item.decodeReservation());
            } catch (Throwable claimFailure) {
                failure = claimFailure;
            }
        }
        DecodeEndpoint.AuthoritativeTerminalProof decodeProof =
                proof.decodeProof();
        if (endpoint != null && decodeProof != null) {
            try {
                endpoint.settleAuthoritativeTerminal(decodeProof);
            } catch (Throwable proofFailure) {
                failure = appendFailure(failure, proofFailure);
            }
        }
        return new FenceEndpointSettlement(transferred, failure);
    }

    private EngineFenceCoordinator.TerminalDisposition
            onEngineFenceTombstoned(
                    RequestSlot entry,
                    RequestSlot.FenceTerminalProof proof) {
        TerminalAction action;
        PreemptionRegistration transferred;
        Throwable isolatedFailure;
        synchronized (entry) {
            FenceEndpointSettlement settlement =
                    settleFenceEndpointTerminalLocked(entry, proof);
            transferred = settlement.transferred();
            isolatedFailure = settlement.failure();
            ScheduledRequest item = entry.activeItem();
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
                return EngineFenceCoordinator.TerminalDisposition.STALE;
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
        return EngineFenceCoordinator.TerminalDisposition.TERMINALIZED;
    }

    private record FenceEndpointSettlement(
            PreemptionRegistration transferred,
            Throwable failure) {
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
    private static RequestState.Snapshot settleCancellationLifecycle(
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
    private PreemptionWork reduceDeferredTerminalFactLocked(
            RequestSlot entry,
            DeferredTerminal terminal) {
        ScheduledRequest item = entry.activeItem();
        if (item == null) {
            return PreemptionWork.STALE;
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
        return materializePreemptionWorkLocked(entry, reduction, null);
    }

    /** Apply an already-owned ordinary outcome. Called with {@code entry} locked. */
    private PublicationWork applyOrdinaryTerminalLocked(
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
            case WORKER ->
                    applyWorkerTerminalLocked(entry, terminal.observation());
            case PRIORITY ->
                    throw new IllegalStateException(
                            "priority terminal requires its typed reducer");
            case DECODE_GENERATION_RETIRED ->
                    throw new IllegalStateException(
                            "retired Decode generation was not marked retired");
        };
    }

    private PublicationWork applyTimeoutTerminalLocked(
            RequestSlot entry,
            DeferredTerminal timeout) {
        return PublicationWork.terminal(beginTerminalLocked(
                entry,
                true,
                true,
                owner -> owner.timeout(timeout.detail()),
                buildErrorResponse(
                        entry.timeoutErrorType(), timeout.detail())));
    }

    private PublicationWork applyFailureTerminalLocked(
            RequestSlot entry,
            DeferredTerminal failure,
            boolean releaseDecode) {
        return PublicationWork.terminal(beginTerminalLocked(
                entry, true, releaseDecode,
                owner -> owner.fail(failure.detail()),
                buildErrorResponse(failure.errorType(), failure.detail())));
    }

    private PublicationWork applyDecodeSettledTerminalLocked(
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
            return PublicationWork.terminal(beginTerminalLocked(
                    entry,
                    false,
                    true,
                    owner -> settleCancellationLifecycle(
                            owner, firstCause, cancellationDetail),
                    buildErrorResponse(
                            entry.cancellationErrorType(firstCause),
                            cancellationDetail)));
        }
        return PublicationWork.terminal(beginTerminalLocked(
                entry,
                false,
                true,
                owner -> owner.fail(detail),
                buildErrorResponse(
                        StrategyErrorType.BATCH_DISPATCH_FAILED, detail)));
    }

    /** Endpoint resources are already settled; only RequestSlot/response remain. */
    private PublicationWork applyWorkerTerminalLocked(
            RequestSlot entry,
            WorkerTerminalObservation observation) {
        if (entry.hasCancellationFirstCause()) {
            String proof = observation.source()
                    == WorkerTerminalSource.PREFILL_BACKED
                            ? "Prefill terminal observed after cancellation"
                            : "Decode terminal observed after cancellation";
            return PublicationWork.terminal(
                    settleCancellationFromWorkerStatusLocked(
                            entry, proof, observation.source()));
        }
        Function<RequestSlot, RequestState.Snapshot> transition;
        Response response;
        if (observation.successful()) {
            transition = owner -> owner.complete("decode completed");
            ScheduledRequest item = entry.activeItem();
            response = buildSuccessResponse(
                    item, entry.snapshot().deliveryClaimKind());
        } else {
            String detail = "worker error code " + observation.errorCode();
            transition = owner -> owner.fail(detail);
            response = buildErrorResponse(
                    StrategyErrorType.WORKER_EXECUTION_FAILED, detail);
        }
        return PublicationWork.terminal(beginWorkerStatusTerminalLocked(
                entry,
                workerStatusCounterpartCleanup(entry, observation.source()),
                transition,
                response));
    }

    public int getInflightSize() {
        return liveRequestCount();
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
     * Age (ms) of the oldest live request slot, 0 when the ledger is empty.
     * Single traversal mirroring {@link #liveRequestCount}: per-entry
     * {@code createdAtMs()} reads the lifecycle snapshot under the same
     * slot monitor the stale sweep uses, so a slot being reduced never
     * produces a torn read. Concrete-class method: the upstream lifecycle
     * port family no longer declares an age accessor, and
     * RequestMetricsOrchestrator depends on this class directly.
     */
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

    /**
     * Weakly-consistent immutable view of all scheduler-owned live request
     * lifecycles. The requestSlots map is authoritative; no diagnostic-only
     * shadow queue is maintained.
     */
    public List<RequestState.Snapshot> snapshotActiveRequests() {
        List<RequestState.Snapshot> snapshots = new ArrayList<>(requestSlots.size());
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

    public RequestState.Snapshot getRequestState(long requestId,
                                                    long expectedBatchId) {
        RequestSlot entry = requestSlots.get(requestId);
        RequestState.Snapshot snapshot = entry == null
                ? null : entry.snapshot();
        return batchMatches(snapshot, expectedBatchId) ? snapshot : null;
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
        PreemptionWork work = null;
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
                work = reduceDeferredTerminalFactLocked(
                        exactSlot, DeferredTerminal.timeout(detail));
            }
        }
        submitTerminal(direct);
        consumePreemptionWork(exactSlot, work);
        return direct != null
                || work != null && work.publication() != null;
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
            PreemptionWork work;
            synchronized (entry) {
                work = reduceDeferredTerminalFactLocked(entry,
                        DeferredTerminal.failure(
                                StrategyErrorType.BATCH_DISPATCH_FAILED,
                                "Worker scheduling queue rejected request: "
                                        + failureDetail));
            }
            consumePreemptionWork(entry, work);
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
        RequestState.Snapshot snapshot = entry.snapshot();
        return entry.ownsActiveItem(item)
                && entry.isOpen()
                && entry.canClaimDelivery()
                && snapshot.state() == RequestState.Phase.QUEUED
                && snapshot.deliveryClaimKind() == DeliveryClaimKind.NONE;
    }

    @Override
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

    @Override
    public SlotDeliveryPort.Claim tryClaimForDelivery(
            ScheduledRequest exactItem,
            SlotDeliveryPort.Identity identity,
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
            SlotDeliveryPort.Identity.ConfirmationBoundary boundary =
                    identity.boundary();
            long expectedCorrelationId = claimCorrelationId(identity);
            SlotDeliveryClaim claim = new SlotDeliveryClaim(
                    this, entry, item, identity);
            if (!endpointHandoff.getAsBoolean()) {
                throw new IllegalStateException(
                        "endpoint ownership was lost while the exact"
                                + " RequestSlot was locked request_id="
                                + item.requestId());
            }
            switch (boundary) {
                case EXTERNAL_ACK -> {
                    entry.startBatchEnqueue(expectedCorrelationId);
                    entry.markBatchEnqueueStarted();
                }
                case COMMIT_CONFIRMED ->
                        entry.startRouteDecisionDelivery();
            }
            return claim;
        }
    }

    private SlotDeliveryClaim exactClaim(SlotDeliveryPort.Claim claim) {
        return claim instanceof SlotDeliveryClaim exact
                        && exact.owner == this
                ? exact : null;
    }

    /** Caller holds {@code exact.slot}. */
    private boolean ownsDeliveryClaim(SlotDeliveryClaim exact) {
        return exact.slot.ownsDeliveryClaim(
                exact.item,
                claimKind(exact.identity),
                claimCorrelationId(exact.identity));
    }

    private static final class SlotDeliveryClaim
            implements SlotDeliveryPort.Claim {
        private final RequestRegistry owner;
        private final RequestSlot slot;
        private final ScheduledRequest item;
        private final SlotDeliveryPort.Identity identity;
        private boolean completed;

        private SlotDeliveryClaim(
                RequestRegistry owner,
                RequestSlot slot,
                ScheduledRequest item,
                SlotDeliveryPort.Identity identity) {
            this.owner = owner;
            this.slot = slot;
            this.item = item;
            this.identity = identity;
        }

        @Override
        public ScheduledRequest item() {
            return item;
        }
    }

    private static DeliveryClaimKind claimKind(
            SlotDeliveryPort.Identity identity) {
        return switch (identity.boundary()) {
            case EXTERNAL_ACK -> DeliveryClaimKind.BATCH_ENQUEUE;
            case COMMIT_CONFIRMED -> DeliveryClaimKind.ROUTE_DECISION;
        };
    }

    private static long claimCorrelationId(
            SlotDeliveryPort.Identity identity) {
        return switch (identity.boundary()) {
            case EXTERNAL_ACK -> identity.requiredCorrelationId();
            case COMMIT_CONFIRMED -> 0L;
        };
    }

    // ==================== SlotDeliveryPort outcomes ====================

    static final class PublicationWork {
        private final TerminalAction terminal;
        private final DeliveryPublication delivery;

        private PublicationWork(
                TerminalAction terminal,
                DeliveryPublication delivery) {
            if ((terminal == null) == (delivery == null)) {
                throw new IllegalArgumentException(
                        "publication work requires exactly one owner");
            }
            this.terminal = terminal;
            this.delivery = delivery;
        }

        static PublicationWork terminal(TerminalAction action) {
            return action == null ? null
                    : new PublicationWork(action, null);
        }

        static PublicationWork delivery(DeliveryPublication delivery) {
            return delivery == null ? null
                    : new PublicationWork(null, delivery);
        }
    }

    private record DeliveryPublication(
            RequestSlot slot,
            ScheduledRequest item,
            Response response,
            RequestSlot.DeliveryConfirmation confirmation,
            DeliveryClaimKind deliveryKind,
            long batchId) {
    }

    /** Called with {@code entry} locked. */
    private PreemptionWork confirmRouteDecisionLocked(
            RequestSlot entry,
            ScheduledRequest item) {
        if (!entry.ownsDeliveryClaim(
                item, DeliveryClaimKind.ROUTE_DECISION, 0L)) {
            return PreemptionWork.STALE;
        }
        return materializePreemptionWorkLocked(
                entry,
                entry.reduceDeliveryConfirmed(0L),
                null);
    }

    @Override
    public void complete(
            SlotDeliveryPort.Claim claim,
            SlotDeliveryPort.Completion completion) {
        SlotDeliveryClaim exact = exactClaim(claim);
        if (exact == null) {
            throw new IllegalArgumentException(
                    "delivery claim was not created by this scheduler");
        }
        PreemptionWork work = null;
        RequestSlot.FenceReduction fenceReduction = null;
        synchronized (exact.slot) {
            if (exact.completed || !ownsDeliveryClaim(exact)) {
                throw new IllegalStateException(
                        "delivery claim is stale or already completed: request_id="
                                + exact.item.requestId());
            }
            exact.completed = true;
            if (completion.status()
                    == SlotDeliveryPort.Completion.Status.DELIVERED) {
                work = switch (exact.identity.boundary()) {
                    case EXTERNAL_ACK -> confirmBatchEnqueueLocked(
                            exact.slot, exact.item);
                    case COMMIT_CONFIRMED -> confirmRouteDecisionLocked(
                            exact.slot, exact.item);
                };
            } else if (completion.status()
                    == SlotDeliveryPort.Completion.Status.FAILED) {
                String detail = "Delivery failed: "
                        + detailOf(completion.cause());
                if (exact.slot.decodeOwnsRequest()) {
                    work = materializePreemptionWorkLocked(
                            exact.slot,
                            exact.slot.reduceDeliveryConfirmed(
                                    claimCorrelationId(exact.identity)),
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
                        case ENGINE_ACCEPTED -> work = materializePreemptionWorkLocked(
                                exact.slot,
                                exact.slot.reduceDeliveryConfirmed(
                                        claimCorrelationId(exact.identity)),
                                null);
                        case CONFLICT -> fenceReduction =
                                exact.slot.requestDeliveryFence(detail);
                        case STALE -> work = PreemptionWork.STALE;
                    }
                }
            } else if (exact.slot.decodeOwnsRequest()) {
                work = materializePreemptionWorkLocked(
                        exact.slot,
                        exact.slot.reduceDeliveryConfirmed(
                                claimCorrelationId(exact.identity)),
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
        consumePreemptionWork(exact.slot, work);
        consumeFenceStart(exact.slot, fenceReduction, true);
    }

    /** Called with {@code entry} locked. */
    private PreemptionWork confirmBatchEnqueueLocked(
            RequestSlot entry,
            ScheduledRequest item) {
        RequestState.Snapshot current = entry.snapshot();
        long batchId = current.batchId();
        if (!entry.ownsDeliveryClaim(
                item, DeliveryClaimKind.BATCH_ENQUEUE, batchId)) {
            Logger.debug("Ignoring EnqueueBatch ACK without a batch claim request_id={}",
                    item.requestId());
            return PreemptionWork.STALE;
        }
        return materializePreemptionWorkLocked(
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
    private PublicationWork deliveryPublication(
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
        return PublicationWork.delivery(new DeliveryPublication(
                entry, item, response, confirmation,
                deliveryKind, batchId));
    }

    /** Claim the canonical slot and move every local cleanup capability once. */
    static TerminalAction beginTerminalLocked(
            RequestSlot entry,
            boolean releaseDecode,
            boolean releasePrefill,
            Function<RequestSlot, RequestState.Snapshot> transition,
            Response response) {
        return beginTerminalLocked(
                entry, true, releaseDecode, releasePrefill, null,
                transition, response);
    }

    /** Endpoint status has already settled source queue/ledger ownership. */
    static TerminalAction beginWorkerStatusTerminalLocked(
            RequestSlot entry,
            Runnable counterpartCleanup,
            Function<RequestSlot, RequestState.Snapshot> transition,
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
            Function<RequestSlot, RequestState.Snapshot> transition,
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
                action.publicationLease() == null
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

    void submitPublication(PublicationWork publication) {
        if (publication == null) {
            return;
        }
        if (publication.terminal != null) {
            submitTerminal(publication.terminal);
        } else {
            publishDelivery(publication.delivery);
        }
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

    private void publishDelivery(DeliveryPublication publication) {
        if (publication == null) {
            return;
        }
        RequestSlot.DeliveryConfirmation confirmation =
                publication.confirmation();
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
                        ? () -> armAcceptanceDeadline(publication.slot())
                        : null);
        if (publication.deliveryKind()
                == DeliveryClaimKind.BATCH_ENQUEUE
                && confirmation.batchEnqueueStartedAtMs() > 0L) {
            long latencyMs = Math.max(
                    0L,
                    System.currentTimeMillis()
                            - confirmation.batchEnqueueStartedAtMs());
            preparationFailure = runTerminalLeaf(
                    preparationFailure,
                    () -> reporter.reportDispatchAckTimeMs(
                            RoleType.PREFILL.name(),
                            publication.item().prefillEp() == null
                                    ? ""
                                    : publication.item().prefillEp().getIp(),
                            latencyMs));
        }
        if (preparationFailure != null) {
            Logger.error(
                    "Delivery publication preparation isolated: request_id={}",
                    publication.item().requestId(),
                    preparationFailure);
        }
        completionPublisher.submitDeliveryResponse(
                confirmation.publication(), publication.response());
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

    @Override
    public void failPrepared(ScheduledRequest exactItem, Throwable cause) {
        ScheduledRequest item = exactItem;
        RequestSlot entry = entryFor(item);
        if (entry == null) {
            return;
        }
        PreemptionWork work = null;
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
            consumePreemptionWork(entry, work);
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
            decodeEp.releasePlacementExact(reservation);
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
        Function<RequestSlot, RequestState.Snapshot> transition =
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
            Function<RequestSlot, RequestState.Snapshot> transition) {
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

    static Response buildAdmissionErrorResponse(AdmissionFailure failure,
                                                String trigger) {
        Response errorResp = Response.error(failure.errorType(), failure.reason());
        String detail = failure.message() + "; trigger=" + trigger;
        errorResp.setErrorMessage(failure.errorType().buildErrorMessage(detail));
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

    private static boolean batchMatches(RequestState.Snapshot snapshot,
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
