package org.flexlb.balance.endpoint;

import org.flexlb.balance.endpoint.DecodeResources.Acquisition;
import org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity;
import org.flexlb.balance.endpoint.DecodeResources.CapacityRelease;
import org.flexlb.balance.endpoint.DecodeResources.CapacityUsage;
import org.flexlb.balance.endpoint.DecodeResources.DecodeRequestStatus;
import org.flexlb.balance.endpoint.DecodeResources.DecodeRequestView;
import org.flexlb.balance.endpoint.DecodeResources.DecodeRoutingView;
import org.flexlb.balance.endpoint.DecodeResources.DispatchOutcome;
import org.flexlb.balance.endpoint.DecodeResources.EngineDispatchPermitAcquireStatus;
import org.flexlb.balance.endpoint.DecodeResources.EngineDispatchPermitTransferStatus;
import org.flexlb.balance.endpoint.DecodeResources.MutationStatus;
import org.flexlb.balance.endpoint.DecodeResources.Ownership;
import org.flexlb.balance.endpoint.DecodeResources.PreemptionBeginResult;
import org.flexlb.balance.endpoint.DecodeResources.PreemptionUpdate;
import org.flexlb.balance.endpoint.DecodeResources.ReleaseReason;
import org.flexlb.balance.endpoint.DecodeResources.RequestReservation;
import org.flexlb.balance.endpoint.DecodeResources.ReservationReleaseResult;
import org.flexlb.balance.endpoint.DecodeResources.ReserveStatus;
import org.flexlb.balance.endpoint.DecodeResources.ResourceChange;
import org.flexlb.balance.endpoint.DecodeResources.ResourceSnapshot;
import org.flexlb.balance.scheduler.ContextResult;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.balance.scheduler.RequestRepository;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.DecodeTaskPhase;
import org.flexlb.enums.TaskPhase;
import org.flexlb.util.PriorityNormalizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.OptionalLong;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Predicate;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;
import static org.flexlb.balance.endpoint.DecodeResources.CapacityRelease.saturatedAddNonNegative;

/** Decode worker boundary: lifecycle selectedWorkers, resource operations and lock-free notifications.
 * This endpoint owns its resource records and their single mutation lock.
 */
public class DecodeEndpoint extends WorkerEndpoint implements EndpointStats.Decode {
    private static final Logger logger = LoggerFactory.getLogger("syncLogger");
    private final RequestRepository requests;
    private final PlacementAvailability placementAvailability;
    private final Set<Runnable> engineDispatchCapacityListeners = ConcurrentHashMap.newKeySet();

    private static final Comparator<RequestReservation> RETIREMENT_ORDER =
            Comparator.comparingLong(RequestReservation::endpointGenerationId)
                    .thenComparing(RequestReservation::requestId)
                    .thenComparingLong(RequestReservation::reservationToken);
    private final ConcurrentHashMap<String, RequestRecord> decodeRequests =
            new ConcurrentHashMap<>();
    private final Map<String, Long> settledHistory = new HashMap<>();
    private final Map<String, PreemptionClaim> victimClaims = new HashMap<>();
    private final Map<Long, PreemptionRecord> preemptionAttempts = new HashMap<>();
    private final ResourceUsage reservedUsage = new ResourceUsage();
    private final ResourceUsage queuedUsage = new ResourceUsage();
    private final ResourceUsage dispatchUsage = new ResourceUsage();
    private volatile int confirmedEngineOwnedCount;
    private volatile long priorityPreemptionHeldKv;
    private volatile long priorityPreemptionHeldExpectedKv;
    private volatile long admissionVersion;
    private volatile DecodeRoutingView routingViewCache;
    private volatile DecodeResources.AdmissionSummary admissionSummaryCache;
    private long nextReservationToken = 1L;

    /**
     * Changed in Endpoint lock; a full observation publishes one revision after all rows settle.
     */
    private boolean calibrating;

    @Override
    public void publishStatus(StatusChange change) {
        if (change instanceof StatusChange.None) {
            return;
        }
        var effects = (StatusChange.Decode) change;
        if (effects.dispatchCapacityChanged()) {
            publishCapacity(CapacityEdge.DISPATCH);
        }
        if (effects.placementCapacityChanged()) {
            publishCapacity(CapacityEdge.PLACEMENT);
        }
        for (var observed : effects.requests()) {
            var kind =
                    observed.kind() == DecodeRequestStatus.Kind.TERMINAL
                            ? observed.errorCode() == 0L
                                    ? ContextResult.WorkerRequestEventType.COMPLETED
                                    : ContextResult.WorkerRequestEventType.FAILED
                            : observed.allocationObserved()
                                    ? ContextResult.WorkerRequestEventType.ALLOCATED
                                    : ContextResult.WorkerRequestEventType.ACTIVE;
            publishRequest(observed.reservation(), kind, observed.errorCode(), "");
        }
    }

    private void publishRequest(
            RequestReservation reservation,
            ContextResult.WorkerRequestEventType kind,
            long errorCode,
            String detail) {
        var context = requests.findActive(reservation.requestId());
        if (context == null) return;
        var route = context.snapshot().route();
        if (route == null || route.decodeEndpoint() != this) return;
        context.data()
                .getRequestScheduler()
                .onWorkerRequestEvent(
                        context,
                        new ContextResult.WorkerRequestEvent(
                                route,
                                reservation,
                                RoleType.DECODE,
                                kind,
                                errorCode,
                                detail,
                                System.currentTimeMillis()));
    }

    DecodeEndpoint(
            WorkerStatus status,
            RequestRepository requests,
            PlacementAvailability placementAvailability) {
        super(status);
        this.requests = java.util.Objects.requireNonNull(requests, "requests");
        this.placementAvailability =
                java.util.Objects.requireNonNull(placementAvailability, "placementAvailability");
    }

    /**
     * LOCAL_ROLLBACK requires local ownership; other evidence may leave Engine/protocol ownership intact.
     */
    public ResourceChange<ReservationReleaseResult> release(
            RequestReservation reservation, ReleaseReason reason) {
        ResourceChange<ReservationReleaseResult> result = releaseReservation(reservation, reason);
        if (result.capacityReleased()) {
            publishCapacity(CapacityEdge.BOTH);
        }
        return result;
    }

    /**
     * Resolve the exact dispatch permit and its request record in one event.
     */
    public ResourceChange<EngineDispatchPermitTransferStatus> dispatch(
            EngineDispatchPermit permit, DispatchOutcome outcome) {
        java.util.Objects.requireNonNull(outcome, "outcome");
        checkArgument(permit.endpoint == this, "Dispatch permit belongs to another endpoint");
        stateLock().lock();
        try {
            if (permit.dispatchResult != null)
                return ResourceChange.of(
                        outcome == DispatchOutcome.ENGINE_OWNED
                                ? permit.dispatchResult
                                : EngineDispatchPermitTransferStatus.OWNERSHIP_LOST,
                        false);
            if (outcome == DispatchOutcome.ENGINE_OWNED && !acceptsNewRequests()) {
                permit.dispatchResult = EngineDispatchPermitTransferStatus.ENDPOINT_RETIRED;
                return ResourceChange.of(permit.dispatchResult, false);
            }
            RequestRecord current = decodeRequests.get(permit.reservation.requestId);
            boolean sending = outcome == DispatchOutcome.ENGINE_OWNED;
            boolean alreadyOwned =
                    current == permit.reservation
                            && (current.confirmed()
                                    || current.phase == DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN)
                            && !victimClaims.containsKey(current.requestId);
            boolean owns = current == permit.reservation && current.dispatchPermit == permit;
            boolean accepted =
                    sending
                            ? alreadyOwned
                                    || owns
                                            && current.queued()
                                            && !victimClaims.containsKey(current.requestId)
                            : owns || permit.retiredByEndpoint;
            if (owns && !alreadyOwned)
                transition(
                        current,
                        sending && accepted ? DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN : current.phase,
                        null,
                        0L,
                        0L);
            var status =
                    accepted
                            ? EngineDispatchPermitTransferStatus.TRANSFERRED
                            : EngineDispatchPermitTransferStatus.OWNERSHIP_LOST;
            permit.dispatchResult =
                    sending ? status : EngineDispatchPermitTransferStatus.OWNERSHIP_LOST;
            // Sending an accepted queued permit replaces its temporary slot with
            // Engine
            // occupancy. Returning or rejecting an owned permit frees that
            // temporary slot.
            return ResourceChange.of(status, owns && (!sending || !accepted));
        } finally {
            stateLock().unlock();
        }
    }

    public static final class EngineDispatchPermit {

        private final DecodeEndpoint endpoint;
        final RequestRecord reservation;
        boolean retiredByEndpoint;

        /**
         * Null until resolved; cached result for dispatch, including ownership lost after release.
         */
        private EngineDispatchPermitTransferStatus dispatchResult;

        EngineDispatchPermit(DecodeEndpoint endpoint, RequestRecord reservation) {
            this.endpoint = endpoint;
            this.reservation = reservation;
        }

        /**
         * Identity check uses immutable reservation fields; it does not wait for an Endpoint lock.
         */
        public boolean belongsTo(DecodeEndpoint expected, RequestReservation reservation) {
            return endpoint == expected
                    && reservation != null
                    && endpoint.getStatus().getGenerationId() == reservation.endpointGenerationId()
                    && this.reservation.matches(reservation);
        }

        /**
         * Transfer this acquired slot without a second capacity check.
         *
         * @return a typed distinction between transfer, prior request-owner
         *         loss, and endpoint-generation retirement
         */
        public ResourceChange<EngineDispatchPermitTransferStatus> dispatch() {
            return endpoint.dispatch(this, DispatchOutcome.ENGINE_OWNED);
        }

        public boolean release() {
            var result = endpoint.dispatch(this, DispatchOutcome.ABANDONED);
            if (result.capacityReleased()) {
                endpoint.publishCapacity(CapacityEdge.BOTH);
            }
            return result.status() == EngineDispatchPermitTransferStatus.TRANSFERRED;
        }
    }

    /**
     * The caller owns this subscription and closes it when its capacity wait ends.
     */
    public Runnable watchEngineDispatchCapacity(Runnable listener) {
        java.util.Objects.requireNonNull(listener, "listener");
        engineDispatchCapacityListeners.add(listener);
        return () -> engineDispatchCapacityListeners.remove(listener);
    }

    public RequestReservation commitPreemption(long attemptToken) {
        return finishPreemption(attemptToken, true);
    }

    public boolean abortPreemption(long attemptToken) {
        boolean aborted = finishPreemption(attemptToken, false) != null;
        if (aborted) {
            publishCapacity(CapacityEdge.BOTH);
        }
        return aborted;
    }

    // Calibration: publish facts only after the resource transaction returns.

    public StatusChange applyPreparedStatus(WorkerStatus ws, WorkerStatus.PreparedStatus prepared) {
        requireStatusGeneration(ws);
        stateLock().lock();
        try {
            checkState(!isRetired(), "Decode endpoint is retired");
            try {
                if (!prepared.observation().alive()) beginRetirement();
                checkArgument(
                        prepared.observation().owner() == ws,
                        "Status belongs to another Decode generation");
                DecodeRoutingView before = routingView();
                var requestStatuses = calibrateLocked(prepared.observation());
                ws.publishReducedStatus(prepared);
                DecodeRoutingView after = routingView();
                boolean capacityImproved =
                        after.engineLoad() < before.engineLoad()
                                || after.realKvAvailable() > before.realKvAvailable()
                                || after.realKvUsed() < before.realKvUsed();
                return new StatusChange.Decode(requestStatuses, capacityImproved, true);
            } catch (RuntimeException | Error failure) {
                beginRetirement();
                throw failure;
            }
        } finally {
            stateLock().unlock();
        }
    }

    public StatusChange applyStatusHeartbeat(
            WorkerStatus ws, WorkerStatus.StatusObservation observation) {
        requireStatusGeneration(ws);
        List<DecodeRequestStatus> requestStatuses = collectHeartbeatRequestStatuses(observation);
        return new StatusChange.Decode(requestStatuses, false, false);
    }

    // Retirement and orphan cleanup.

    protected void closeEndpoint() {
        List<RequestReservation> reservations = retire();
        try {
            for (var reservation : reservations) {
                var context = requests.findActive(reservation.requestId());
                if (context == null) continue;
                context.data()
                        .getRequestScheduler()
                        .onEndpointRetired(
                                context,
                                new ContextResult.EndpointRetirement(
                                        context.snapshot().route(),
                                        reservation,
                                        RoleType.DECODE,
                                        this,
                                        "Decode generation retired: "
                                                + reservation.endpointGenerationId(),
                                        System.currentTimeMillis()));
            }
        } finally {
            publishCapacity(CapacityEdge.DISPATCH);
        }
    }

    public int evictExpiredRequests(long ttlMs, Predicate<String> retainForSchedulerCleanup) {
        var orphanCandidates = cleanupCandidates();
        orphanCandidates.keySet().removeIf(retainForSchedulerCleanup::test);
        CleanupResult result = expireReservations(ttlMs, orphanCandidates);
        if (result.capacityReleased()) {
            publishCapacity(CapacityEdge.BOTH);
        }
        return result.expiredReservations();
    }

    public enum CapacityEdge {
        DISPATCH,
        PLACEMENT,
        BOTH
    }

    public void publishCapacity(CapacityEdge edge) {
        if (edge != CapacityEdge.PLACEMENT) {
            for (Runnable listener : engineDispatchCapacityListeners) {
                try {
                    listener.run();
                } catch (Throwable failure) {
                    logger.warn("Decode capacity listener failed", failure);
                }
            }
        }
        if (edge != CapacityEdge.DISPATCH) {
            placementAvailability.changed(this);
        }
    }

    public Acquisition<ReserveStatus, RequestReservation> tryReserveQueuedRequest(
            String requestId,
            long hardKv,
            long expectedKv,
            int priority,
            AdmissionCapacity capacity) {
        stateLock().lock();
        try {
            if (!acceptsNewRequests())
                return Acquisition.unavailable(ReserveStatus.ENDPOINT_RETIRED);
            if (!requestIdAvailable(requestId)) {
                return Acquisition.unavailable(ReserveStatus.ALREADY_OWNED);
            }
            if (capacity != null
                    && !capacity.evaluate(
                                    routingView().placementUsage(),
                                    hardKv,
                                    expectedKv,
                                    CapacityRelease.NONE)
                            .fits()) {
                return Acquisition.unavailable(ReserveStatus.CAPACITY_FULL);
            }
            return createReservation(
                    requestId,
                    hardKv,
                    expectedKv,
                    priority,
                    DecodeTaskPhase.MASTER_QUEUED_NOT_DISPATCHED);
        } finally {
            stateLock().unlock();
        }
    }

    private Acquisition<ReserveStatus, RequestReservation> createReservation(
            String requestId, long hardKv, long expectedKv, int priority, DecodeTaskPhase phase) {
        checkState(
                nextReservationToken > 0L && nextReservationToken != Long.MAX_VALUE,
                "Decode reservation token space exhausted");
        RequestRecord request =
                new RequestRecord(requestId, hardKv, expectedKv, priority, nextReservationToken++);
        var acquired = new Acquisition<>(ReserveStatus.ACQUIRED, request.reservation());
        decodeRequests.put(requestId, request);
        reservedUsage.adjust(1, hardKv, expectedKv);
        if (phase == DecodeTaskPhase.MASTER_QUEUED_NOT_DISPATCHED) {
            queuedUsage.adjust(1, hardKv, expectedKv);
        }
        request.phase = phase;
        admissionVersion++;
        return acquired;
    }

    private boolean requestIdAvailable(String requestId) {
        // Every installed incoming attempt retains its exact request row until commit/abort/expiry.
        return !decodeRequests.containsKey(requestId)
                && !victimClaims.containsKey(requestId)
                && !settledHistory.containsKey(requestId);
    }

    /**
     * Exact resource obligation; absence also covers retired or replaced reservation identities.
     */
    public Ownership ownership(RequestReservation exact) {
        if (exact == null || exact.endpointGenerationId() != getStatus().getGenerationId()) {
            return Ownership.NONE;
        }
        stateLock().lock();
        try {
            RequestRecord request = decodeRequests.get(exact.requestId());
            if (request == null || !request.matches(exact)) {
                PreemptionClaim claim = victimClaims.get(exact.requestId());
                return claim != null && claim.reservation.equals(exact)
                        ? Ownership.HELD
                        : Ownership.NONE;
            }
            if (request.confirmed()) {
                return Ownership.CONFIRMED;
            }
            return request.queued()
                            && request.dispatchPermit == null
                            && request.incomingAttemptToken == 0L
                            && !victimClaims.containsKey(exact.requestId())
                    ? Ownership.WITHDRAWABLE
                    : Ownership.HELD;
        } finally {
            stateLock().unlock();
        }
    }

    ResourceChange<ReservationReleaseResult> releaseReservation(
            RequestReservation exact, ReleaseReason reason) {
        java.util.Objects.requireNonNull(reason, "reason");
        if (exact == null) {
            checkArgument(
                    reason != ReleaseReason.LOCAL_ROLLBACK
                            && reason != ReleaseReason.NOT_SENT
                            && reason != ReleaseReason.UNSENT_ROUTE_ROLLBACK,
                    "Decode reservation is required for release");
            return ResourceChange.of(ReservationReleaseResult.STALE, false);
        }
        if (exact.endpointGenerationId() != getStatus().getGenerationId()) {
            return ResourceChange.of(ReservationReleaseResult.STALE, false);
        }
        stateLock().lock();
        try {
            if (reason == ReleaseReason.EXPIRED || reason == ReleaseReason.REMOTE_CLEANUP) {
                boolean released = settle(exact, Settlement.EXPIRED);
                return ResourceChange.of(
                        released
                                ? ReservationReleaseResult.RELEASED
                                : ReservationReleaseResult.STALE,
                        released);
            }
            RequestRecord request = decodeRequests.get(exact.requestId());
            boolean matches = request != null && request.matches(exact);
            if (reason == ReleaseReason.NOT_SENT || reason == ReleaseReason.UNSENT_ROUTE_ROLLBACK) {
                if (matches && request.confirmed()) {
                    return ResourceChange.of(ReservationReleaseResult.ENGINE_ACCEPTED, false);
                }
                if (matches
                        && !victimClaims.containsKey(exact.requestId())
                        && request.incomingAttemptToken != 0L) {
                    return ResourceChange.of(ReservationReleaseResult.CONFLICT, false);
                }
            } else if (matches
                    && (request.incomingAttemptToken != 0L
                            || request.confirmed()
                            || request.phase == DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN
                            || victimClaims.containsKey(exact.requestId()))) {
                checkState(
                        reason != ReleaseReason.LOCAL_ROLLBACK,
                        "Illegal Decode local release: exact ownership is held by Engine/protocol lifecycle: %s",
                        exact);
                return ResourceChange.of(ReservationReleaseResult.STILL_OWNED, false);
            }
            if (!matches) {
                return ResourceChange.of(ReservationReleaseResult.STALE, false);
            }
            boolean released =
                    settle(
                            exact,
                            reason == ReleaseReason.LOCAL_ROLLBACK
                                            || reason == ReleaseReason.UNSENT_ROUTE_ROLLBACK
                                    ? Settlement.ROLLBACK
                                    : Settlement.TERMINAL);
            return ResourceChange.of(
                    released ? ReservationReleaseResult.RELEASED : ReservationReleaseResult.STALE,
                    released);
        } finally {
            stateLock().unlock();
        }
    }

    private enum Settlement {
        ROLLBACK,
        TERMINAL,
        OBSERVED_TERMINAL,
        EXPIRED
    }

    /**
     * Only validated evidence reaches this kernel; request and claim obligations settle together.
     */
    private boolean settle(RequestReservation exact, Settlement proof) {
        RequestRecord request = decodeRequests.get(exact.requestId());
        boolean matches = request != null && request.matches(exact);
        PreemptionClaim claim = victimClaims.get(exact.requestId());
        if (claim != null && !claim.reservation.equals(exact)) {
            claim = null;
        }
        if (!matches && claim == null) {
            return false;
        }
        long version = admissionVersion;
        if (matches && request.incomingAttemptToken != 0L) {
            checkState(
                    proof == Settlement.EXPIRED,
                    "Decode terminal still owns its incoming attempt: %s",
                    exact);
            finishPreemption(request.incomingAttemptToken, false);
        }
        if (claim != null) {
            PreemptionRecord attempt = preemptionAttempts.get(claim.attemptToken);
            if (attempt != null) {
                attempt.remainingVictims.remove(exact);
            }
            claim.close();
        }
        if (matches && decodeRequests.remove(exact.requestId(), request)) {
            transition(request, null, null, 0L, 0L);
        }
        // Ordinary Engine completion may be rediscovered; cancellation claims retain their explicit
        // fence.
        if (proof != Settlement.ROLLBACK
                && (proof != Settlement.OBSERVED_TERMINAL || claim != null)) {
            settledHistory.put(exact.requestId(), System.currentTimeMillis());
        }
        if (!calibrating && admissionVersion == version) {
            admissionVersion++;
        }
        return true;
    }

    /**
     * Update derived counters once from old/new contributions, without allocating contribution objects.
     */
    private void transition(
            RequestRecord request,
            DecodeTaskPhase next,
            EngineDispatchPermit permit,
            long engineKv,
            long observedAtMs) {
        boolean reserved = !request.confirmed();
        boolean queued = request.queued();
        boolean dispatching = request.dispatchPermit != null;
        boolean nextReserved = next != null && !next.isEngineConfirmed();
        boolean nextQueued = next == DecodeTaskPhase.MASTER_QUEUED_NOT_DISPATCHED;
        boolean nextDispatching = permit != null;
        long hard = request.kvTokens, expected = request.expectedKvTokens;
        reservedUsage.adjust(
                (nextReserved ? 1 : 0) - (reserved ? 1 : 0),
                (nextReserved ? hard : 0L) - (reserved ? hard : 0L),
                (nextReserved ? expected : 0L) - (reserved ? expected : 0L));
        queuedUsage.adjust(
                (nextQueued ? 1 : 0) - (queued ? 1 : 0),
                (nextQueued ? hard : 0L) - (queued ? hard : 0L),
                (nextQueued ? expected : 0L) - (queued ? expected : 0L));
        dispatchUsage.adjust(
                (nextDispatching ? 1 : 0) - (dispatching ? 1 : 0),
                (nextDispatching ? hard : 0L) - (dispatching ? hard : 0L),
                (nextDispatching ? expected : 0L) - (dispatching ? expected : 0L));
        confirmedEngineOwnedCount +=
                (next != null && next.isEngineConfirmed() ? 1 : 0) - (reserved ? 0 : 1);
        if (reserved && next != null && next.isEngineConfirmed()) {
            request.kvTokens = Math.max(0L, engineKv);
            request.expectedKvTokens = request.kvTokens;
        }
        request.dispatchPermit = permit;
        if (observedAtMs > 0L) {
            request.observedAtMs = observedAtMs;
        }
        // A removed row retains its last phase for advisory readers that already captured it.
        if (next != null) {
            request.phase = next;
        }
        if (!calibrating) {
            admissionVersion++;
        }
    }

    public ResourceChange<MutationStatus> markQueued(RequestReservation exact) {
        checkArgument(exact != null, "Decode reservation is required for queued transition");
        stateLock().lock();
        try {
            if (!acceptsNewRequests()) return ResourceChange.of(MutationStatus.STALE, false);
            RequestRecord request = decodeRequests.get(exact.requestId());
            if (request == null || !request.matches(exact) || request.confirmed()) {
                return ResourceChange.of(MutationStatus.STALE, false);
            }
            if (request.queued()) {
                return ResourceChange.of(MutationStatus.APPLIED, false);
            }
            if (request.phase == DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN) {
                return ResourceChange.of(MutationStatus.STALE, false);
            }
            transition(request, DecodeTaskPhase.MASTER_QUEUED_NOT_DISPATCHED, null, 0L, 0L);
        } finally {
            stateLock().unlock();
        }
        publishCapacity(CapacityEdge.BOTH);
        return ResourceChange.of(MutationStatus.APPLIED, true);
    }

    public Acquisition<EngineDispatchPermitAcquireStatus, EngineDispatchPermit>
            acquireDispatchPermit(RequestReservation exact, AdmissionCapacity capacity) {
        java.util.Objects.requireNonNull(exact, "reservation");
        java.util.Objects.requireNonNull(capacity, "capacity");

        stateLock().lock();
        try {
            if (!acceptsNewRequests())
                return Acquisition.unavailable(EngineDispatchPermitAcquireStatus.ENDPOINT_RETIRED);
            RequestRecord request = decodeRequests.get(exact.requestId());
            if (request == null
                    || !request.matches(exact)
                    || victimClaims.containsKey(exact.requestId())) {
                return Acquisition.unavailable(EngineDispatchPermitAcquireStatus.NOT_OWNED);
            }
            if (request.confirmed() || request.phase == DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN) {
                return new Acquisition<>(
                        EngineDispatchPermitAcquireStatus.ALREADY_ACCEPTED,
                        new EngineDispatchPermit(this, request));
            }
            if (!request.queued()) {
                return Acquisition.unavailable(EngineDispatchPermitAcquireStatus.NOT_QUEUED);
            }
            if (request.dispatchPermit != null) {
                return Acquisition.unavailable(EngineDispatchPermitAcquireStatus.ALREADY_ACQUIRED);
            }
            if (!capacity.evaluate(
                            usage(true, getStatus().committedWorkerStatus().fields()),
                            request.kvTokens,
                            request.expectedKvTokens,
                            CapacityRelease.NONE)
                    .fits()) {
                return Acquisition.unavailable(EngineDispatchPermitAcquireStatus.CAPACITY_FULL);
            }
            EngineDispatchPermit permit = new EngineDispatchPermit(this, request);
            transition(request, request.phase, permit, 0L, 0L);
            return new Acquisition<>(EngineDispatchPermitAcquireStatus.ACQUIRED, permit);
        } finally {
            stateLock().unlock();
        }
    }

    public boolean shouldRetryDispatch(RequestReservation exact, AdmissionCapacity capacity) {
        stateLock().lock();
        try {
            RequestRecord request = decodeRequests.get(exact.requestId());
            return !acceptsNewRequests()
                    || exact.endpointGenerationId() != getStatus().getGenerationId()
                    || request == null
                    || !request.matches(exact)
                    || !request.queued()
                    || request.dispatchPermit != null
                    || capacity.evaluate(
                                    usage(true, getStatus().committedWorkerStatus().fields()),
                                    request.kvTokens,
                                    request.expectedKvTokens,
                                    CapacityRelease.NONE)
                            .fits();
        } finally {
            stateLock().unlock();
        }
    }

    private CapacityUsage usage(boolean dispatch, WorkerStatus.EngineObservation fields) {
        long hard =
                dispatch
                        ? Math.max(0L, reservedUsage.hardKv - queuedUsage.hardKv)
                                + dispatchUsage.hardKv
                        : reservedUsage.hardKv;
        long expected =
                dispatch
                        ? Math.max(0L, reservedUsage.expectedKv - queuedUsage.expectedKv)
                                + dispatchUsage.expectedKv
                        : reservedUsage.expectedKv;
        long reported =
                fields.totalKvCacheTokens() > 0L
                        ? Math.max(
                                0L, fields.totalKvCacheTokens() - fields.availableKvCacheTokens())
                        : 0L;
        int count = confirmedEngineOwnedCount + reservedUsage.requests;
        if (dispatch) {
            count +=
                    Math.max(0, dispatchUsage.requests)
                            - Math.clamp(
                                    queuedUsage.requests, 0, Math.max(0, reservedUsage.requests));
        }
        return new CapacityUsage(
                count,
                Math.max(0L, fields.totalKvCacheTokens()),
                Math.max(0L, fields.availableKvCacheTokens()),
                saturatedAddNonNegative(hard, priorityPreemptionHeldKv),
                saturatedAddNonNegative(
                        saturatedAddNonNegative(reported, expected),
                        priorityPreemptionHeldExpectedKv));
    }

    public RequestReservation replaceQueuedRequests(
            List<RequestReservation> victims,
            String incomingId,
            long hardKv,
            long expectedKv,
            int priority,
            AdmissionCapacity capacity) {
        RequestReservation replaced;
        stateLock().lock();
        try {
            if (!acceptsNewRequests()) return null;
            if (victims == null || victims.isEmpty() || !requestIdAvailable(incomingId)) {
                return null;
            }
            Set<String> unique = new HashSet<>(victims.size());
            CapacityRelease released = CapacityRelease.NONE;
            for (RequestReservation victim : victims) {
                if (victim == null
                        || java.util.Objects.equals(victim.requestId(), incomingId)
                        || !unique.add(victim.requestId())
                        || ownership(victim) != Ownership.WITHDRAWABLE) {
                    return null;
                }
                released = released.plus(decodeRequests.get(victim.requestId()).capacityRelease());
            }
            CapacityUsage usage = routingView().placementUsage();
            if (capacity.evaluate(usage, hardKv, expectedKv, CapacityRelease.NONE).fits()
                    || !capacity.evaluate(usage, hardKv, expectedKv, released).fits()) {
                return null;
            }
            for (RequestReservation victim : victims) {
                settle(victim, Settlement.ROLLBACK);
            }
            replaced =
                    createReservation(
                                    incomingId,
                                    hardKv,
                                    expectedKv,
                                    priority,
                                    DecodeTaskPhase.LOCAL_RESERVED)
                            .value();
        } finally {
            stateLock().unlock();
        }
        if (replaced != null) publishCapacity(CapacityEdge.BOTH);
        return replaced;
    }

    public PreemptionBeginResult beginPreemption(
            long attemptToken,
            List<RequestReservation> victims,
            String incomingRequestId,
            long incomingKvTokens,
            long incomingExpectedKvTokens,
            int incomingPriority,
            AdmissionCapacity capacity) {
        checkArgument(
                attemptToken > 0 && victims != null && !victims.isEmpty(),
                "attempt token and victims are required");

        stateLock().lock();
        try {
            if (!acceptsNewRequests()) return PreemptionBeginResult.ENDPOINT_RETIRED;
            if (preemptionAttempts.containsKey(attemptToken)) {
                return PreemptionBeginResult.ATTEMPT_ALREADY_EXISTS;
            }
            if (!requestIdAvailable(incomingRequestId)) {
                return PreemptionBeginResult.INCOMING_ALREADY_RESERVED;
            }

            Set<RequestReservation> exactVictims = new HashSet<>();
            CapacityRelease released = CapacityRelease.NONE;
            for (RequestReservation victim : victims) {
                if (victim == null
                        || victim.endpointGenerationId() != getStatus().getGenerationId()
                        || java.util.Objects.equals(victim.requestId(), incomingRequestId)
                        || !exactVictims.add(victim)) {
                    return PreemptionBeginResult.VICTIM_GONE;
                }
                String victimId = victim.requestId();
                RequestRecord request = decodeRequests.get(victimId);
                if (victimClaims.containsKey(victimId)
                        || request != null && request.incomingAttemptToken != 0L) {
                    return PreemptionBeginResult.VICTIM_ALREADY_CLAIMED;
                }
                if (request == null || !request.matches(victim)) {
                    return PreemptionBeginResult.VICTIM_GONE;
                }
                if (request.dispatchPermit != null || request.queued()) {
                    return PreemptionBeginResult.VICTIM_GONE;
                }
                if (request.priority <= 0 || request.priority >= incomingPriority) {
                    return PreemptionBeginResult.INVALID_PRIORITY;
                }
                released = released.plus(request.capacityRelease());
            }
            CapacityUsage usage = routingView().placementUsage();
            if (capacity.evaluate(
                                    usage,
                                    incomingKvTokens,
                                    incomingExpectedKvTokens,
                                    CapacityRelease.NONE)
                            .fits()
                    || !capacity.evaluate(
                                    usage, incomingKvTokens, incomingExpectedKvTokens, released)
                            .fits()) {
                return PreemptionBeginResult.INFEASIBLE;
            }

            // Allocate every victim claim before installing any incoming or
            // protocol ownership. The endpoint lock keeps these exact states
            // stable through the subsequent allocation-free installation.
            Map<String, PreemptionClaim> preparedClaims = new HashMap<>();
            for (RequestReservation victim : victims) {
                RequestRecord request = decodeRequests.get(victim.requestId());
                preparedClaims.put(
                        victim.requestId(),
                        new PreemptionClaim(
                                attemptToken, victim, request.kvTokens, request.expectedKvTokens));
            }

            // Provisional incoming ownership closes the free-pool race while
            // Cancel runs.  It is not visible to the prefill queue yet.
            RequestReservation incomingReservation =
                    createReservation(
                                    incomingRequestId,
                                    incomingKvTokens,
                                    incomingExpectedKvTokens,
                                    incomingPriority,
                                    DecodeTaskPhase.LOCAL_RESERVED)
                            .value();
            PreemptionRecord preparedAttempt = null;
            try {
                preparedAttempt = new PreemptionRecord(incomingReservation, exactVictims);
                preemptionAttempts.put(attemptToken, preparedAttempt);
                decodeRequests.get(incomingRequestId).incomingAttemptToken = attemptToken;
                for (Map.Entry<String, PreemptionClaim> claim : preparedClaims.entrySet()) {
                    victimClaims.put(claim.getKey(), claim.getValue());
                }
            } catch (RuntimeException | Error installationFailure) {
                if (preparedAttempt != null) {
                    preemptionAttempts.remove(attemptToken, preparedAttempt);
                    decodeRequests.get(incomingRequestId).incomingAttemptToken = 0L;
                }
                for (Map.Entry<String, PreemptionClaim> claim : preparedClaims.entrySet()) {
                    claim.getValue().close();
                }
                settle(incomingReservation, Settlement.ROLLBACK);
                throw installationFailure;
            }
            admissionVersion++;
            return PreemptionBeginResult.SUCCESS;
        } finally {
            stateLock().unlock();
        }
    }

    /**
     * Resource transaction only. Its workflow owner publishes the returned edge after unlocking.
     */
    public ResourceChange<MutationStatus> updatePreemption(long token, PreemptionUpdate update) {
        java.util.Objects.requireNonNull(update, "update");
        stateLock().lock();
        try {
            PreemptionClaim claim = victimClaims.get(update.reservation().requestId());
            if (claim == null
                    || claim.attemptToken != token
                    || !claim.reservation.equals(update.reservation())) {
                return ResourceChange.of(MutationStatus.STALE, false);
            }
            switch (update.kind()) {
                case CANCELED, REQUEST_FENCED -> {
                    boolean changed = settle(claim.reservation, Settlement.TERMINAL);
                    return ResourceChange.of(
                            changed ? MutationStatus.APPLIED : MutationStatus.STALE, changed);
                }
                case CANCEL_HANDED_OFF -> claim.cancellationHandedOff = true;
                case ACTIVE, FINISHED -> {
                    long version = admissionVersion;
                    claim.close();
                    if (update.kind() == PreemptionUpdate.Kind.FINISHED) {
                        settle(claim.reservation, Settlement.OBSERVED_TERMINAL);
                    }
                    if (!calibrating && admissionVersion == version) {
                        admissionVersion++;
                    }
                }
                default ->
                        throw new IllegalStateException(
                                "Unhandled preemption update: " + update.kind());
            }
            return ResourceChange.of(MutationStatus.APPLIED, update.releasesCapacity());
        } finally {
            stateLock().unlock();
        }
    }

    RequestReservation finishPreemption(long token, boolean commit) {
        stateLock().lock();
        try {
            PreemptionRecord attempt = preemptionAttempts.get(token);
            if (attempt == null) {
                return null;
            }
            RequestRecord incoming = decodeRequests.get(attempt.incoming.requestId());
            if (commit
                    && (!acceptsNewRequests()
                            || incoming == null
                            || !incoming.matches(attempt.incoming)
                            || incoming.confirmed()
                            || !attempt.remainingVictims.isEmpty())) {
                return null;
            }
            long version = admissionVersion;
            preemptionAttempts.remove(token);
            if (incoming != null && incoming.matches(attempt.incoming)) {
                incoming.incomingAttemptToken = 0L;
            }
            if (!commit) {
                for (RequestReservation victim : attempt.remainingVictims) {
                    PreemptionClaim claim = victimClaims.get(victim.requestId());
                    if (claim != null
                            && claim.attemptToken == token
                            && claim.reservation.equals(victim)
                            && !claim.cancellationHandedOff) {
                        claim.close();
                    }
                }
                if (incoming != null && !incoming.confirmed()) {
                    settle(attempt.incoming, Settlement.ROLLBACK);
                }
            }
            if (!calibrating && admissionVersion == version) {
                admissionVersion++;
            }
            return attempt.incoming;
        } finally {
            stateLock().unlock();
        }
    }

    List<DecodeRequestStatus> calibrateLocked(WorkerStatus.StatusObservation observation) {
        checkState(
                stateLock().isHeldByCurrentThread(),
                "Decode calibration requires an Endpoint lock");
        checkArgument(
                observation.owner() == getStatus(), "Status belongs to another Decode generation");
        checkState(!calibrating, "Decode calibration cannot reenter");
        calibrating = true;
        try {
            List<DecodeRequestStatus> evidence = new ArrayList<>();
            Set<String> present = new HashSet<>(),
                    allocated = new HashSet<>(),
                    terminal = new HashSet<>();
            observation.finishedTasks().values().forEach(task -> terminal.add(task.requestId()));
            long now = System.currentTimeMillis();
            for (WorkerStatus.TaskObservation task :
                    observation.engine().runningTaskList().values()) {
                String id = task.requestId();
                if (terminal.contains(id) || settledHistory.containsKey(id)) {
                    continue;
                }
                present.add(id);
                boolean allocation =
                        task.phase() == TaskPhase.KV_ALLOCATED || task.phase() == TaskPhase.RUNNING;
                if (allocation) {
                    allocated.add(id);
                }
                observeActive(task, allocation, now);
                RequestRecord request = decodeRequests.get(id);
                if (request != null && request.reservationToken > 0L) {
                    evidence.add(
                            allocation
                                    ? DecodeRequestStatus.allocated(request.reservation())
                                    : DecodeRequestStatus.active(request.reservation()));
                }
            }
            // Capture exact terminal proof before absent-owner pruning can discard its identity.
            for (WorkerStatus.TaskObservation task : observation.finishedTasks().values()) {
                String id = task.requestId();
                if (settledHistory.containsKey(id)) {
                    continue;
                }
                RequestRecord request = decodeRequests.get(id);
                PreemptionClaim claim = victimClaims.get(id);
                RequestReservation exact =
                        request != null && request.reservationToken > 0L
                                ? request.reservation()
                                : claim == null ? null : claim.reservation;
                if (exact != null && settle(exact, Settlement.OBSERVED_TERMINAL)) {
                    evidence.add(DecodeRequestStatus.terminal(exact, task.errorCode()));
                } else if (exact == null && request != null && request.confirmed()) {
                    decodeRequests.remove(id, request);
                    transition(request, null, null, 0L, 0L);
                }
            }
            for (RequestRecord request : decodeRequests.values()) {
                if (!request.confirmed()) {
                    continue;
                }
                PreemptionClaim claim = victimClaims.get(request.requestId);
                if (claim != null) {
                    claim.hold(!allocated.contains(request.requestId));
                } else if (!present.contains(request.requestId)) {
                    // FULL absence ends this Engine owner and invalidates its provisional
                    // admission.
                    // Abort keeps already-outbound victim claims until their independent terminal
                    // proof.
                    if (request.incomingAttemptToken != 0L) {
                        finishPreemption(request.incomingAttemptToken, false);
                    }
                    if (decodeRequests.remove(request.requestId, request)) {
                        transition(request, null, null, 0L, 0L);
                    }
                    settledHistory.put(request.requestId, now);
                }
            }
            return List.copyOf(evidence);
        } finally {
            calibrating = false;
            admissionVersion++;
        }
    }

    private void observeActive(WorkerStatus.TaskObservation task, boolean allocated, long now) {
        RequestRecord request = decodeRequests.get(task.requestId());
        if (allocated) {
            PreemptionClaim claim = victimClaims.get(task.requestId());
            if (claim != null) {
                claim.hold(false);
            }
            if (request == null) {
                request = new RequestRecord(task.requestId(), 0L, 0L, 0, 0L);
                decodeRequests.put(task.requestId(), request);
                reservedUsage.adjust(1, 0L, 0L);
            }
            transition(
                    request,
                    task.phase() == TaskPhase.RUNNING
                            ? DecodeTaskPhase.RUNNING
                            : DecodeTaskPhase.ACCEPTED_NOT_RUNNING,
                    null,
                    task.inputLength(),
                    now);
        } else if (request != null) {
            transition(
                    request,
                    request.confirmed() ? request.phase : DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN,
                    null,
                    0L,
                    now);
        }
    }

    /**
     * A private candidate has no local owners and therefore cannot emit request evidence.
     */
    public void initializeFromPreparedStatus(
            WorkerStatus ws, WorkerStatus.StatusObservation observation) {
        requireStatusGeneration(ws);
        checkArgument(
                observation.owner() == getStatus(), "Status belongs to another Decode generation");
        stateLock().lock();
        try {
            checkState(
                    calibrateLocked(observation).isEmpty(),
                    "Private Decode candidate produced locally-owned request statuses");
        } finally {
            stateLock().unlock();
        }
    }

    List<DecodeRequestStatus> collectHeartbeatRequestStatuses(
            WorkerStatus.StatusObservation observation) {
        checkArgument(
                observation.owner() == getStatus(),
                "Status observation belongs to another Decode generation");
        List<DecodeRequestStatus> captured;
        stateLock().lock();
        try {
            List<DecodeRequestStatus> evidence = new ArrayList<>(observation.activeTasks().size());
            for (WorkerStatus.TaskObservation task : observation.activeTasks().values()) {
                RequestRecord request = decodeRequests.get(task.requestId());
                PreemptionClaim claim = victimClaims.get(task.requestId());
                RequestReservation exact =
                        request != null && request.reservationToken > 0L
                                ? request.reservation()
                                : request == null && claim != null ? claim.reservation : null;
                if (exact != null) {
                    evidence.add(
                            task.phase() == TaskPhase.KV_ALLOCATED
                                            || task.phase() == TaskPhase.RUNNING
                                    ? DecodeRequestStatus.allocated(exact)
                                    : DecodeRequestStatus.active(exact));
                }
            }
            captured = evidence;
        } finally {
            stateLock().unlock();
        }
        return List.copyOf(captured);
    }

    List<RequestReservation> retire() {
        Set<RequestReservation> owners = new HashSet<>();
        List<RequestReservation> foreignVictims = new ArrayList<>();
        long generationId = getStatus().getGenerationId();
        stateLock().lock();
        try {
            decodeRequests.forEach(
                    (requestId, reservation) -> {
                        if (reservation.reservationToken > 0L) {
                            owners.add(
                                    new RequestReservation(
                                            generationId, requestId, reservation.reservationToken));
                        }
                    });
            for (PreemptionRecord attempt : preemptionAttempts.values()) {
                owners.add(attempt.incoming);
                for (RequestReservation victim : attempt.remainingVictims) {
                    if (victim.endpointGenerationId() != generationId) {
                        foreignVictims.add(victim);
                        continue;
                    }
                    owners.add(victim);
                }
            }
            victimClaims.values().forEach(claim -> owners.add(claim.reservation));
            // Capture exact identities before clearing permits and accounting.
            for (RequestRecord reservation : decodeRequests.values()) {
                EngineDispatchPermit permit = reservation.dispatchPermit;
                if (permit != null) {
                    permit.retiredByEndpoint = true;
                    reservation.dispatchPermit = null;
                }
            }
            decodeRequests.clear();
            settledHistory.clear();
            victimClaims.clear();
            dispatchUsage.reset();
            reservedUsage.reset();
            queuedUsage.reset();
            confirmedEngineOwnedCount = 0;
            preemptionAttempts.clear();
            priorityPreemptionHeldKv = 0L;
            priorityPreemptionHeldExpectedKv = 0L;
            admissionVersion++;
        } finally {
            stateLock().unlock();
        }
        for (var victim : foreignVictims) {
            logger.error(
                    "Decode retirement ignored a priority victim from another generation: "
                            + "request_id={} expected_generation={} actual_generation={}",
                    victim.requestId(),
                    generationId,
                    victim.endpointGenerationId());
        }
        List<RequestReservation> ordered = new ArrayList<>(owners);
        ordered.sort(RETIREMENT_ORDER);
        return List.copyOf(ordered);
    }

    record CleanupCandidate(long reservationToken, DecodeTaskPhase phase, long observedAtMs) {}

    Map<String, CleanupCandidate> cleanupCandidates() {
        stateLock().lock();
        try {
            Map<String, CleanupCandidate> candidates = new HashMap<>();
            decodeRequests.forEach(
                    (id, request) ->
                            candidates.put(
                                    id,
                                    new CleanupCandidate(
                                            request.reservationToken,
                                            request.phase,
                                            request.observedAtMs)));
            return candidates;
        } finally {
            stateLock().unlock();
        }
    }

    CleanupResult expireReservations(long ttlMs, Map<String, CleanupCandidate> orphanCandidates) {
        stateLock().lock();
        try {
            long nowMs = System.currentTimeMillis();
            long cutoff = nowMs - ttlMs;
            int expiredReservations = 0;
            boolean changed = settledHistory.values().removeIf(settledAt -> settledAt < cutoff);
            for (Map.Entry<String, RequestRecord> entry : decodeRequests.entrySet()) {
                String requestId = entry.getKey();
                RequestRecord request = entry.getValue();
                CleanupCandidate candidate = orphanCandidates.get(requestId);
                if (victimClaims.containsKey(requestId)
                        || candidate == null
                        || candidate.reservationToken != request.reservationToken
                        || candidate.phase != request.phase
                        || candidate.observedAtMs != request.observedAtMs) {
                    continue;
                }
                boolean confirmed = request.confirmed();
                boolean expired =
                        confirmed
                                ? request.observedAtMs < cutoff
                                : nowMs - request.observedAtMs > ttlMs;
                if (!expired) {
                    continue;
                }
                // An orphan incoming row must also surrender its attempt and local
                // victim claims.
                if (request.incomingAttemptToken != 0L) {
                    finishPreemption(request.incomingAttemptToken, false);
                }
                if (decodeRequests.remove(requestId, request)) {
                    transition(request, null, null, 0L, 0L);
                }
                if (!confirmed) {
                    expiredReservations++;
                }
                changed = true;
            }
            if (changed) {
                admissionVersion++;
            }
            return new CleanupResult(expiredReservations, changed);
        } finally {
            stateLock().unlock();
        }
    }

    record CleanupResult(int expiredReservations, boolean capacityReleased) {}

    @Override
    public OptionalLong tryGetLoadMetric() {
        if (!stateLock().tryLock()) return OptionalLong.empty();
        try {
            return OptionalLong.of(confirmedEngineOwnedCount + reservedUsage.requests);
        } finally {
            stateLock().unlock();
        }
    }

    public DecodeRoutingView routingView() {
        return routingView(ipPort());
    }

    DecodeRoutingView routingView(String address) {
        DecodeRoutingView observed = routingViewCache;
        if (observed != null
                && observed.address().equals(address)
                && observed.admissionVersion() == admissionVersion
                && observed.workerStatus() == getStatus().committedWorkerStatus()
                && observed.topology() == getStatus().topologySnapshot()) {
            return observed;
        }
        stateLock().lock();
        try {
            WorkerStatus.CommittedWorkerStatus committed = getStatus().committedWorkerStatus();
            WorkerStatus.TopologySnapshot topology = getStatus().topologySnapshot();
            var cached = routingViewCache;
            if (cached == null
                    || !cached.address().equals(address)
                    || cached.admissionVersion() != admissionVersion
                    || cached.workerStatus() != committed
                    || cached.topology() != topology) {
                cached =
                        new DecodeRoutingView(
                                address,
                                getStatus().getGenerationId(),
                                topology,
                                committed,
                                admissionVersion,
                                confirmedEngineOwnedCount + reservedUsage.requests,
                                confirmedEngineOwnedCount
                                        + reservedUsage.requests
                                        - Math.clamp(
                                                queuedUsage.requests,
                                                0,
                                                Math.max(0, reservedUsage.requests)),
                                usage(false, committed.fields()),
                                usage(true, committed.fields()),
                                reservedUsage.hardKv);
                routingViewCache = cached;
            }
            return cached;
        } finally {
            stateLock().unlock();
        }
    }

    public DecodeResources.AdmissionSummary admissionSummary() {
        DecodeResources.AdmissionSummary observed = admissionSummaryCache;
        if (observed != null && observed.routing() == routingView()) {
            return observed;
        }
        long[] requests = new long[PriorityNormalizer.MAX_PRIORITY + 1];
        long[] hardKv = new long[PriorityNormalizer.MAX_PRIORITY + 1];
        long[] expectedKv = new long[PriorityNormalizer.MAX_PRIORITY + 1];
        long[] engineRequests = new long[PriorityNormalizer.MAX_PRIORITY + 1];
        long[] engineHardKv = new long[PriorityNormalizer.MAX_PRIORITY + 1];
        long[] engineExpectedKv = new long[PriorityNormalizer.MAX_PRIORITY + 1];
        DecodeRoutingView routing;
        stateLock().lock();
        try {
            var capturedRouting = routingView();
            for (RequestRecord task : decodeRequests.values()) {
                int priority =
                        task.priorityKnown() && PriorityNormalizer.isValid(task.priority)
                                ? task.priority
                                : 0;
                requests[priority]++;
                long hard = task.kvTokens;
                long expected = task.confirmed() ? task.kvTokens : task.expectedKvTokens;
                hardKv[priority] = saturatedAddNonNegative(hardKv[priority], hard);
                expectedKv[priority] = saturatedAddNonNegative(expectedKv[priority], expected);
                if (task.confirmed() || !task.queued() || task.dispatchPermit != null) {
                    engineRequests[priority]++;
                    engineHardKv[priority] = saturatedAddNonNegative(engineHardKv[priority], hard);
                    engineExpectedKv[priority] =
                            saturatedAddNonNegative(engineExpectedKv[priority], expected);
                }
            }
            routing = capturedRouting;
        } finally {
            stateLock().unlock();
        }
        CapacityRelease[] placementOccupancy =
                new CapacityRelease[PriorityNormalizer.MAX_PRIORITY + 1];
        CapacityRelease[] engineOccupancy =
                new CapacityRelease[PriorityNormalizer.MAX_PRIORITY + 1];
        for (int priority = 0; priority < placementOccupancy.length; priority++) {
            placementOccupancy[priority] =
                    requests[priority] == 0
                            ? CapacityRelease.NONE
                            : new CapacityRelease(
                                    requests[priority], hardKv[priority], expectedKv[priority]);
            engineOccupancy[priority] =
                    engineRequests[priority] == 0
                            ? CapacityRelease.NONE
                            : new CapacityRelease(
                                    engineRequests[priority],
                                    engineHardKv[priority],
                                    engineExpectedKv[priority]);
        }
        var summary =
                new DecodeResources.AdmissionSummary(routing, placementOccupancy, engineOccupancy);
        stateLock().lock();
        try {
            // A concurrent admission may have changed the version during object
            // construction.
            if (routing == routingView()) admissionSummaryCache = summary;
        } finally {
            stateLock().unlock();
        }
        return summary;
    }

    public ResourceSnapshot resourceSnapshot() {
        Map<String, DecodeRequestView> requests = new HashMap<>();
        DecodeRoutingView routing;
        int queued;
        int dispatching;
        stateLock().lock();
        try {
            decodeRequests.forEach(
                    (requestId, task) ->
                            requests.put(
                                    requestId,
                                    new DecodeRequestView(
                                            requestId,
                                            task.priority,
                                            task.kvTokens,
                                            task.expectedKvTokens,
                                            task.phase,
                                            task.priorityKnown(),
                                            task.reservationToken,
                                            victimClaims.containsKey(requestId))));
            routing = routingView();
            queued = queuedUsage.requests;
            dispatching = dispatchUsage.requests;
        } finally {
            stateLock().unlock();
        }
        return new ResourceSnapshot(routing, requests, queued, dispatching);
    }

    public long placementVersion() {
        return admissionVersion;
    }

    public EndpointStats.DecodeCounts stats() {
        long oldest = Long.MAX_VALUE;
        for (RequestRecord request : decodeRequests.values()) {
            long observed = request.observedAtMs;
            if (!request.confirmed()) {
                oldest = Math.min(oldest, observed);
            }
        }
        long age =
                oldest == Long.MAX_VALUE ? 0L : Math.max(0L, System.currentTimeMillis() - oldest);
        return new EndpointStats.DecodeCounts(
                reservedUsage.requests,
                confirmedEngineOwnedCount + reservedUsage.requests,
                reservedUsage.expectedKv,
                reservedUsage.hardKv,
                age);
    }

    private static final class ResourceUsage {
        private volatile int requests;
        private volatile long hardKv;
        private volatile long expectedKv;

        void adjust(int count, long hard, long expected) {
            if (count != 0) {
                requests += count;
            }
            if (hard != 0L) {
                hardKv += hard;
            }
            if (expected != 0L) {
                expectedKv += expected;
            }
        }

        void reset() {
            requests = 0;
            hardKv = 0L;
            expectedKv = 0L;
        }
    }

    final class RequestRecord {
        private final String requestId;
        private final long reservationToken;
        private final int priority;
        private long kvTokens;
        private long expectedKvTokens;
        private long observedAtMs = System.currentTimeMillis();
        private volatile DecodeTaskPhase phase = DecodeTaskPhase.LOCAL_RESERVED;
        private EngineDispatchPermit dispatchPermit;

        /**
         * Non-owning reverse index, maintained atomically with preemptionAttempts in Endpoint lock.
         */
        private long incomingAttemptToken;

        RequestRecord(String requestId, long hard, long expected, int priority, long token) {
            this.requestId = requestId;
            this.reservationToken = token;
            this.kvTokens = hard;
            this.expectedKvTokens = expected;
            this.priority = priority;
        }

        RequestReservation reservation() {
            return new RequestReservation(
                    getStatus().getGenerationId(), requestId, reservationToken);
        }

        boolean matches(RequestReservation exact) {
            return exact != null
                    && exact.endpointGenerationId() == getStatus().getGenerationId()
                    && java.util.Objects.equals(exact.requestId(), requestId)
                    && exact.reservationToken() == reservationToken;
        }

        boolean queued() {
            return phase == DecodeTaskPhase.MASTER_QUEUED_NOT_DISPATCHED;
        }

        boolean confirmed() {
            return phase.isEngineConfirmed();
        }

        boolean priorityKnown() {
            return reservationToken > 0L;
        }

        CapacityRelease capacityRelease() {
            return new CapacityRelease(1L, kvTokens, expectedKvTokens);
        }
    }

    private record PreemptionRecord(
            RequestReservation incoming, Set<RequestReservation> remainingVictims) {}

    private final class PreemptionClaim {
        private final long attemptToken;
        private final RequestReservation reservation;
        private final long hardKvTokens;
        private final long expectedKvTokens;
        private boolean cancellationHandedOff;
        private boolean kvHeldAfterWorkerRelease;

        PreemptionClaim(long token, RequestReservation reservation, long hard, long expected) {
            checkArgument(
                    hard >= 0L && expected >= hard,
                    "Priority preemption claim requires expected KV >= hard KV >= 0");
            this.attemptToken = token;
            this.reservation = reservation;
            this.hardKvTokens = hard;
            this.expectedKvTokens = expected;
        }

        void close() {
            if (victimClaims.get(reservation.requestId()) != this) {
                return;
            }
            hold(false);
            victimClaims.remove(reservation.requestId(), this);
        }

        void hold(boolean held) {
            checkState(
                    stateLock().isHeldByCurrentThread(),
                    "Priority preemption KV hold mutation requires an Endpoint lock");
            if (kvHeldAfterWorkerRelease == held) {
                return;
            }
            long hard = priorityPreemptionHeldKv, expected = priorityPreemptionHeldExpectedKv;
            checkState(
                    hard >= 0L && expected >= hard, "Invalid priority preemption KV hold counters");
            checkState(
                    held || hard >= hardKvTokens && expected >= expectedKvTokens,
                    "Priority preemption KV hold counter underflow");
            try {
                hard = Math.addExact(hard, held ? hardKvTokens : -hardKvTokens);
                expected = Math.addExact(expected, held ? expectedKvTokens : -expectedKvTokens);
            } catch (ArithmeticException overflow) {
                throw new IllegalStateException(
                        "Priority preemption KV hold counter overflow", overflow);
            }
            checkState(
                    hard >= 0L && expected >= hard, "Invalid priority preemption KV hold counters");
            if (held) {
                priorityPreemptionHeldExpectedKv = expected;
                priorityPreemptionHeldKv = hard;
            } else {
                priorityPreemptionHeldKv = hard;
                priorityPreemptionHeldExpectedKv = expected;
            }
            kvHeldAfterWorkerRelease = held;
        }
    }
}
