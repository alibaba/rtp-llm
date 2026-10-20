package org.flexlb.balance.endpoint;

import org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity;
import org.flexlb.balance.endpoint.DecodeResources.Acquisition;
import org.flexlb.balance.endpoint.DecodeResources.ResourceChange;
import org.flexlb.balance.endpoint.DecodeResources.ReserveStatus;
import org.flexlb.balance.endpoint.DecodeResources.MutationStatus;
import org.flexlb.balance.endpoint.DecodeResources.AdmissionSummary;
import org.flexlb.balance.endpoint.DecodeResources.DecodeRequestStatus;
import org.flexlb.balance.endpoint.DecodeResources.DecodeRoutingView;
import org.flexlb.balance.endpoint.DecodeResources.DispatchOutcome;
import org.flexlb.balance.endpoint.DecodeResources.EngineDispatchPermitAcquireStatus;
import org.flexlb.balance.endpoint.DecodeResources.EngineDispatchPermitTransferStatus;
import org.flexlb.balance.endpoint.DecodeResources.PreemptionBeginResult;
import org.flexlb.balance.endpoint.DecodeResources.PreemptionUpdate;
import org.flexlb.balance.endpoint.DecodeResources.ReleaseReason;
import org.flexlb.balance.endpoint.DecodeResources.ReservationHandle;
import org.flexlb.balance.endpoint.DecodeResources.ReservationReleaseResult;
import org.flexlb.balance.endpoint.DecodeResources.ResourceSnapshot;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.balance.scheduler.ContextResult;
import org.flexlb.balance.scheduler.RequestRepository;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.OptionalLong;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.LongPredicate;

import static com.google.common.base.Preconditions.checkArgument;

/** Decode worker boundary: lifecycle pins, resource operations and lock-free notifications.
 * DecodeState owns the generation-local resource ledger and its single mutation lock.
 */
public class DecodeEndpoint extends WorkerEndpoint {
    private static final Logger logger = LoggerFactory.getLogger("syncLogger");
    private final DecodeState state;
    private final RequestRepository requests;
    private final PlacementAvailability placementAvailability;
    private final Set<Runnable> engineDispatchCapacityListeners = ConcurrentHashMap.newKeySet();

    // Construction

    @Override
    public void publishStatus(StatusChange change) {
        if (change instanceof StatusChange.None) { return; }
        var effects = (StatusChange.Decode) change;
        if (effects.dispatchCapacityChanged()) { publishCapacity(CapacityEdge.DISPATCH); }
        if (effects.placementCapacityChanged()) { publishCapacity(CapacityEdge.PLACEMENT); }
        for (DecodeRequestStatus requestStatus : effects.requests()) {
            var context = requests.findActive(requestStatus.reservation().requestId());
            if (context == null) { continue; }
            var route = context.snapshot().route();
            if (route == null || route.decodeEp() != this) { continue; }
            context.data().getScheduler().onEngineEvidence(context, new ContextResult.EngineEvidence(
                    route, requestStatus.reservation(), RoleType.DECODE,
                    requestStatus.kind() == DecodeRequestStatus.Kind.TERMINAL
                            ? requestStatus.errorCode() == 0L ? ContextResult.EngineKind.COMPLETED : ContextResult.EngineKind.FAILED
                            : requestStatus.allocationObserved() ? ContextResult.EngineKind.ALLOCATED : ContextResult.EngineKind.ACTIVE,
                    requestStatus.errorCode(), "", System.currentTimeMillis()));
        }
    }

    DecodeEndpoint(WorkerStatus status, RequestRepository requests,
                   PlacementAvailability placementAvailability) {
        super(status);
        this.state = new DecodeState(status);
        this.requests = java.util.Objects.requireNonNull(requests, "requests");
        this.placementAvailability = java.util.Objects.requireNonNull(placementAvailability, "placementAvailability");
    }

    // Reservation and release. Lifecycle pins remain valid for the entire handoff.

    public Acquisition<ReserveStatus, ReservationHandle> tryReserveQueuedRequest(GenerationPin pin, long requestId, long hardKv,
                                     long expectedKv, int priority, AdmissionCapacity capacity) {
        requirePinnedGeneration(pin);
        return state.tryReserveQueuedRequest(requestId, hardKv, expectedKv, priority, capacity);
    }

    /** LOCAL_ROLLBACK requires local ownership; other evidence may leave Engine/protocol ownership intact. */
    public ResourceChange<ReservationReleaseResult> release(ReservationHandle reservation, ReleaseReason reason) {
        ResourceChange<ReservationReleaseResult> result = state.release(reservation, reason);
        if (result.capacityReleased()) { publishCapacity(CapacityEdge.BOTH); }
        return result;
    }

    /** Exact resource obligation; absence also covers retired or replaced reservation identities. */
    public DecodeResources.Ownership ownership(ReservationHandle reservation) { return state.ownership(reservation); }

    // Dispatch: acquire capacity, hand over ownership, or return the permit.

    public ResourceChange<MutationStatus> markQueued(GenerationPin pin, ReservationHandle reservation) {
        requirePinnedGeneration(pin);
        ResourceChange<MutationStatus> change = state.markQueued(reservation);
        if (change.capacityReleased()) { publishCapacity(CapacityEdge.BOTH); }
        return change;
    }

    public Acquisition<EngineDispatchPermitAcquireStatus, EngineDispatchPermit> acquireDispatchPermit(
            ReservationHandle reservation, AdmissionCapacity capacity) {
        java.util.Objects.requireNonNull(reservation, "reservation");
        java.util.Objects.requireNonNull(capacity, "capacity");
        GenerationPin pin = tryPinGeneration();
        if (pin == null) {
            return Acquisition.unavailable(EngineDispatchPermitAcquireStatus.ENDPOINT_RETIRED);
        }
        try (pin) {
            var acquired = state.acquireDispatchPermit(reservation, capacity);
            return acquired.handle() == null ? Acquisition.unavailable(acquired.status())
                    : new Acquisition<>(acquired.status(), new EngineDispatchPermit(this, acquired.handle()));
        }
    }

    /** Resolve a permit once. ENGINE_OWNED means handoff, not Worker acceptance. */
    public ResourceChange<EngineDispatchPermitTransferStatus> dispatch(EngineDispatchPermit permit, DispatchOutcome outcome) {
        java.util.Objects.requireNonNull(permit, "permit");
        java.util.Objects.requireNonNull(outcome, "outcome");
        checkArgument(permit.endpoint == this, "Dispatch permit belongs to another endpoint");
        synchronized (permit) {
            if (permit.dispatchResult != null) {
                return ResourceChange.of(outcome == DispatchOutcome.ENGINE_OWNED
                        ? permit.dispatchResult : EngineDispatchPermitTransferStatus.OWNERSHIP_LOST, false);
            }
            ResourceChange<EngineDispatchPermitTransferStatus> result;
            GenerationPin pin = outcome == DispatchOutcome.ENGINE_OWNED ? tryPinGeneration() : null;
            if (outcome == DispatchOutcome.ENGINE_OWNED && pin == null) {
                result = ResourceChange.of(EngineDispatchPermitTransferStatus.ENDPOINT_RETIRED, false);
            } else {
                try (pin) {
                    result = state.dispatch(permit.lease, outcome);
                    if (result.capacityReleased()) {
                        publishCapacity(outcome == DispatchOutcome.ABANDONED ? CapacityEdge.BOTH : CapacityEdge.DISPATCH);
                    }
                }
            }
            // Returning an unused permit succeeds once, but cannot grant sending ownership later.
            permit.dispatchResult = result.status() == EngineDispatchPermitTransferStatus.TRANSFERRED
                    && outcome == DispatchOutcome.ABANDONED
                    ? EngineDispatchPermitTransferStatus.OWNERSHIP_LOST : result.status();
            return result;
        }
    }

    /** Lock-free waiter hint, including retirement or lost ownership. Acquisition still rechecks capacity. */
    public boolean shouldRetryDispatch(ReservationHandle reservation, AdmissionCapacity capacity) {
        return isRetired() || state.shouldRetryDispatch(reservation, capacity);
    }

    public static final class EngineDispatchPermit {

        private final DecodeEndpoint endpoint;
        private final DecodeState.DispatchLease lease;
        /** Null until resolved; cached result for dispatch, including ownership lost after release. */
        private EngineDispatchPermitTransferStatus dispatchResult;

        private EngineDispatchPermit(DecodeEndpoint endpoint, DecodeState.DispatchLease lease) {
            this.endpoint = endpoint;
            this.lease = lease;
        }

        /** Identity check uses immutable lease fields; it does not acquire the resource lock. */
        public boolean belongsTo(DecodeEndpoint expected, ReservationHandle reservation) {
            return endpoint == expected && reservation != null
                    && endpoint.getStatus().getGenerationId() == reservation.endpointGenerationId()
                    && lease.matches(reservation);
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
            return endpoint.dispatch(this, DispatchOutcome.ABANDONED).status()
                    == EngineDispatchPermitTransferStatus.TRANSFERRED;
        }

    }

    /** The caller owns this subscription and closes it when its capacity wait ends. */
    public Runnable watchEngineDispatchCapacity(Runnable listener) {
        java.util.Objects.requireNonNull(listener, "listener");
        engineDispatchCapacityListeners.add(listener);
        return () -> engineDispatchCapacityListeners.remove(listener);
    }

    // Preemption: atomic local replacement or remote cancellation.

    public ReservationHandle replaceQueuedRequests(List<ReservationHandle> victims, long incomingRequestId,
                                         long hardKv, long expectedKv, int priority, AdmissionCapacity capacity) {
        GenerationPin pin = tryPinGeneration();
        if (pin == null) { return null; }
        try (pin) {
            ReservationHandle replaced = state.replaceQueuedRequests(victims, incomingRequestId, hardKv, expectedKv, priority, capacity);
            if (replaced != null) { publishCapacity(CapacityEdge.BOTH); }
            return replaced;
        }
    }

    public PreemptionBeginResult beginPreemption(long attemptToken, List<ReservationHandle> victims,
                                                 long incomingRequestId, long hardKv, long expectedKv,
                                                 int priority, AdmissionCapacity capacity) {
        checkArgument(attemptToken > 0 && victims != null && !victims.isEmpty(),
                "attempt token and victims are required");
        GenerationPin pin = tryPinGeneration();
        if (pin == null) { return PreemptionBeginResult.ENDPOINT_RETIRED; }
        try (pin) {
            return state.beginPreemption(attemptToken, victims, incomingRequestId, hardKv, expectedKv, priority, capacity);
        }
    }

    /** Resource transaction only. Its workflow owner publishes the returned edge after unlocking. */
    public ResourceChange<MutationStatus> updatePreemption(long attemptToken, PreemptionUpdate update) {
        return state.updatePreemption(attemptToken, update);
    }

    public ReservationHandle commitPreemption(long attemptToken) {
        return state.finishPreemption(attemptToken, true);
    }

    public boolean abortPreemption(long attemptToken) {
        boolean aborted = state.finishPreemption(attemptToken, false) != null;
        if (aborted) { publishCapacity(CapacityEdge.BOTH); }
        return aborted;
    }

    // Calibration: publish facts only after the resource transaction returns.

    public StatusChange applyPreparedStatus(WorkerStatus ws, WorkerStatus.PreparedStatus prepared) {
        requireStatusGeneration(ws);
        if (!prepared.observation().alive()) { beginRetirement(); }
        List<DecodeRequestStatus> requestStatuses;
        boolean capacityImproved;
        var lock = state.ownershipLock();
        lock.lock();
        try {
            checkArgument(prepared.observation().owner() == ws, "Status belongs to another Decode generation");
            DecodeRoutingView before = state.routingView(ipPort());
            requestStatuses = state.calibrateLocked(prepared.observation());
            ws.publishPreparedStatus(prepared);
            DecodeRoutingView after = state.routingView(ipPort());
            capacityImproved = after.engineLoad() < before.engineLoad()
                    || after.realKvAvailable() > before.realKvAvailable()
                    || after.realKvUsed() < before.realKvUsed();
        } catch (RuntimeException | Error failure) {
            beginRetirement();
            throw failure;
        } finally { lock.unlock(); }
        return new StatusChange.Decode(requestStatuses, capacityImproved, true);
    }

    public void initializeFromPreparedStatus(WorkerStatus ws, WorkerStatus.StatusObservation observation) {
        requireStatusGeneration(ws);
        state.initialize(observation);
    }

    public StatusChange applyStatusHeartbeat(WorkerStatus ws, WorkerStatus.StatusObservation observation) {
        requireStatusGeneration(ws);
        List<DecodeRequestStatus> requestStatuses = state.collectHeartbeatRequestStatuses(observation);
        return new StatusChange.Decode(requestStatuses, false, false);
    }

    // Read-only resource and capacity views.

    public AdmissionSummary admissionSummary() {
        return state.admissionSummary();
    }

    public DecodeRoutingView routingView() { return state.routingView(ipPort()); }

    DecodeRoutingView routingView(String address) { return state.routingView(address); }

    public ResourceSnapshot resourceSnapshot() { return state.resourceSnapshot(); }

    public long placementVersion() { return state.placementVersion(); }

    public OptionalLong tryGetLoadMetric() { return state.tryLoadMetric(); }

    @Override
    public OptionalLong getLoadMetric() { return OptionalLong.of(state.routingView(ipPort()).totalLoad()); }

    // Retirement and orphan cleanup.

    public boolean isRetired() { return isGenerationRetiringOrRetired(); }

    protected void closeEndpoint() {
        List<ReservationHandle> reservations = state.retire();
        try { for (ReservationHandle reservation : reservations) {
            var context = requests.findActive(reservation.requestId());
            if (context == null) { continue; }
            var route = context.snapshot().route();
            if (route != null && route.decodeEp() == this) {
                context.data().getScheduler().onEngineEvidence(context, new ContextResult.EngineEvidence(
                        route, reservation, RoleType.DECODE, ContextResult.EngineKind.RETIRED, 0L,
                        "Decode generation retired: " + reservation.endpointGenerationId(), System.currentTimeMillis()));
            }
        } }
        finally { publishCapacity(CapacityEdge.DISPATCH); }
    }

    public int evictExpiredRequests(long ttlMs, LongPredicate retainForSchedulerCleanup) {
        var orphanCandidates = state.cleanupCandidates();
        orphanCandidates.keySet().removeIf(retainForSchedulerCleanup::test);
        DecodeState.CleanupResult result = state.evictExpiredRequests(ttlMs, orphanCandidates);
        if (result.capacityReleased()) { publishCapacity(CapacityEdge.BOTH); }
        return result.expiredReservations();
    }

    // Metrics and shared lock-free capacity notifications.

    public void reportBatchMetrics(DeliveryMetricsReporter reporter) {
        DecodeState.Stats stats = state.stats();
        reporter.reportDecodeInflight(getIp(), stats.inflight(), stats.totalLoad(),
                stats.expectedKv(), stats.hardKv(), stats.oldestAgeMs());
    }

    public void reportAdmissionMetrics(RequestSchedulerReporter reporter) {
        reporter.reportDecodeAdmission(ipPort(), resourceSnapshot());
    }

    public enum CapacityEdge { DISPATCH, PLACEMENT, BOTH }

    public void publishCapacity(CapacityEdge edge) {
        if (edge != CapacityEdge.PLACEMENT) {
            for (Runnable listener : engineDispatchCapacityListeners) {
                try { listener.run(); }
                catch (Throwable failure) { logger.warn("Decode capacity listener failed", failure); }
            }
        }
        if (edge != CapacityEdge.DISPATCH) {
            var topology = getStatus().topologySnapshot();
            placementAvailability.changed(RoleType.DECODE, topology.group(), ipPort());
        }
    }
}
