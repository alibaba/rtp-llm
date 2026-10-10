package org.flexlb.balance.endpoint;

import org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity;
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

    private void notifyRequestStatuses(List<DecodeRequestStatus> requestStatuses) {
        for (DecodeRequestStatus requestStatus : requestStatuses) {
            var context = requests.findActive(requestStatus.reservation().requestId());
            if (context != null) { context.scheduler().onDecodeStatus(context, this, requestStatus); }
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

    public ReservationHandle tryReserveQueuedRequest(GenerationPin pin, long requestId, long hardKv,
                                     long expectedKv, int priority, AdmissionCapacity capacity) {
        requirePinnedGeneration(pin);
        return state.tryReserveQueuedRequest(requestId, hardKv, expectedKv, priority, capacity);
    }

    /** LOCAL_ROLLBACK requires local ownership; other evidence may leave Engine/protocol ownership intact. */
    public ReservationReleaseResult release(ReservationHandle reservation, ReleaseReason reason) {
        ReservationReleaseResult result = state.release(reservation, reason);
        if (result == ReservationReleaseResult.RELEASED) { publishCapacityRelease(); }
        return result;
    }

    /** Exact resource obligation; absence also covers retired or replaced reservation identities. */
    public boolean hasOwnedResources(ReservationHandle reservation) { return state.hasOwnedResources(reservation); }

    public boolean isAcceptedByEngine(ReservationHandle reservation) { return state.isAcceptedByEngine(reservation); }

    // Dispatch: acquire capacity, hand over ownership, or return the permit.

    public boolean markQueued(GenerationPin pin, ReservationHandle reservation) {
        requirePinnedGeneration(pin);
        boolean changed = state.markQueued(reservation);
        if (changed) { publishCapacityRelease(); }
        return changed;
    }

    public EngineDispatchPermitAcquisition acquireDispatchPermit(
            ReservationHandle reservation, AdmissionCapacity capacity) {
        java.util.Objects.requireNonNull(reservation, "reservation");
        java.util.Objects.requireNonNull(capacity, "capacity");
        GenerationPin pin = tryPinGeneration();
        if (pin == null) {
            return new EngineDispatchPermitAcquisition(EngineDispatchPermitAcquireStatus.ENDPOINT_RETIRED, null);
        }
        try (pin) {
            DecodeState.DispatchAcquisition acquired = state.acquireDispatchPermit(reservation, capacity);
            return new EngineDispatchPermitAcquisition(acquired.status(), acquired.permit() == null ? null
                    : new EngineDispatchPermit(this, acquired.permit()));
        }
    }

    /** Resolve a permit once. ENGINE_OWNED means handoff, not Worker acceptance. */
    public EngineDispatchPermitTransferStatus dispatch(EngineDispatchPermit permit, DispatchOutcome outcome) {
        java.util.Objects.requireNonNull(permit, "permit");
        java.util.Objects.requireNonNull(outcome, "outcome");
        checkArgument(permit.endpoint == this, "Dispatch permit belongs to another endpoint");
        synchronized (permit) {
            if (permit.dispatchResult != null) {
                return outcome == DispatchOutcome.ENGINE_OWNED
                        ? permit.dispatchResult : EngineDispatchPermitTransferStatus.OWNERSHIP_LOST;
            }
            EngineDispatchPermitTransferStatus result;
            GenerationPin pin = outcome == DispatchOutcome.ENGINE_OWNED ? tryPinGeneration() : null;
            if (outcome == DispatchOutcome.ENGINE_OWNED && pin == null) {
                result = EngineDispatchPermitTransferStatus.ENDPOINT_RETIRED;
            } else {
                try (pin) {
                    DecodeState.DispatchResult applied = state.dispatch(permit.lease, outcome);
                    if (applied.capacityReleased()) {
                        if (outcome == DispatchOutcome.ABANDONED) { publishCapacityRelease(); } else { notifyEngineDispatchCapacityListeners(); }
                    }
                    result = applied.status();
                }
            }
            // Returning an unused permit succeeds once, but cannot grant sending ownership later.
            permit.dispatchResult = result == EngineDispatchPermitTransferStatus.TRANSFERRED
                    && outcome == DispatchOutcome.ABANDONED
                    ? EngineDispatchPermitTransferStatus.OWNERSHIP_LOST : result;
            return result;
        }
    }

    /** Lock-free waiter hint, including retirement or lost ownership. Acquisition still rechecks capacity. */
    public boolean shouldRetryDispatch(long requestId, AdmissionCapacity capacity) {
        return isRetired() || state.shouldRetryDispatch(requestId, capacity);
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
        public EngineDispatchPermitTransferStatus dispatch() {
            return endpoint.dispatch(this, DispatchOutcome.ENGINE_OWNED);
        }

        public boolean release() {
            return endpoint.dispatch(this, DispatchOutcome.ABANDONED)
                    == EngineDispatchPermitTransferStatus.TRANSFERRED;
        }

    }

    public record EngineDispatchPermitAcquisition(
            EngineDispatchPermitAcquireStatus status,
            EngineDispatchPermit permit) {

        public EngineDispatchPermitAcquisition {
            checkArgument(status != null, "permit acquisition status is required");
            boolean ownsHandoff = status == EngineDispatchPermitAcquireStatus.ACQUIRED
                    || status == EngineDispatchPermitAcquireStatus.ALREADY_ACCEPTED;
            checkArgument(ownsHandoff == (permit != null),
                    "only ACQUIRED or ALREADY_ACCEPTED results carry an engine dispatch permit");
        }
    }

    public void addEngineDispatchCapacityListener(Runnable listener) {
        if (listener != null) {
            engineDispatchCapacityListeners.add(listener);
        }
    }

    public void removeEngineDispatchCapacityListener(Runnable listener) {
        if (listener != null) {
            engineDispatchCapacityListeners.remove(listener);
        }
    }

    // Preemption: atomic local replacement or remote cancellation.

    public ReservationHandle replaceQueuedRequests(List<ReservationHandle> victims, long incomingRequestId,
                                         long hardKv, long expectedKv, int priority, AdmissionCapacity capacity) {
        GenerationPin pin = tryPinGeneration();
        if (pin == null) { return null; }
        try (pin) {
            ReservationHandle replaced = state.replaceQueuedRequests(victims, incomingRequestId, hardKv, expectedKv, priority, capacity);
            if (replaced != null) { publishCapacityRelease(); }
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

    public boolean updatePreemption(long attemptToken, PreemptionUpdate update) {
        boolean changed = reconcilePreemptionResources(attemptToken, update);
        if (changed && update.releasesCapacity()) { publishCapacityRelease(); }
        return changed;
    }

    /** Change exact resource ownership without callbacks; Scheduler publishes the resulting capacity edge after unlocking. */
    public boolean reconcilePreemptionResources(long attemptToken, PreemptionUpdate update) {
        return state.updatePreemption(attemptToken, update);
    }

    public ReservationHandle commitPreemption(long attemptToken) {
        return state.finishPreemption(attemptToken, true);
    }

    public boolean abortPreemption(long attemptToken) {
        boolean aborted = state.finishPreemption(attemptToken, false) != null;
        if (aborted) { publishCapacityRelease(); }
        return aborted;
    }

    // Calibration: publish facts only after the resource transaction returns.

    public Runnable applyPreparedStatus(WorkerStatus ws, WorkerStatus.PreparedStatus prepared) {
        requireStatusGeneration(ws);
        if (!prepared.observation().alive()) { beginRetirement(); }
        List<DecodeRequestStatus> requestStatuses;
        boolean capacityImproved;
        var lock = state.ownershipLock();
        lock.lock();
        try {
            checkArgument(prepared.observation().owner() == ws, "Status belongs to another Decode generation");
            DecodeRoutingView before = state.routingView();
            requestStatuses = state.calibrateLocked(prepared.observation());
            ws.publishPreparedStatus(prepared);
            capacityImproved = DecodeState.placementCapacityImproved(before, state.routingView());
        } catch (RuntimeException | Error failure) {
            beginRetirement();
            throw failure;
        } finally { lock.unlock(); }
        notifyEngineDispatchCapacityListeners();
        if (capacityImproved) { signalPlacementCapacityChanged(); }
        return () -> notifyRequestStatuses(requestStatuses);
    }

    public void initializeFromPreparedStatus(WorkerStatus ws, WorkerStatus.StatusObservation observation) {
        requireStatusGeneration(ws);
        state.initialize(observation);
    }

    public Runnable applyStatusHeartbeat(WorkerStatus ws, WorkerStatus.StatusObservation observation) {
        requireStatusGeneration(ws);
        List<DecodeRequestStatus> requestStatuses = state.collectHeartbeatRequestStatuses(observation);
        return () -> notifyRequestStatuses(requestStatuses);
    }

    // Read-only resource and capacity views.

    public AdmissionSummary admissionSummary() {
        return state.admissionSummary();
    }

    public DecodeRoutingView routingView() { return state.routingView(); }

    DecodeRoutingView routingViewSnapshot(String address) { return state.routingViewSnapshot(address); }

    public ResourceSnapshot resourceSnapshot() { return state.resourceSnapshot(); }

    public long placementVersion() { return state.placementVersion(); }

    public OptionalLong getLoadMetric() { return OptionalLong.of(state.getTotalLoad()); }

    // Retirement and orphan cleanup.

    public boolean isRetired() { return isGenerationRetiringOrRetired(); }

    protected void closeEndpoint() {
        List<ReservationHandle> reservations = state.retire();
        try { for (ReservationHandle reservation : reservations) {
            var context = requests.findActive(reservation.requestId());
            if (context != null) { context.scheduler().onDecodeGenerationRetired(context, this, reservation); }
        } }
        finally { notifyEngineDispatchCapacityListeners(); }
    }

    public int evictExpiredRequests(long ttlMs, LongPredicate retainForSchedulerCleanup) {
        var orphanCandidates = state.cleanupCandidates();
        orphanCandidates.keySet().removeIf(retainForSchedulerCleanup::test);
        DecodeState.CleanupResult result = state.evictExpiredRequests(ttlMs, orphanCandidates);
        if (result.capacityReleased()) { publishCapacityRelease(); }
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

    public void publishCapacityRelease() {
        notifyEngineDispatchCapacityListeners();
        signalPlacementCapacityChanged();
    }

    private void notifyEngineDispatchCapacityListeners() {
        for (Runnable listener : engineDispatchCapacityListeners) {
            try {
                listener.run();
            } catch (Throwable listenerFailure) {
                logger.warn("Decode capacity listener failed", listenerFailure);
            }
        }
    }

    private void signalPlacementCapacityChanged() {
        WorkerStatus.TopologySnapshot topology =
                getStatus().topologySnapshot();
        placementAvailability.changed(
                RoleType.DECODE, topology.group(), ipPort());
    }
}
