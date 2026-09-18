package org.flexlb.balance.endpoint;

import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.balance.scheduler.EndpointEventProjector;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.DecodeTaskPhase;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;
import java.util.OptionalLong;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.LongPredicate;

/** Decode worker boundary: lifecycle pins, resource operations and lock-free notifications.
 * DecodeState owns the generation-local resource ledger and its single mutation lock.
 */
public class DecodeEndpoint extends WorkerEndpoint {
    private static final Logger logger = LoggerFactory.getLogger("syncLogger");
    private final DecodeState state;
    private final EndpointEventProjector endpointEvents;
    private final PlacementAvailability placementAvailability;
    private final Set<Runnable> engineDispatchCapacityListeners = ConcurrentHashMap.newKeySet();

    // Construction

    public DecodeEndpoint(WorkerStatus status, EndpointEventProjector endpointEvents) {
        this(status, endpointEvents, new PlacementAvailability());
    }

    DecodeEndpoint(WorkerStatus status, EndpointEventProjector endpointEvents,
                   PlacementAvailability placementAvailability) {
        super(status);
        this.state = new DecodeState(status);
        this.endpointEvents = java.util.Objects.requireNonNull(endpointEvents, "endpointEvents");
        this.placementAvailability = java.util.Objects.requireNonNull(placementAvailability, "placementAvailability");
    }

    // Reservation and release. Lifecycle pins remain valid for the entire handoff.

    public ReservationHandle reserve(GenerationPin pin, long requestId, long hardKv,
                                     long expectedKv, int priority) {
        return reserve(pin, requestId, hardKv, expectedKv, priority, null);
    }

    public ReservationHandle reserve(GenerationPin pin, long requestId, long hardKv,
                                     long expectedKv, int priority, AdmissionCapacity capacity) {
        requirePinnedGeneration(pin);
        return state.reserve(requestId, hardKv, expectedKv, priority, true, capacity);
    }

    /** An engine-facing shadow for work already outside the local queue. */
    public ReservationHandle reserveUnqueued(GenerationPin pin, long requestId, long hardKv,
                                             long expectedKv, int priority) {
        requirePinnedGeneration(pin);
        ReservationHandle reservation = state.reserve(requestId, hardKv, expectedKv, priority, false, null);
        if (reservation == null) {
            throw new IllegalStateException("Decode request id is already owned: " + requestId);
        }
        return reservation;
    }

    /** LOCAL_ROLLBACK requires local ownership; other evidence may leave Engine/protocol ownership intact. */
    public ReservationReleaseResult release(ReservationHandle reservation, ReleaseReason reason) {
        ReservationReleaseResult result = state.release(reservation, reason);
        if (result == ReservationReleaseResult.RELEASED) { publishCapacityRelease(); }
        return result;
    }

    /** Select no user outcome here: Prefill rejection alone cannot release Decode ownership. */
    public boolean settleFailedRequest(ReservationHandle reservation, DeliveryResult.Status source) {
        if (source != DeliveryResult.Status.NOT_SENT && source != DeliveryResult.Status.PREFILL_REJECTED) {
            throw new IllegalArgumentException("expected a definite request failure");
        }
        if (reservation == null) { return true; }
        if (source == DeliveryResult.Status.NOT_SENT) { release(reservation, ReleaseReason.NOT_SENT); }
        return !state.hasOwnedResources(reservation);
    }

    public boolean isAcceptedByEngine(ReservationHandle reservation) { return state.isAcceptedByEngine(reservation); }

    public ReservationHandle reservationHandle(long requestId) {
        return isRetired() ? null : state.reservationHandle(requestId);
    }

    public record ReservationHandle(
            long endpointGenerationId,
            long requestId,
            long reservationToken) {

        public ReservationHandle {
            if (endpointGenerationId <= 0L || reservationToken <= 0L) {
                throw new IllegalArgumentException(
                        "Decode reservation identity must be positive");
            }
        }
    }

    public enum ReleaseReason {
        LOCAL_ROLLBACK, COUNTERPART_FINISHED, NOT_SENT, EXPIRED
    }

    public enum ReservationReleaseResult {
        RELEASED,
        ENGINE_ACCEPTED,
        STILL_OWNED,
        STALE,
        CONFLICT;

        public boolean released() { return this == RELEASED; }
    }

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
                    : new EngineDispatchPermit(this, reservation.requestId(), acquired.permit()));
        }
    }

    /** Resolve a permit once. ENGINE_OWNED means handoff, not Worker acceptance. */
    public EngineDispatchPermitTransferStatus dispatch(EngineDispatchPermit permit, DispatchOutcome outcome) {
        java.util.Objects.requireNonNull(permit, "permit");
        java.util.Objects.requireNonNull(outcome, "outcome");
        if (permit.endpoint != this) { throw new IllegalArgumentException("Dispatch permit belongs to another endpoint"); }
        return permit.resolve(outcome);
    }

    private EngineDispatchPermitTransferStatus applyDispatch(EngineDispatchPermit permit, DispatchOutcome outcome) {
        GenerationPin pin = outcome == DispatchOutcome.ENGINE_OWNED ? tryPinGeneration() : null;
        if (outcome == DispatchOutcome.ENGINE_OWNED && pin == null) {
            return EngineDispatchPermitTransferStatus.ENDPOINT_RETIRED;
        }
        try (pin) {
            DecodeState.DispatchResult result = state.dispatch(permit.lease, outcome);
            if (result.capacityReleased()) {
                if (outcome == DispatchOutcome.ABANDONED) { publishCapacityRelease(); }
                else { notifyEngineDispatchCapacityListeners(); }
            }
            return result.status();
        }
    }

    /** Lock-free waiter hint, including retirement or lost ownership. Acquisition still rechecks capacity. */
    public boolean shouldRetryDispatch(long requestId, AdmissionCapacity capacity) {
        return isRetired() || state.shouldRetryDispatch(requestId, capacity);
    }

    public static final class EngineDispatchPermit {

        private enum Resolution {
            ACQUIRED,
            ENGINE_LIFECYCLE_OWNED,
            RELEASED,
            INVALIDATED,
            ENDPOINT_RETIRED
        }

        private final DecodeEndpoint endpoint;
        private final long requestId;
        private final DecodeState.DispatchLease lease;
        private Resolution resolution = Resolution.ACQUIRED;

        private EngineDispatchPermit(DecodeEndpoint endpoint,
                                     long requestId,
                                     DecodeState.DispatchLease lease) {
            this.endpoint = endpoint;
            this.requestId = requestId;
            this.lease = lease;
        }

        public long requestId() {
            return requestId;
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

        private synchronized EngineDispatchPermitTransferStatus resolve(DispatchOutcome outcome) {
            if (outcome == DispatchOutcome.ENGINE_OWNED) {
                if (resolution == Resolution.ENGINE_LIFECYCLE_OWNED) {
                    return EngineDispatchPermitTransferStatus.TRANSFERRED;
                }
                if (resolution == Resolution.ENDPOINT_RETIRED) {
                    return EngineDispatchPermitTransferStatus.ENDPOINT_RETIRED;
                }
            }
            if (resolution != Resolution.ACQUIRED) {
                return EngineDispatchPermitTransferStatus.OWNERSHIP_LOST;
            }
            EngineDispatchPermitTransferStatus result = endpoint.applyDispatch(this, outcome);
            resolution = switch (result) {
                case TRANSFERRED -> outcome == DispatchOutcome.ENGINE_OWNED
                        ? Resolution.ENGINE_LIFECYCLE_OWNED : Resolution.RELEASED;
                case ENDPOINT_RETIRED -> Resolution.ENDPOINT_RETIRED;
                case OWNERSHIP_LOST -> Resolution.INVALIDATED;
            };
            return result;
        }
    }

    public record EngineDispatchPermitAcquisition(
            EngineDispatchPermitAcquireStatus status,
            EngineDispatchPermit permit) {

        public EngineDispatchPermitAcquisition {
            if (status == null) {
                throw new IllegalArgumentException("permit acquisition status is required");
            }
            boolean ownsHandoff = status == EngineDispatchPermitAcquireStatus.ACQUIRED
                    || status == EngineDispatchPermitAcquireStatus.ALREADY_ACCEPTED;
            if (ownsHandoff != (permit != null)) {
                throw new IllegalArgumentException(
                        "only ACQUIRED or ALREADY_ACCEPTED results carry an engine dispatch permit");
            }
        }
    }

    public enum EngineDispatchPermitAcquireStatus {
        /** An exact-reservation permit now owns one Decode hard-gate slot. */
        ACQUIRED,
        /** Engine already owns this reservation; the permit carries identity without charging capacity. */
        ALREADY_ACCEPTED,
        /** Concurrency or Decode KV has no unreserved hard capacity. */
        CAPACITY_FULL,
        /** The request no longer owns a live shadow reservation. */
        NOT_OWNED,
        /** The reservation is already engine-facing rather than Prefill-queued. */
        NOT_QUEUED,
        /** Another pre-delivery attempt already owns this request's permit. */
        ALREADY_ACQUIRED,
        /** This exact Decode endpoint generation no longer accepts delivery. */
        ENDPOINT_RETIRED
    }

    public enum EngineDispatchPermitTransferStatus {
        TRANSFERRED,
        OWNERSHIP_LOST,
        ENDPOINT_RETIRED
    }

    public enum DispatchOutcome { ENGINE_OWNED, ABANDONED }

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

    public boolean replaceQueuedRequests(List<ReservationHandle> victims, long incomingRequestId,
                                         long hardKv, long expectedKv, int priority, AdmissionCapacity capacity) {
        GenerationPin pin = tryPinGeneration();
        if (pin == null) { return false; }
        try (pin) {
            boolean replaced = state.replaceQueuedRequests(victims, incomingRequestId, hardKv, expectedKv, priority, capacity);
            if (replaced) { publishCapacityRelease(); }
            return replaced;
        }
    }

    public PreemptionBeginResult beginPreemption(long attemptToken, List<ReservationHandle> victims,
                                                 long incomingRequestId, long hardKv, long expectedKv,
                                                 int priority, AdmissionCapacity capacity) {
        if (attemptToken <= 0 || victims == null || victims.isEmpty()) {
            throw new IllegalArgumentException("attempt token and victims are required");
        }
        GenerationPin pin = tryPinGeneration();
        if (pin == null) { return PreemptionBeginResult.ENDPOINT_RETIRED; }
        try (pin) {
            return state.beginPreemption(attemptToken, victims, incomingRequestId, hardKv, expectedKv, priority, capacity);
        }
    }

    public boolean updatePreemption(long attemptToken, PreemptionUpdate update) {
        boolean changed = state.updatePreemption(attemptToken, update);
        if (changed && update.releasesCapacity()) { publishCapacityRelease(); }
        return changed;
    }

    public boolean finishPreemption(long attemptToken, PreemptionDecision decision) {
        boolean changed = state.finishPreemption(attemptToken, decision);
        if (changed && decision == PreemptionDecision.ABORT) { publishCapacityRelease(); }
        return changed;
    }

    public enum PreemptionBeginResult {
        SUCCESS,
        ENDPOINT_RETIRED,
        VICTIM_GONE,
        VICTIM_ALREADY_CLAIMED,
        INVALID_PRIORITY,
        INFEASIBLE,
        INCOMING_ALREADY_RESERVED,
        ATTEMPT_ALREADY_EXISTS
    }

    public enum PreemptionDecision { COMMIT, ABORT }

    public record PreemptionUpdate(Kind kind, long requestId, ReservationHandle reservation,
                                   PreemptionCancelPhase phase) {
        public enum Kind { CANCEL_SENDING, CANCEL_REPLY, CANCELED, REQUEST_FENCED, ACTIVE, FINISHED }

        public PreemptionUpdate {
            java.util.Objects.requireNonNull(kind, "kind");
            if (kind == Kind.CANCEL_REPLY) {
                if (phase != PreemptionCancelPhase.CANCEL_REQUESTED
                        && phase != PreemptionCancelPhase.NOT_FOUND_STALE
                        && phase != PreemptionCancelPhase.CANCEL_UNKNOWN) {
                    throw new IllegalArgumentException("Expected a Cancel reply phase");
                }
            } else if (phase != null) {
                throw new IllegalArgumentException("Only a Cancel reply carries a phase");
            }
            if (kind != Kind.CANCEL_REPLY && kind != Kind.CANCEL_SENDING) {
                java.util.Objects.requireNonNull(reservation, "reservation");
                if (requestId != reservation.requestId()) { throw new IllegalArgumentException("Victim identity mismatch"); }
            } else if (reservation != null) {
                throw new IllegalArgumentException("Cancel progress does not carry a reservation");
            }
        }
        public static PreemptionUpdate cancelSending() { return new PreemptionUpdate(Kind.CANCEL_SENDING, 0, null, null); }
        public static PreemptionUpdate cancelReply(long requestId, PreemptionCancelPhase phase) {
            return new PreemptionUpdate(Kind.CANCEL_REPLY, requestId, null, phase);
        }
        public static PreemptionUpdate canceled(ReservationHandle victim) { return victim(Kind.CANCELED, victim); }
        public static PreemptionUpdate fenced(ReservationHandle victim) { return victim(Kind.REQUEST_FENCED, victim); }
        public static PreemptionUpdate active(ReservationHandle victim) { return victim(Kind.ACTIVE, victim); }
        public static PreemptionUpdate finished(ReservationHandle victim) { return victim(Kind.FINISHED, victim); }
        private static PreemptionUpdate victim(Kind kind, ReservationHandle victim) {
            return new PreemptionUpdate(kind, victim.requestId(), victim, null);
        }
        boolean releasesCapacity() { return kind != Kind.CANCEL_SENDING && kind != Kind.CANCEL_REPLY; }
    }

    // Calibration: publish facts only after the resource transaction returns.

    public Runnable applyPreparedStatus(WorkerStatus ws, WorkerStatus.PreparedStatus prepared) {
        requireStatusGeneration(ws);
        if (!prepared.observation().alive()) { beginRetirement(); }
        DecodeState.CalibrationResult result;
        try {
            result = state.calibrate(prepared);
        } catch (RuntimeException | Error failure) {
            beginRetirement();
            throw failure;
        }
        notifyEngineDispatchCapacityListeners();
        if (result.capacityImproved()) { signalPlacementCapacityChanged(); }
        return () -> endpointEvents.onDecodeStatus(this, result.facts());
    }

    public Runnable initializeFromPreparedStatus(WorkerStatus ws, WorkerStatus.StatusObservation observation) {
        requireStatusGeneration(ws);
        state.initialize(observation);
        return () -> { };
    }

    public Runnable observeStatusHeartbeat(WorkerStatus ws, WorkerStatus.StatusObservation observation) {
        requireStatusGeneration(ws);
        List<WorkerStatusFact> facts = state.observeHeartbeat(observation);
        return () -> endpointEvents.onDecodeStatus(this, facts);
    }

    public record WorkerStatusFact(
            Kind kind,
            ReservationHandle reservation,
            long errorCode) {
        public WorkerStatusFact {
            java.util.Objects.requireNonNull(kind, "kind");
            java.util.Objects.requireNonNull(reservation, "reservation");
            if (kind != Kind.TERMINAL && errorCode != 0L) {
                throw new IllegalArgumentException(
                        "only a terminal Decode fact may carry an error code");
            }
        }

        public static WorkerStatusFact active(ReservationHandle reservation) {
            return new WorkerStatusFact(Kind.ACTIVE, reservation, 0L);
        }

        public static WorkerStatusFact accepted(ReservationHandle reservation) {
            return new WorkerStatusFact(Kind.ACCEPTED, reservation, 0L);
        }

        public static WorkerStatusFact terminal(
                ReservationHandle reservation, long errorCode) {
            return new WorkerStatusFact(Kind.TERMINAL, reservation, errorCode);
        }

        public enum Kind {
            ACTIVE,
            ACCEPTED,
            TERMINAL
        }
    }

    // Read-only resource and capacity views.

    public DecodeRoutingView routingView() { return state.routingView(); }

    DecodeRoutingView routingViewSnapshot(String address) { return state.routingViewSnapshot(address); }

    public LayeredAdmissionView resourceSnapshot() { return state.resourceSnapshot(); }

    public long placementVersion() { return state.placementVersion(); }

    public long realKvAvailable() { return state.realKvAvailable(); }

    public int getInflightCount() { return state.getInflightCount(); }

    public OptionalLong getLoadMetric() { return OptionalLong.of(state.getTotalLoad()); }

    public record LayeredAdmissionView(DecodeRoutingView routing,
                                       Map<Long, DecodeRequestView> reserved,
                                       List<DecodeRequestView> confirmed,
                                       int queuedCount,
                                       int activeDispatchPermits) {

        public long admissionVersion() {
            return routing.admissionVersion();
        }

        public int acceptedCount() {
            return phaseCount(DecodeTaskPhase.ACCEPTED_NOT_RUNNING);
        }

        public int runningCount() {
            return phaseCount(DecodeTaskPhase.RUNNING);
        }

        public int engineCapacityUsed() {
            return routing.engineCapacityUsed();
        }

        public boolean isQueued(long requestId) {
            DecodeRequestView request = reserved.get(requestId);
            return request != null && request.queued();
        }

        private int phaseCount(DecodeTaskPhase phase) {
            int count = 0;
            for (DecodeRequestView task : confirmed) {
                if (task.phase() == phase) {
                    count++;
                }
            }
            return count;
        }
    }

    public record DecodeRoutingView(
            String address,
            long generationId,
            WorkerStatus.TopologySnapshot topology,
            WorkerStatus.CommittedWorkerStatus workerStatus,
            long admissionVersion,
            int totalLoad,
            int engineLoad,
            CapacityUsage placementUsage,
            CapacityUsage dispatchUsage,
            long inflightHardKv,
            long inflightExpectedKv) {

        public int engineCapacityUsed() { return Math.toIntExact(dispatchUsage.occupiedRequests()); }
        public long realKvUsed() { return placementUsage.expectedKvUsed(); }
        public long realKvAvailable() { return placementUsage.hardKvAvailable(); }
        public long engineFacingKvUsed() { return dispatchUsage.expectedKvUsed(); }
        public long engineFacingKvAvailable() { return dispatchUsage.hardKvAvailable(); }
        public long totalKv() { return placementUsage.totalKvTokens(); }

        public DecodeRoutingView {
            java.util.Objects.requireNonNull(address, "address");
            java.util.Objects.requireNonNull(topology, "topology");
            java.util.Objects.requireNonNull(workerStatus, "workerStatus");
            if (generationId <= 0L) {
                throw new IllegalArgumentException(
                        "Decode routing view requires a positive generation");
            }
        }
    }

    public record DecodeRequestView(long requestId,
                                    int priority,
                                    long kvTokens,
                                    long expectedKvTokens,
                                    DecodeTaskPhase phase,
                                    boolean priorityKnown,
                                    long reservationToken,
                                    boolean queued,
                                    boolean claimedForPreemption) {
        public CapacityRelease placementRelease() {
            return new CapacityRelease(1L, kvTokens, expectedKvTokens);
        }
    }

    public record AdmissionCapacity(
            long maxEngineRequests,
            long maxKvUsagePercent) {

        public AdmissionCapacity {
            if (maxEngineRequests < 0L || maxKvUsagePercent < 0L
                    || maxKvUsagePercent > RoutingConfig.PERCENTAGE_SCALE) {
                throw new IllegalArgumentException(
                        "Decode admission limits are outside their domain");
            }
        }

        public CapacityDeficit evaluate(CapacityUsage usage, long hardKvTokens, long expectedKvTokens) {
            return evaluate(usage, hardKvTokens, expectedKvTokens, CapacityRelease.NONE);
        }

        /** Use the same occupancy scope for the observation and every exact victim release. */
        public CapacityDeficit evaluate(CapacityUsage usage, long hardKvTokens, long expectedKvTokens,
                                        CapacityRelease release) {
            java.util.Objects.requireNonNull(usage, "usage");
            java.util.Objects.requireNonNull(release, "release");
            if (hardKvTokens < 0L || expectedKvTokens < hardKvTokens) {
                throw new IllegalArgumentException("Decode demand must satisfy expected >= hard >= 0");
            }
            long requests = maxEngineRequests == 0L ? 0L
                    : shortfall(Math.max(0L, usage.occupiedRequests - release.requests),
                            1L, maxEngineRequests, 0L);
            if (usage.totalKvTokens == 0L && maxKvUsagePercent > 0L) {
                return new CapacityDeficit(requests, 0L, 0L);
            }
            return new CapacityDeficit(requests,
                    shortfall(usage.hardReservedKvTokens, hardKvTokens,
                            usage.availableKvTokens, release.hardKvTokens),
                    shortfall(Math.max(0L, usage.expectedKvUsed - release.expectedKvTokens),
                            expectedKvTokens, kvBudget(usage.totalKvTokens), 0L));
        }

        /** Integer arithmetic never rounds the configured KV budget up. */
        public long kvBudget(long totalKv) {
            if (totalKv < 0L) {
                throw new IllegalArgumentException("negative KV capacity");
            }
            return totalKv / 100L * maxKvUsagePercent + totalKv % 100L * maxKvUsagePercent / 100L;
        }

        private static long shortfall(long used, long incoming, long capacity, long released) {
            long remainingUsed = Math.max(0L, used - released);
            long remainingCapacity = DecodeState.saturatedAddNonNegative(capacity, Math.max(0L, released - used));
            return remainingUsed > remainingCapacity
                    ? DecodeState.saturatedAddNonNegative(remainingUsed - remainingCapacity, incoming)
                    : Math.max(0L, incoming - (remainingCapacity - remainingUsed));
        }
    }

    public record CapacityUsage(long occupiedRequests, long totalKvTokens, long availableKvTokens,
                                long hardReservedKvTokens, long expectedKvUsed) {
        public CapacityUsage {
            if (occupiedRequests < 0L || totalKvTokens < 0L || availableKvTokens < 0L
                    || hardReservedKvTokens < 0L || expectedKvUsed < 0L) {
                throw new IllegalArgumentException("Decode occupancy must be non-negative");
            }
        }

        public long hardKvAvailable() {
            return Math.max(0L, availableKvTokens - hardReservedKvTokens);
        }
    }

    public record CapacityRelease(long requests, long hardKvTokens, long expectedKvTokens) {
        public static final CapacityRelease NONE = new CapacityRelease(0L, 0L, 0L);

        public CapacityRelease {
            if (requests < 0L || hardKvTokens < 0L || expectedKvTokens < hardKvTokens) {
                throw new IllegalArgumentException("invalid Decode capacity release");
            }
        }

        public CapacityRelease plus(CapacityRelease other) {
            return new CapacityRelease(DecodeState.saturatedAddNonNegative(requests, other.requests),
                    DecodeState.saturatedAddNonNegative(hardKvTokens, other.hardKvTokens),
                    DecodeState.saturatedAddNonNegative(expectedKvTokens, other.expectedKvTokens));
        }
    }

    public record CapacityDeficit(long requests, long hardKvTokens, long expectedKvTokens) {
        public boolean fits() { return requests == 0L && !needsKv(); }
        public boolean needsKv() { return hardKvTokens > 0L || expectedKvTokens > 0L; }
        public long kvTokens() { return Math.max(hardKvTokens, expectedKvTokens); }
    }

    // Retirement and orphan cleanup.

    public boolean isRetired() { return isGenerationRetiringOrRetired(); }

    protected void closeEndpoint() {
        List<ReservationHandle> reservations = state.retire();
        try { endpointEvents.onDecodeGenerationRetired(this, reservations); }
        finally { notifyEngineDispatchCapacityListeners(); }
    }

    public int evictExpiredRequests(long ttlMs, LongPredicate retainForSchedulerCleanup) {
        DecodeState.CleanupResult result = state.evictExpiredRequests(ttlMs, retainForSchedulerCleanup);
        if (result.capacityReleased()) { publishCapacityRelease(); }
        return result.expiredReservations();
    }

    // Metrics and shared lock-free capacity notifications.

    public void reportBatchMetrics(BatchSchedulerReporter reporter) {
        DecodeState.Stats stats = state.stats();
        reporter.reportInflightRequestCount(RoleType.DECODE.name(), getIp(), stats.inflight());
        reporter.reportDecodeTotalLoad(getIp(), stats.totalLoad());
        reporter.reportDecodeInflightKvReserved(getIp(), stats.expectedKv());
        reporter.reportDecodeInflightHardKvReserved(getIp(), stats.hardKv());
        reporter.reportInflightMaxAgeMs(RoleType.DECODE.name(), getIp(), stats.oldestAgeMs());
    }

    public void reportAdmissionMetrics(RequestSchedulerReporter reporter) {
        LayeredAdmissionView view = resourceSnapshot();
        String endpoint = ipPort();
        reporter.reportDecodeReservedCount(endpoint, view.reserved().size());
        reporter.reportDecodeShadowKvReserved(
                endpoint, view.routing().inflightHardKv());
        reporter.reportDecodeRunningCount(endpoint, view.runningCount());
        reporter.reportDecodeAcceptedCount(endpoint, view.acceptedCount());
        reporter.reportDecodeEngineLoad(endpoint, view.routing().engineLoad());
    }

    private void publishCapacityRelease() {
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
        placementAvailability.capacityChanged(
                RoleType.DECODE, topology.group(), ipPort());
    }
}
