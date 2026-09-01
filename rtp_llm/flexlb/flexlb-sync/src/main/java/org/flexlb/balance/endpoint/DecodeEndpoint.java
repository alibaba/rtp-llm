package org.flexlb.balance.endpoint;

import org.flexlb.balance.execution.TtlEvictor;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.DecodeTaskPhase;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
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
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.LongPredicate;

/**
 * Decode-side endpoint with Auto-TPM shadow admission accounting.
 *
 * <p><b>Layered view (Phase 5):</b> {@code inflightRequests} only ever holds
 * shadow entries; engine-confirmed requests are
 * folded into {@code confirmedEngineOwnedCount} by calibrate exactly as in
 * Phase 4 (accounting unchanged), and are additionally tracked per-request in
 * {@link #trackedConfirmed} split by phase — {@code KV_ALLOCATED} →
 * {@code ACCEPTED_NOT_RUNNING} layer, {@code RUNNING} → {@code RUNNING}
 * layer — for accepted-eviction planning and layered gauges.
 * {@code totalLoad = confirmedEngineOwnedCount + reserved inflight count}.
 *
 * <p><b>Known accepted cost:</b> the admission lock and version bump on every
 * reserve/release/calibrate stay active even when Auto-TPM is disabled; the
 * uncontended ReentrantLock + AtomicLong overhead is negligible and keeping it
 * unconditional avoids divergent code paths (task10 P2-9, no structural change).
 */
public class DecodeEndpoint extends WorkerEndpoint {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");
    private static final Comparator<ReservationHandle> RETIREMENT_ORDER =
            Comparator.comparingLong(ReservationHandle::endpointGenerationId)
                    .thenComparingLong(ReservationHandle::requestId)
                    .thenComparingLong(ReservationHandle::reservationToken);

    private final EndpointEventSink endpointEventSink;
    private final PlacementAvailability placementAvailability;
    private final ConcurrentHashMap<Long, RequestInflight> inflightRequests = new ConcurrentHashMap<>();
    /** Reservation tokens already transferred beyond Master rollback. Guarded by admissionLock. */
    private final Set<Long> engineLifecycleReservationTokens = new HashSet<>();
    private final AtomicLong inflightKvReservedTotal = new AtomicLong(0);
    private final AtomicLong inflightExpectedKvReservedTotal = new AtomicLong(0);
    /**
     * Engine-confirmed request ownership: KV_ALLOCATED plus RUNNING, including
     * synthetic fenced slots. This is deliberately not the Engine's physical
     * running concurrency; {@link #trackedConfirmed} provides the phase split.
     */
    private volatile int confirmedEngineOwnedCount;
    private final TtlEvictor<Long, RequestInflight> requestEvictor;

    /**
     * Layered registry of engine-confirmed requests (Phase 5): requestId →
     * accepted/running membership. Rebuilt against every calibrate report;
     * carries no shadow accounting — confirmed KV is engine-reported and the
     * slot count stays in {@code confirmedEngineOwnedCount}, so this registry is
     * pure metadata for eviction planning, cancel dedup and layered gauges.
     */
    private final ConcurrentHashMap<Long, ConfirmedTask> trackedConfirmed = new ConcurrentHashMap<>();

    /**
     * Token-fenced priority-preemption ownership.  Victim accounting remains
     * in its original layer until a typed Prefill WorkerStatus CANCELED event
     * is settled; an ACCEPTED Cancel response only advances the claim state.
     * All access is under {@link #admissionLock}.
     */
    private final Map<Long, PreemptionClaim> preemptionClaims = new HashMap<>();
    private final Map<Long, EndpointPreemptionAttempt> preemptionAttempts = new HashMap<>();

    /** KV that Decode has reported free but the Prefill CANCELED fence has not settled yet. */
    private final AtomicLong priorityPreemptionHeldKv = new AtomicLong();
    /** Expected-demand counterpart; invariant: expected hold >= hard hold >= 0. */
    private final AtomicLong priorityPreemptionHeldExpectedKv = new AtomicLong();

    /**
     * Request-scoped ownership retained by a generic scheduler EngineFence.
     *
     * <p>All map and entry mutations are serialized by {@link #admissionLock}.
     * A protection initially pins an existing shadow or confirmed owner. If a
     * confirmed request temporarily disappears from WorkerStatus, ownership is
     * transferred to a synthetic slot plus local KV hold until the exact fence
     * clears or an ordinary authoritative terminal arrives. The two atomic KV
     * totals are read lock-free by routing; they do not own entry lifecycle.
     */
    private final Map<Long, EngineFenceProtection> engineFenceProtections = new HashMap<>();
    private final AtomicLong engineFenceHeldKv = new AtomicLong();
    private final AtomicLong engineFenceHeldExpectedKv = new AtomicLong();
    /** Number of generic synthetic slots; guarded by {@link #admissionLock}. */
    private int engineFenceHeldSlotCount;

    /**
     * Request-id fence against stale WorkerStatus resurrecting any
     * authoritatively settled request: requestId -> settlement time.
     *
     * <p>Only generations settled through a priority/generic EngineFence or an
     * explicit Engine TOMBSTONED proof publish into this registry. Ordinary
     * WorkerStatus completion needs no retained fence: status calibration is
     * serialized, and retaining every completion would make this map grow with
     * request throughput rather than with the small number of ambiguous
     * generations. The endpoint inflight TTL bounds retained fence lifetime;
     * request ids must not be reused inside that reconciliation window because
     * WorkerStatus does not carry a dispatch generation.
     */
    private final Map<Long, Long> settledTombstones = new HashMap<>();

    /**
     * Reserved entries whose request is still sitting in a prefill queue —
     * committed by the scheduler but not yet dispatched to the engine (N2,
     * plan-commit redesign). These reservations keep protecting KV against
     * oversell, but must not count against the decode concurrency limit:
     * counting them produced the shadow-saturation 8400 storm (root cause C —
     * queued reservations saturating {@code getTotalLoad()} while the engine
     * sat idle). Queue schedulers mark an exact reservation before queue
     * publication and unmark through an acquired
     * {@link EngineDispatchPermit}; release/calibrate prune it
     * alongside {@code inflightRequests}. DIRECT paths never mark, so their
     * accounting is unchanged.
     */
    /**
     * O(1) mirror of queued reservations in {@link #inflightRequests} (PR-C):
     * incremented when a reservation is marked queued and decremented when
     * it is dispatched / released / calibrated out, so {@link #getEngineLoad}
     * avoids the per-call O(n) scan of the former full-scan formula. Read lock-free;
     * written under {@link #admissionLock}.
     */
    private final AtomicInteger queuedPhaseCount = new AtomicInteger(0);

    /** Prompt-only KV held by reservations which are still Prefill-queued. */
    private final AtomicLong queuedHardKvReservedTotal = new AtomicLong(0);

    /** Expected KV held by reservations which are still Prefill-queued. */
    private final AtomicLong queuedExpectedKvReservedTotal = new AtomicLong(0);

    /**
     * Decode slots reserved before a queued request is irreversibly exposed to
     * its Prefill/Decode delivery path. The request remains in
     * Their reservation owner is marked queued, so {@link #getEngineLoad()}
     * continues to describe
     * only engine-facing work. Capacity acquisition instead uses
     * {@code getEngineLoad() + activeEngineDispatchPermitCount} under
     * {@link #admissionLock}.
     *
     * <p>Each entry carries both the immutable reservation identity and a
     * monotonic token. The pair fences request-id reuse while the slot is
     * reserved. A successful commit removes the lease permanently: committed
     * Decode ownership is never rolled back by a delivery token.
     */
    /** Hard prompt KV already committed to acquired pre-delivery permits. */
    private final AtomicLong engineDispatchPermitHardKvReservedTotal = new AtomicLong();
    /** Expected KV already committed to acquired pre-delivery permits. */
    private final AtomicLong engineDispatchPermitExpectedKvReservedTotal = new AtomicLong();
    /** Permit tokens invalidated specifically by this endpoint generation retiring. */
    /** Guarded by admissionLock; replaced atomically by retirement commit. */
    private Set<Long> retiredEngineDispatchPermitTokens = new HashSet<>();
    /** Mutated under admissionLock; volatile for the lock-free waiter predicate. */
    private volatile int activeEngineDispatchPermitCount;
    /** Guarded by {@link #admissionLock}; zero is never issued. */
    private long nextReservationToken = 1L;
    /** Guarded by {@link #admissionLock}; zero is never issued. */
    private long nextEngineDispatchPermitToken = 1L;
    /**
     * Prefill workers currently routing to this Decode endpoint. Listeners are
     * invoked only after dropping {@link #admissionLock}.
     */
    private final Set<Runnable> engineDispatchCapacityListeners =
            ConcurrentHashMap.newKeySet();

    /**
     * Serializes reserve/release, dispatch-permit, calibration, and eviction
     * transactions. Reads stay lock-free.
     */
    private final ReentrantLock admissionLock = new ReentrantLock();

    /** Monotonic diagnostic generation for the layered admission projection. */
    private final AtomicLong admissionVersion = new AtomicLong();

    /**
     * Generation-local, non-authoritative cache for bulk routing traversal.
     *
     * <p>Both key components are required: local admission mutations advance
     * {@link #admissionVersion}, while a committed Engine observation is
     * replaced by holder identity. The cache owns no endpoint or registry-map
     * reference and therefore disappears with this endpoint generation.</p>
     */
    private volatile RoutingViewCache routingViewCache;

    private record RoutingViewCache(
            WorkerStatus.CommittedWorkerStatus workerStatus,
            long admissionVersion,
            DecodeRoutingView routing) {
    }

    public DecodeEndpoint(
            WorkerStatus status,
            EndpointEventSink endpointEventSink) {
        this(status, endpointEventSink, new PlacementAvailability());
    }

    DecodeEndpoint(
            WorkerStatus status,
            EndpointEventSink endpointEventSink,
            PlacementAvailability placementAvailability) {
        super(status);
        this.endpointEventSink = java.util.Objects.requireNonNull(
                endpointEventSink, "endpointEventSink");
        this.placementAvailability = java.util.Objects.requireNonNull(
                placementAvailability, "placementAvailability");
        this.requestEvictor = TtlEvictor.withKeyCallback(
                inflightRequests, (requestId, req) -> {
                    removeEngineDispatchPermitLocked(req);
                    engineLifecycleReservationTokens.remove(
                            req.reservationToken());
                    removeQueuedPhaseLocked(requestId, req);
                    inflightKvReservedTotal.addAndGet(-req.kvTokens());
                    inflightExpectedKvReservedTotal.addAndGet(-req.expectedKvTokens());
                });
    }

    /**
     * Immutable identity of one exact shadow reservation.
     *
     * <p>The endpoint generation fences same-address replacement; the token
     * fences request-id reuse inside one generation. Both are required when a
     * route result is rebound to an endpoint after selection.</p>
     */
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

    /** Result of settling one transport-level rejection for an exact reservation. */
    public enum DispatchRejectionSettlement {
        RELEASED,
        ENGINE_ACCEPTED,
        STALE,
        CONFLICT
    }

    /** Immutable hard-capacity policy for one exact eviction transaction. */
    public record AdmissionCapacity(
            long maxEngineRequests,
            long maxKvUsagePercent) {

        public AdmissionCapacity {
            if (maxEngineRequests < 0L || maxKvUsagePercent < 0L
                    || maxKvUsagePercent > 100L) {
                throw new IllegalArgumentException(
                        "Decode admission limits are outside their domain");
            }
        }
    }

    /** Immutable demand projected by both routing and exact dispatch admission. */
    public record EngineDispatchDemand(
            long hardKvTokens,
            long expectedKvTokens) {

        public EngineDispatchDemand {
            if (hardKvTokens < 0L || expectedKvTokens < hardKvTokens) {
                throw new IllegalArgumentException(
                        "Decode dispatch demand must satisfy expected >= hard >= 0");
            }
        }
    }

    /** One exact fact interpreted by canonical Decode accounting. */
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

    /** Exact RequestSlot facts emitted after the Decode lock is released. */
    public record StatusReduction(
            DecodeEndpoint source,
            List<WorkerStatusFact> facts)
            implements EndpointStatusReduction {
        public StatusReduction {
            facts = List.copyOf(facts);
        }
    }

    /**
     * Exact ownership retained while an asynchronous Engine fence is pending.
     *
     * <p>The endpoint, reservation and protection object together form the
     * capability. Closing an old lease cannot clear a protection installed for
     * a reused request id.</p>
     */
    public static final class EngineFenceLease implements AutoCloseable {
        private final DecodeEndpoint endpoint;
        private final ReservationHandle reservation;
        private final EngineFenceProtection protection;

        private EngineFenceLease(
                DecodeEndpoint endpoint,
                ReservationHandle reservation,
                EngineFenceProtection protection) {
            this.endpoint = endpoint;
            this.reservation = reservation;
            this.protection = protection;
        }

        /**
         * Convert the exact fence owner into an opaque terminal capability.
         * The caller may do this only after validating an authoritative Engine
         * terminal response; the endpoint revalidates the exact owner on use.
         */
        public AuthoritativeTerminalProof authoritativeTerminalProof() {
            return new AuthoritativeTerminalProof(
                    endpoint, reservation, protection,
                    AuthoritativeTerminalOwner.ENGINE_FENCE);
        }

        @Override
        public void close() {
            endpoint.closeEngineFenceExact(this);
        }
    }

    /**
     * Opaque proof bound to one exact Decode reservation and Engine-fence
     * owner. It cannot be constructed from a request id or by arbitrary
     * callers; only the exact {@link EngineFenceLease} can issue it.
     */
    public static final class AuthoritativeTerminalProof {
        private final DecodeEndpoint endpoint;
        private final ReservationHandle reservation;
        private final EngineFenceProtection protection;
        private final AuthoritativeTerminalOwner owner;

        private AuthoritativeTerminalProof(
                DecodeEndpoint endpoint,
                ReservationHandle reservation,
                EngineFenceProtection protection,
                AuthoritativeTerminalOwner owner) {
            this.endpoint = endpoint;
            this.reservation = reservation;
            this.protection = protection;
            this.owner = owner;
        }
    }

    private enum AuthoritativeTerminalOwner {
        ENGINE_FENCE,
        DISPATCH_REJECTION,
        WORKER_STATUS
    }

    /** Stop this exact endpoint generation and retire all generation-local ownership. */
    @Override
    protected void closeEndpoint() {
        List<ReservationHandle> retiredReservations;
        admissionLock.lock();
        try {
            Set<Long> plannedRetiredPermitTokens = new HashSet<>(
                    retiredEngineDispatchPermitTokens);
            for (RequestInflight reservation : inflightRequests.values()) {
                EngineDispatchPermit permit = reservation.dispatchPermit();
                if (permit != null) {
                    plannedRetiredPermitTokens.add(permit.token);
                }
            }
            retiredReservations = retireGenerationOwnershipLocked(
                    plannedRetiredPermitTokens);
            admissionVersion.incrementAndGet();
        } finally {
            admissionLock.unlock();
        }
        try {
            endpointEventSink.onDecodeGenerationRetired(
                    this, retiredReservations);
        } finally {
            notifyEngineDispatchCapacityListeners();
        }
    }

    /** Materialize every exact owner, then atomically drain this generation. */
    private List<ReservationHandle> retireGenerationOwnershipLocked(
                    Set<Long> plannedRetiredPermitTokens) {
        Set<ReservationHandle> owners = new HashSet<>();
        long generationId = getStatus().getGenerationId();

        inflightRequests.forEach((requestId, reservation) ->
                addRetiredOwner(
                        owners, generationId, requestId,
                        reservation.reservationToken()));
        trackedConfirmed.forEach((requestId, confirmed) ->
                addRetiredOwner(
                        owners, generationId, requestId,
                        confirmed.reservationToken()));
        engineFenceProtections.forEach((requestId, protection) ->
                addRetiredOwner(
                        owners, generationId, requestId,
                        protection.reservationToken));
        preemptionClaims.forEach((requestId, claim) ->
                addRetiredOwner(
                        owners, generationId, requestId,
                        claim.reservationToken));
        Set<Long> mappedReservationTokens = new HashSet<>();
        for (RequestInflight reservation : inflightRequests.values()) {
            mappedReservationTokens.add(reservation.reservationToken());
        }
        for (long engineOwnedToken : engineLifecycleReservationTokens) {
            if (!mappedReservationTokens.contains(engineOwnedToken)) {
                logger.error(
                        "Decode retirement found an engine-lifecycle marker without a request owner: "
                                + "generation={} reservation_token={}",
                        generationId, engineOwnedToken);
            }
        }
        for (EndpointPreemptionAttempt attempt : preemptionAttempts.values()) {
            addRetiredOwner(
                    owners, generationId,
                    attempt.incomingRequestId,
                    attempt.incomingReservationToken);
            for (ReservationHandle victim : attempt.remainingVictims.values()) {
                if (victim.endpointGenerationId() != generationId) {
                    logger.error(
                            "Decode retirement ignored a priority victim from another generation: "
                                    + "request_id={} expected_generation={} actual_generation={}",
                            victim.requestId(), generationId,
                            victim.endpointGenerationId());
                    continue;
                }
                owners.add(victim);
            }
        }

        List<ReservationHandle> ordered = new ArrayList<>(owners);
        ordered.sort(RETIREMENT_ORDER);
        List<ReservationHandle> retiredReservations = List.copyOf(ordered);

        // Canonical retirement commit. The immutable owner list and every
        // exact ReservationHandle have been validated above this line.
        retiredEngineDispatchPermitTokens = plannedRetiredPermitTokens;
        inflightRequests.values().forEach(RequestInflight::clearDispatchPermit);
        activeEngineDispatchPermitCount = 0;
        engineDispatchPermitHardKvReservedTotal.set(0L);
        engineDispatchPermitExpectedKvReservedTotal.set(0L);

        inflightRequests.clear();
        engineLifecycleReservationTokens.clear();
        inflightKvReservedTotal.set(0L);
        inflightExpectedKvReservedTotal.set(0L);

        trackedConfirmed.clear();
        confirmedEngineOwnedCount = 0;

        queuedPhaseCount.set(0);
        queuedHardKvReservedTotal.set(0L);
        queuedExpectedKvReservedTotal.set(0L);

        preemptionClaims.clear();
        preemptionAttempts.clear();
        priorityPreemptionHeldKv.set(0L);
        priorityPreemptionHeldExpectedKv.set(0L);

        engineFenceProtections.clear();
        engineFenceHeldKv.set(0L);
        engineFenceHeldExpectedKv.set(0L);
        engineFenceHeldSlotCount = 0;

        settledTombstones.clear();
        return retiredReservations;
    }

    private static void addRetiredOwner(
            Set<ReservationHandle> owners,
            long generationId,
            long requestId,
            long reservationToken) {
        if (reservationToken > 0L) {
            owners.add(new ReservationHandle(
                    generationId, requestId, reservationToken));
        }
    }

    /** A blocked worker treats retirement as a wakeup and retries for a typed failure. */
    public boolean isRetired() {
        return isGenerationRetiringOrRetired();
    }

    /** Reserve through the exact route pin captured before endpoint detach. */
    public ReservationHandle reservePinned(
            GenerationPin pin,
            long requestId,
            long kvTokens,
            long expectedKvTokens,
            int priority) {
        admissionLock.lock();
        try {
            requirePinnedGeneration(pin);
            return reserveLocked(
                    requestId, kvTokens, expectedKvTokens, priority);
        } finally {
            admissionLock.unlock();
        }
    }

    /** Caller holds admissionLock. */
    private ReservationHandle reserveLocked(long requestId,
                                            long kvTokens,
                                            long expectedKvTokens,
                                            int priority) {
        if (!requestIdAvailableForReservationLocked(requestId)) {
            throw new IllegalStateException(
                    "Decode request id is still owned by this endpoint generation: "
                            + requestId);
        }
        long reservationToken = nextReservationTokenLocked();
        RequestInflight newReservation =
                new RequestInflight(
                        kvTokens, expectedKvTokens, priority, reservationToken);
        if (inflightRequests.putIfAbsent(requestId, newReservation) != null) {
            throw new IllegalStateException(
                    "Decode reservation appeared while admissionLock was held: "
                            + requestId);
        }
        inflightKvReservedTotal.addAndGet(kvTokens);
        inflightExpectedKvReservedTotal.addAndGet(expectedKvTokens);
        admissionVersion.incrementAndGet();
        return new ReservationHandle(
                getStatus().getGenerationId(), requestId, reservationToken);
    }

    /** Reserve queued ownership through the exact route pin. */
    public ReservationHandle reserveQueuedPinned(
            GenerationPin pin,
            long requestId,
            long kvTokens,
            long expectedKvTokens,
            int priority) {
        admissionLock.lock();
        try {
            requirePinnedGeneration(pin);
            ReservationHandle reservation = reserveLocked(
                    requestId, kvTokens, expectedKvTokens, priority);
            addQueuedPhaseLocked(requestId, inflightRequests.get(requestId));
            return reservation;
        } finally {
            admissionLock.unlock();
        }
    }

    /**
     * Acquire a soft queued hold for placement, or report that this request id
     * is still fenced by the exact endpoint generation. The fence is a
     * transient placement blocker: WorkerStatus cannot distinguish a reused
     * request id until its settlement tombstone expires.
     */
    public ReservationHandle tryReserveQueuedPinned(
            GenerationPin pin,
            long requestId,
            long kvTokens,
            long expectedKvTokens,
            int priority) {
        admissionLock.lock();
        try {
            requirePinnedGeneration(pin);
            if (!requestIdAvailableForReservationLocked(requestId)) {
                return null;
            }
            ReservationHandle reservation = reserveLocked(
                    requestId, kvTokens, expectedKvTokens, priority);
            addQueuedPhaseLocked(requestId, inflightRequests.get(requestId));
            return reservation;
        } finally {
            admissionLock.unlock();
        }
    }

    /**
     * Acquire queued ownership only if this exact generation can still accept
     * the request. A null result mutates nothing.
     */
    public ReservationHandle tryReserveQueuedPinned(
            GenerationPin pin,
            long requestId,
            long kvTokens,
            long expectedKvTokens,
            int priority,
            AdmissionCapacity capacity) {
        java.util.Objects.requireNonNull(capacity, "capacity");
        admissionLock.lock();
        try {
            requirePinnedGeneration(pin);
            if (!requestIdAvailableForReservationLocked(requestId)) {
                return null;
            }
            if (queuedPlacementIsFullLocked(
                    kvTokens, expectedKvTokens, capacity)) {
                return null;
            }
            ReservationHandle reservation = reserveLocked(
                    requestId, kvTokens, expectedKvTokens, priority);
            addQueuedPhaseLocked(requestId, inflightRequests.get(requestId));
            return reservation;
        } finally {
            admissionLock.unlock();
        }
    }

    /** Remove every local owner for one request. Caller holds admissionLock. */
    private boolean releaseLocked(long requestId) {
        boolean changed = removeEngineDispatchPermitLocked(requestId);
        changed = clearEngineFenceProtectionLocked(requestId) || changed;
        RequestInflight removed = inflightRequests.remove(requestId);
        removeEngineLifecycleReservationLocked(removed);
        changed = removeQueuedPhaseLocked(requestId, removed) || changed;
        if (removed != null) {
            inflightKvReservedTotal.addAndGet(-removed.kvTokens());
            inflightExpectedKvReservedTotal.addAndGet(-removed.expectedKvTokens());
            changed = true;
        }
        if (changed) {
            admissionVersion.incrementAndGet();
        }
        return changed;
    }

    /** Register a stable worker wakeup for Decode dispatch-capacity changes. */
    public void addEngineDispatchCapacityListener(Runnable listener) {
        if (listener != null) {
            engineDispatchCapacityListeners.add(listener);
        }
    }

    /** Remove a worker wakeup when its Prefill batcher shuts down. */
    public void removeEngineDispatchCapacityListener(Runnable listener) {
        if (listener != null) {
            engineDispatchCapacityListeners.remove(listener);
        }
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

    /**
     * Roll back one exact reservation that never crossed into Engine ownership.
     *
     * <p>A missing or replaced handle is an idempotent success. If the exact
     * identity is still present but has an Engine lifecycle, generic fence or
     * priority-protocol owner, local rollback is a caller invariant violation:
     * that owner must consume an authoritative terminal instead.</p>
     */
    public void rollbackExact(ReservationHandle reservation) {
        if (releaseLocalReservationExact(reservation)) {
            signalPlacementCapacityChanged();
        }
    }

    /** Release capacity owned by a published placement and wake that domain. */
    public void releasePlacementExact(ReservationHandle reservation) {
        if (releaseLocalReservationExact(reservation)) {
            signalPlacementCapacityChanged();
        }
    }

    private boolean releaseLocalReservationExact(
            ReservationHandle reservation) {
        if (reservation == null) {
            throw new IllegalArgumentException(
                    "Decode reservation is required for rollback");
        }
        if (reservation.endpointGenerationId()
                != getStatus().getGenerationId()) {
            return false;
        }
        boolean capacityChanged = false;
        admissionLock.lock();
        try {
            long requestId = reservation.requestId();
            RequestInflight current = inflightRequests.get(
                    requestId);
            boolean exactShadow = isExactReservation(current, reservation);
            ConfirmedTask confirmed = trackedConfirmed.get(requestId);
            EngineFenceProtection protection = engineFenceProtections.get(requestId);
            PreemptionClaim claim = preemptionClaims.get(requestId);
            EngineDispatchPermit dispatchPermit = current == null
                    ? null : current.dispatchPermit();

            boolean exactConfirmed = confirmed != null
                    && confirmed.reservationToken()
                            == reservation.reservationToken();
            boolean exactProtection = protection != null
                    && protection.reservationToken
                            == reservation.reservationToken();
            boolean exactClaim = claim != null
                    && claim.reservationToken
                            == reservation.reservationToken();
            boolean exactAttempt = hasExactIncomingAttemptLocked(reservation);
            boolean exactDispatchPermit = dispatchPermit != null
                    && isExactReservation(
                            dispatchPermit.reservation, reservation);
            boolean exactEngineLifecycle =
                    hasEngineLifecycleReservationExactLocked(reservation);

            if (!exactShadow) {
                if (exactConfirmed || exactProtection || exactClaim
                        || exactAttempt || exactDispatchPermit
                        || exactEngineLifecycle) {
                    throw rollbackInvariant(reservation,
                            "exact ownership already crossed local shadow rollback");
                }
                return false;
            }
            if (exactEngineLifecycle
                    || exactConfirmed || exactProtection
                    || exactClaim || exactAttempt) {
                throw rollbackInvariant(reservation,
                        "exact ownership is held by Engine/protocol lifecycle");
            }

            if (exactDispatchPermit) {
                removeEngineDispatchPermitLocked(requestId);
            } else if (dispatchPermit != null) {
                throw rollbackInvariant(reservation,
                        "request id has a dispatch permit for another reservation");
            }
            if (!inflightRequests.remove(requestId, current)) {
                throw rollbackInvariant(reservation,
                        "exact shadow changed while admissionLock was held");
            }
            removeEngineLifecycleReservationLocked(current);
            removeQueuedPhaseLocked(requestId, current);
            inflightKvReservedTotal.addAndGet(-current.kvTokens());
            inflightExpectedKvReservedTotal.addAndGet(
                    -current.expectedKvTokens());
            admissionVersion.incrementAndGet();
            capacityChanged = true;
        } finally {
            admissionLock.unlock();
        }
        if (capacityChanged) {
            notifyEngineDispatchCapacityListeners();
        }
        return capacityChanged;
    }

    /**
     * Total counterpart cleanup for a terminal observed on another role. Only
     * an exact local shadow may be removed; Engine/protocol ownership makes the
     * operation a no-op and remains with its canonical reducer.
     */
    public boolean releaseLocalShadowIfExact(ReservationHandle reservation) {
        if (reservation == null
                || reservation.endpointGenerationId()
                        != getStatus().getGenerationId()) {
            return false;
        }
        boolean released = false;
        admissionLock.lock();
        try {
            long requestId = reservation.requestId();
            RequestInflight shadow = inflightRequests.get(requestId);
            if (!isExactReservation(shadow, reservation)
                    || hasEngineLifecycleReservationExactLocked(reservation)) {
                return false;
            }
            ConfirmedTask confirmed = trackedConfirmed.get(requestId);
            EngineFenceProtection protection = engineFenceProtections.get(requestId);
            PreemptionClaim claim = preemptionClaims.get(requestId);
            if ((confirmed != null
                        && confirmed.reservationToken()
                                == reservation.reservationToken())
                    || (protection != null
                        && protection.reservationToken
                                == reservation.reservationToken())
                    || (claim != null
                        && claim.reservationToken
                                == reservation.reservationToken())
                    || hasExactIncomingAttemptLocked(reservation)) {
                return false;
            }
            EngineDispatchPermit dispatchPermit = shadow.dispatchPermit();
            if (dispatchPermit != null
                    && !isExactReservation(
                            dispatchPermit.reservation, reservation)) {
                return false;
            }
            if (dispatchPermit != null) {
                removeEngineDispatchPermitLocked(requestId);
            }
            if (!inflightRequests.remove(requestId, shadow)) {
                return false;
            }
            removeEngineLifecycleReservationLocked(shadow);
            removeQueuedPhaseLocked(requestId, shadow);
            inflightKvReservedTotal.addAndGet(-shadow.kvTokens());
            inflightExpectedKvReservedTotal.addAndGet(
                    -shadow.expectedKvTokens());
            if (!hasAnyRequestOwnerLocked(requestId)) {
                rememberSettledLocked(requestId, System.currentTimeMillis());
            }
            admissionVersion.incrementAndGet();
            released = true;
        } finally {
            admissionLock.unlock();
        }
        if (released) {
            notifyEngineDispatchCapacityListeners();
            signalPlacementCapacityChanged();
        }
        return released;
    }

    /** Caller holds {@link #admissionLock}. */
    private boolean hasExactIncomingAttemptLocked(
            ReservationHandle reservation) {
        for (EndpointPreemptionAttempt attempt : preemptionAttempts.values()) {
            if (attempt.incomingRequestId == reservation.requestId()
                    && attempt.incomingReservationToken
                            == reservation.reservationToken()) {
                return true;
            }
        }
        return false;
    }

    private static boolean isExactReservation(
            RequestInflight current,
            ReservationHandle reservation) {
        return current != null
                && current.reservationToken()
                        == reservation.reservationToken();
    }

    /** Caller holds {@link #admissionLock}. */
    private boolean hasEngineLifecycleReservationExactLocked(
            ReservationHandle reservation) {
        return engineLifecycleReservationTokens.contains(
                reservation.reservationToken());
    }

    private static IllegalStateException rollbackInvariant(
            ReservationHandle reservation,
            String detail) {
        return new IllegalStateException(
                "Illegal Decode local rollback: requestId="
                        + reservation.requestId()
                        + ", reservationToken="
                        + reservation.reservationToken()
                        + ", detail=" + detail);
    }

    /** Snapshot the exact current identity for an internally created owner. */
    public ReservationHandle reservationHandle(long requestId) {
        admissionLock.lock();
        try {
            if (isGenerationRetiringOrRetired()) {
                return null;
            }
            RequestInflight current = inflightRequests.get(requestId);
            if (current == null || current.reservationToken() <= 0L) {
                return null;
            }
            return new ReservationHandle(
                    getStatus().getGenerationId(),
                    requestId,
                    current.reservationToken());
        } finally {
            admissionLock.unlock();
        }
    }

    /**
     * Retain the request's current Decode accounting while an EngineFence
     * reconciles ambiguous delivery ownership.
     *
     * <p>The protection is exact-reservation scoped and idempotent. It may
     * attach to a shadow reservation, an engine-confirmed entry, or an existing
     * priority claim carrying the same token. A missing or replaced identity is
     * not materialized. Callers may hold a scheduler request-entry monitor;
     * this method only acquires {@link #admissionLock} and never calls back into
     * scheduler/batcher code.
     *
     * @return an opaque exact lease, or {@code null} when that reservation no
     *         longer owns Decode accounting
     */
    public EngineFenceLease beginEngineFenceProtection(
            ReservationHandle reservation) {
        if (reservation == null) {
            throw new IllegalArgumentException(
                    "Decode reservation is required for Engine fencing");
        }
        if (reservation.endpointGenerationId()
                != getStatus().getGenerationId()) {
            return null;
        }
        GenerationPin generationPin = tryPinGeneration();
        if (generationPin == null) {
            return null;
        }
        long requestId = reservation.requestId();
        try (generationPin) {
            admissionLock.lock();
            try {
                EngineFenceProtection existing = engineFenceProtections.get(requestId);
                if (existing != null) {
                    return existing.reservationToken == reservation.reservationToken()
                            ? new EngineFenceLease(this, reservation, existing)
                            : null;
                }
                RequestInflight shadow = inflightRequests.get(requestId);
                ConfirmedTask confirmed = trackedConfirmed.get(requestId);
                PreemptionClaim priorityClaim = preemptionClaims.get(requestId);
                boolean priorityOwnsAccounting = isPriorityAccountingOwner(priorityClaim);
                boolean exactShadow = shadow != null
                        && shadow.reservationToken() == reservation.reservationToken();
                boolean exactConfirmed = confirmed != null
                        && confirmed.reservationToken() == reservation.reservationToken();
                boolean exactPriorityClaim = priorityOwnsAccounting
                        && priorityClaim.reservationToken == reservation.reservationToken();
                if (!exactShadow && !exactConfirmed && !exactPriorityClaim) {
                    return null;
                }

                long hardKvTokens = exactShadow ? shadow.kvTokens()
                        : exactConfirmed ? confirmed.kvTokens()
                        : priorityClaim.hardKvTokens;
                long expectedKvTokens = exactShadow ? shadow.expectedKvTokens()
                        : exactPriorityClaim ? priorityClaim.expectedKvTokens
                        : hardKvTokens;
                boolean confirmedOwner = exactConfirmed
                        || (exactPriorityClaim
                            && priorityClaim.owner == ClaimOwner.ENGINE_CONFIRMED);
                EngineFenceProtection protection = new EngineFenceProtection(
                        reservation.reservationToken(), hardKvTokens,
                        expectedKvTokens, confirmedOwner);
                engineFenceProtections.put(requestId, protection);
                admissionVersion.incrementAndGet();
                return new EngineFenceLease(this, reservation, protection);
            } finally {
                admissionLock.unlock();
            }
        }
    }

    /**
     * Release only the generic EngineFence owner. An overlapping token-fenced
     * priority owner remains charged independently; WorkerStatus or the exact
     * priority settlement path decides its lifetime.
     *
     */
    private void closeEngineFenceExact(EngineFenceLease lease) {
        boolean capacityChanged = false;
        admissionLock.lock();
        try {
            if (lease == null || lease.endpoint != this
                    || lease.reservation.endpointGenerationId()
                            != getStatus().getGenerationId()
                    || !engineFenceProtections.remove(
                            lease.reservation.requestId(), lease.protection)) {
                return;
            }
            releaseEngineFenceSyntheticHoldLocked(lease.protection);
            if (!hasAnyRequestOwnerLocked(lease.reservation.requestId())) {
                rememberSettledLocked(
                        lease.reservation.requestId(), System.currentTimeMillis());
            }
            admissionVersion.incrementAndGet();
            capacityChanged = true;
        } finally {
            admissionLock.unlock();
        }
        if (capacityChanged) {
            notifyEngineDispatchCapacityListeners();
        }
    }

    /**
     * Atomically consume one exact authoritative Engine terminal proof.
     * Normal return guarantees that no owner for the proof's reservation token
     * remains; a reused request id is left untouched.
     */
    public void settleAuthoritativeTerminal(
            AuthoritativeTerminalProof proof) {
        if (proof == null) {
            throw new IllegalArgumentException(
                    "Authoritative Decode terminal proof is required");
        }
        if (proof.endpoint != this
                || proof.reservation.endpointGenerationId()
                        != getStatus().getGenerationId()) {
            throw new IllegalArgumentException(
                    "Authoritative Decode terminal proof belongs to another endpoint generation");
        }
        boolean capacityChanged = false;
        admissionLock.lock();
        try {
            boolean changed = settleAuthoritativeTerminalLocked(
                    proof, System.currentTimeMillis());
            if (changed) {
                admissionVersion.incrementAndGet();
            }
            capacityChanged = changed;
        } finally {
            admissionLock.unlock();
        }
        if (capacityChanged) {
            notifyEngineDispatchCapacityListeners();
        }
    }

    /**
     * Settle a definitive transport rejection without treating an ambiguous
     * Engine outcome as a local rollback. The reservation token and endpoint
     * generation fence request-id reuse; a concurrent WorkerStatus acceptance
     * always wins.
     */
    public DispatchRejectionSettlement settleDefiniteDispatchRejection(
            ReservationHandle reservation) {
        if (reservation == null) {
            throw new IllegalArgumentException(
                    "Decode reservation is required for dispatch rejection");
        }
        if (reservation.endpointGenerationId()
                != getStatus().getGenerationId()) {
            return DispatchRejectionSettlement.STALE;
        }

        boolean capacityChanged = false;
        DispatchRejectionSettlement result;
        admissionLock.lock();
        try {
            long requestId = reservation.requestId();
            long reservationToken = reservation.reservationToken();
            ConfirmedTask confirmed = trackedConfirmed.get(requestId);
            if (confirmed != null
                    && confirmed.reservationToken() == reservationToken) {
                return DispatchRejectionSettlement.ENGINE_ACCEPTED;
            }

            PreemptionClaim claim = preemptionClaims.get(requestId);
            if (claim != null && claim.reservationToken == reservationToken) {
                if (claim.owner == ClaimOwner.ENGINE_CONFIRMED) {
                    return DispatchRejectionSettlement.ENGINE_ACCEPTED;
                }
                if (!settlePriorityClaimTerminalLocked(
                        claim.attemptToken, reservation, claim)) {
                    return DispatchRejectionSettlement.CONFLICT;
                }
                capacityChanged = true;
                result = DispatchRejectionSettlement.RELEASED;
            } else {
                if (hasExactIncomingAttemptLocked(reservation)) {
                    return DispatchRejectionSettlement.CONFLICT;
                }

                EngineFenceProtection protection =
                        engineFenceProtections.get(requestId);
                boolean exactProtection = protection != null
                        && protection.reservationToken == reservationToken;
                if (exactProtection && protection.confirmedOwner) {
                    return DispatchRejectionSettlement.ENGINE_ACCEPTED;
                }

                RequestInflight shadow = inflightRequests.get(requestId);
                EngineDispatchPermit dispatchPermit = shadow == null
                        ? null : shadow.dispatchPermit();
                boolean exactShadow = isExactReservation(shadow, reservation);
                boolean exactDispatchPermit = dispatchPermit != null
                        && isExactReservation(
                                dispatchPermit.reservation, reservation);
                boolean exactEngineLifecycle =
                        hasEngineLifecycleReservationExactLocked(reservation);
                if (!exactShadow && !exactDispatchPermit
                        && !exactEngineLifecycle && !exactProtection) {
                    return DispatchRejectionSettlement.STALE;
                }

                AuthoritativeTerminalProof proof =
                        new AuthoritativeTerminalProof(
                                this,
                                reservation,
                                null,
                                AuthoritativeTerminalOwner.DISPATCH_REJECTION);
                capacityChanged = settleAuthoritativeTerminalLocked(
                        proof, System.currentTimeMillis());
                if (capacityChanged) {
                    admissionVersion.incrementAndGet();
                }
                result = DispatchRejectionSettlement.RELEASED;
            }
        } finally {
            admissionLock.unlock();
        }
        if (capacityChanged) {
            notifyEngineDispatchCapacityListeners();
        }
        return result;
    }

    /** Caller holds {@link #admissionLock}. */
    private boolean settleAuthoritativeTerminalLocked(
            AuthoritativeTerminalProof proof,
            long settledAtMs) {
        ReservationHandle reservation = proof.reservation;
        long requestId = reservation.requestId();
        long reservationToken = reservation.reservationToken();

        PreemptionClaim claim = preemptionClaims.get(requestId);
        if (isPriorityAccountingOwner(claim)
                && claim.reservationToken == reservationToken) {
            throw terminalInvariant(reservation,
                    "priority claim must settle before generic terminal ownership");
        }
        if (hasExactIncomingAttemptLocked(reservation)) {
            throw terminalInvariant(reservation,
                    "priority attempt still owns the exact incoming reservation");
        }

        boolean changed = false;
        EngineFenceProtection currentProtection =
                engineFenceProtections.get(requestId);
        boolean exactProtection = currentProtection != null
                && currentProtection.reservationToken == reservationToken;
        if (proof.owner == AuthoritativeTerminalOwner.ENGINE_FENCE
                && exactProtection
                && currentProtection != proof.protection) {
            throw terminalInvariant(reservation,
                    "another Engine fence owns the same reservation generation");
        }

        boolean syntheticSlotRemoved = exactProtection
                && currentProtection.syntheticHeld;
        if (exactProtection
                && engineFenceProtections.remove(
                        requestId, currentProtection)) {
            releaseEngineFenceSyntheticHoldLocked(currentProtection);
            changed = true;
        }

        RequestInflight permitOwner = inflightRequests.get(requestId);
        EngineDispatchPermit dispatchPermit = permitOwner == null
                ? null : permitOwner.dispatchPermit();
        if (dispatchPermit != null
                && isExactReservation(
                        dispatchPermit.reservation, reservation)) {
            changed = removeEngineDispatchPermitLocked(requestId) || changed;
        }

        ConfirmedTask confirmed = trackedConfirmed.get(requestId);
        if (confirmed != null
                && confirmed.reservationToken() == reservationToken
                && trackedConfirmed.remove(requestId, confirmed)) {
            if (!syntheticSlotRemoved) {
                confirmedEngineOwnedCount = Math.max(
                        0, confirmedEngineOwnedCount - 1);
            }
            changed = true;
        }

        RequestInflight shadow = inflightRequests.get(requestId);
        if (isExactReservation(shadow, reservation)
                && inflightRequests.remove(requestId, shadow)) {
            removeEngineLifecycleReservationLocked(shadow);
            removeQueuedPhaseLocked(requestId, shadow);
            inflightKvReservedTotal.addAndGet(-shadow.kvTokens());
            inflightExpectedKvReservedTotal.addAndGet(
                    -shadow.expectedKvTokens());
            changed = true;
        }
        changed = removeEngineLifecycleReservationExactLocked(
                reservationToken) || changed;

        if (!hasAnyRequestOwnerLocked(requestId)
                && (proof.owner == AuthoritativeTerminalOwner.ENGINE_FENCE
                    || proof.owner
                        == AuthoritativeTerminalOwner.DISPATCH_REJECTION
                    || exactProtection)) {
            changed = rememberSettledLocked(requestId, settledAtMs) || changed;
        }

        if (hasExactOwnerLocked(reservation)) {
            throw terminalInvariant(reservation,
                    "exact accounting remains after authoritative settlement");
        }
        return changed;
    }

    /** Caller holds {@link #admissionLock} after validating a finished status. */
    private AuthoritativeTerminalProof workerStatusTerminalProofLocked(
            long requestId) {
        ConfirmedTask confirmed = trackedConfirmed.get(requestId);
        long reservationToken;
        if (confirmed != null) {
            reservationToken = confirmed.reservationToken();
        } else {
            RequestInflight shadow = inflightRequests.get(requestId);
            reservationToken = shadow == null
                    ? 0L : shadow.reservationToken();
        }
        if (reservationToken <= 0L) {
            return null;
        }
        return new AuthoritativeTerminalProof(
                this,
                new ReservationHandle(
                        getStatus().getGenerationId(),
                        requestId, reservationToken),
                null,
                AuthoritativeTerminalOwner.WORKER_STATUS);
    }

    /**
     * Remove only one Engine task that never had a local reservation identity.
     * A same-id shadow/fence necessarily belongs to another generation and is
     * deliberately preserved.
     */
    private boolean settleUntrackedWorkerTerminalLocked(long requestId) {
        ConfirmedTask confirmed = trackedConfirmed.get(requestId);
        if (confirmed == null || confirmed.reservationToken() > 0L
                || !trackedConfirmed.remove(requestId, confirmed)) {
            return false;
        }
        confirmedEngineOwnedCount = Math.max(0, confirmedEngineOwnedCount - 1);
        return true;
    }

    /** Caller holds {@link #admissionLock}. */
    private boolean removeEngineLifecycleReservationExactLocked(
            long reservationToken) {
        return engineLifecycleReservationTokens.remove(reservationToken);
    }

    /** Caller holds {@link #admissionLock}. */
    private boolean removeEngineLifecycleReservationLocked(
            RequestInflight reservation) {
        return reservation != null
                && engineLifecycleReservationTokens.remove(
                        reservation.reservationToken());
    }

    /** Caller holds {@link #admissionLock}. */
    private boolean hasAnyRequestOwnerLocked(long requestId) {
        return inflightRequests.containsKey(requestId)
                || trackedConfirmed.containsKey(requestId)
                || engineFenceProtections.containsKey(requestId)
                || preemptionClaims.containsKey(requestId);
    }

    /**
     * WorkerStatus identifies work only by request id. Reuse is therefore
     * forbidden while this endpoint generation still owns that id or retains
     * an ambiguity tombstone for it.
     */
    private boolean requestIdAvailableForReservationLocked(long requestId) {
        if (hasAnyRequestOwnerLocked(requestId)
                || settledTombstones.containsKey(requestId)) {
            return false;
        }
        for (EndpointPreemptionAttempt attempt : preemptionAttempts.values()) {
            if (attempt.incomingRequestId == requestId) {
                return false;
            }
        }
        return true;
    }

    /** Caller holds {@link #admissionLock}. */
    private boolean hasExactOwnerLocked(ReservationHandle reservation) {
        long requestId = reservation.requestId();
        long token = reservation.reservationToken();
        RequestInflight shadow = inflightRequests.get(requestId);
        ConfirmedTask confirmed = trackedConfirmed.get(requestId);
        EngineDispatchPermit dispatchPermit = shadow == null
                ? null : shadow.dispatchPermit();
        EngineFenceProtection protection =
                engineFenceProtections.get(requestId);
        PreemptionClaim claim = preemptionClaims.get(requestId);
        if (isExactReservation(shadow, reservation)
                || (confirmed != null && confirmed.reservationToken() == token)
                || (dispatchPermit != null
                    && isExactReservation(
                            dispatchPermit.reservation, reservation))
                || (protection != null && protection.reservationToken == token)
                || (claim != null && claim.reservationToken == token)
                || hasExactIncomingAttemptLocked(reservation)) {
            return true;
        }
        return engineLifecycleReservationTokens.contains(token);
    }

    private static IllegalStateException terminalInvariant(
            ReservationHandle reservation,
            String detail) {
        return new IllegalStateException(
                "Illegal Decode authoritative settlement: requestId="
                        + reservation.requestId()
                        + ", reservationToken="
                        + reservation.reservationToken()
                        + ", detail=" + detail);
    }

    /** Atomic result of one exact local-Decode eviction admission. */
    public enum LocalEvictionResult {
        COMMITTED,
        CONFLICT,
        INFEASIBLE,
        ENDPOINT_RETIRED
    }

    /**
     * Atomically replace a fully validated set of exact Master-queued
     * reservations with one incoming reservation. Any stale victim or
     * incoming-id conflict leaves every owner untouched.
     */
    public LocalEvictionResult tryEvictLocalReservationsAndReserveIncoming(
            List<ReservationHandle> victims,
            long incomingRequestId, long kvTokens, long expectedKvTokens,
            int priority,
            AdmissionCapacity capacity) {
        GenerationPin generationPin = tryPinGeneration();
        if (generationPin == null) {
            return LocalEvictionResult.ENDPOINT_RETIRED;
        }
        try (generationPin) {
            return evictLocalReservationsAndReserveIncomingPinned(
                    victims,
                    incomingRequestId,
                    kvTokens,
                    expectedKvTokens,
                    priority,
                    capacity);
        }
    }

    private LocalEvictionResult
            evictLocalReservationsAndReserveIncomingPinned(
            List<ReservationHandle> victims,
            long incomingRequestId, long kvTokens, long expectedKvTokens,
            int priority,
            AdmissionCapacity capacity) {
        boolean committed = false;
        admissionLock.lock();
        try {
            if (victims == null || victims.isEmpty()
                    || !requestIdAvailableForReservationLocked(
                            incomingRequestId)) {
                return LocalEvictionResult.CONFLICT;
            }
            Set<Long> uniqueVictims = new HashSet<>(victims.size());
            long freedHardKv = 0L;
            long freedExpectedUsage = 0L;
            for (ReservationHandle victim : victims) {
                if (victim == null
                        || victim.endpointGenerationId()
                                != getStatus().getGenerationId()
                        || victim.requestId() == incomingRequestId
                        || !uniqueVictims.add(victim.requestId())) {
                    return LocalEvictionResult.CONFLICT;
                }
                RequestInflight held = inflightRequests.get(victim.requestId());
                EngineDispatchPermit permit = held == null
                        ? null : held.dispatchPermit();
                if (!isExactReservation(held, victim)
                        || !held.queued()
                        || hasEngineLifecycleReservationExactLocked(victim)
                        || preemptionClaims.containsKey(victim.requestId())
                        || engineFenceProtections.containsKey(victim.requestId())
                        || hasExactIncomingAttemptLocked(victim)
                        || permit != null) {
                    return LocalEvictionResult.CONFLICT;
                }
                freedHardKv = saturatedAddNonNegative(
                        freedHardKv, held.kvTokens());
                freedExpectedUsage = saturatedAddNonNegative(
                        freedExpectedUsage, held.expectedKvTokens());
            }
            if (projectedEvictionCapacityFitsLocked(
                    capacity, kvTokens, expectedKvTokens,
                    0L, 0L, 0L)) {
                return LocalEvictionResult.INFEASIBLE;
            }
            if (!projectedEvictionCapacityFitsLocked(
                    capacity, kvTokens, expectedKvTokens,
                    0L, freedHardKv, freedExpectedUsage)) {
                return LocalEvictionResult.INFEASIBLE;
            }

            for (ReservationHandle victim : victims) {
                RequestInflight exact = inflightRequests.get(victim.requestId());
                if (!inflightRequests.remove(victim.requestId(), exact)) {
                    throw rollbackInvariant(
                            victim,
                            "validated victim changed while admissionLock was held");
                }
                removeQueuedPhaseLocked(victim.requestId(), exact);
                inflightKvReservedTotal.addAndGet(-exact.kvTokens());
                inflightExpectedKvReservedTotal.addAndGet(
                        -exact.expectedKvTokens());
            }
            reserveLocked(
                    incomingRequestId,
                    kvTokens,
                    expectedKvTokens,
                    priority);
            committed = true;
            return LocalEvictionResult.COMMITTED;
        } finally {
            admissionLock.unlock();
            if (committed) {
                notifyEngineDispatchCapacityListeners();
            }
        }
    }

    /** Consistent reserved/confirmed ownership view captured under admissionLock. */
    public LayeredAdmissionView layeredAdmissionView() {
        admissionLock.lock();
        try {
            List<ConfirmedTaskView> confirmed = new java.util.ArrayList<>(trackedConfirmed.size());
            trackedConfirmed.forEach((requestId, task) ->
                    confirmed.add(new ConfirmedTaskView(requestId, task.priority(),
                            task.kvTokens(), task.phase(), task.priorityKnown(),
                            task.reservationToken(),
                            preemptionClaims.containsKey(requestId)
                                    || engineFenceProtections.containsKey(requestId))));
            Set<Long> claimed = Set.copyOf(preemptionClaims.keySet());
            if (!engineFenceProtections.isEmpty()) {
                Set<Long> combined = new HashSet<>(claimed);
                combined.addAll(engineFenceProtections.keySet());
                claimed = Set.copyOf(combined);
            }
            return new LayeredAdmissionView(routingViewLocked(),
                    Map.copyOf(inflightRequests), List.copyOf(confirmed),
                    queuedRequestIdsLocked(),
                    claimed, activeEngineDispatchPermitCount);
        } finally {
            admissionLock.unlock();
        }
    }

    private Set<Long> queuedRequestIdsLocked() {
        Set<Long> queued = new HashSet<>();
        inflightRequests.forEach((requestId, reservation) -> {
            if (reservation.queued()) {
                queued.add(requestId);
            }
        });
        return Set.copyOf(queued);
    }

    /** Atomic reserved/confirmed/queued ownership tuple. */
    public record LayeredAdmissionView(DecodeRoutingView routing,
                                       Map<Long, RequestInflight> reserved,
                                       List<ConfirmedTaskView> confirmed,
                                       java.util.Set<Long> queued,
                                       Set<Long> claimed,
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

        private int phaseCount(DecodeTaskPhase phase) {
            int count = 0;
            for (ConfirmedTaskView task : confirmed) {
                if (task.phase() == phase) {
                    count++;
                }
            }
            return count;
        }
    }

    /**
     * One on-demand routing projection captured under the canonical admission
     * lock. It is not a second owner: every value is derived from the live
     * registries and the one committed WorkerStatus holder at capture time.
     */
    public record DecodeRoutingView(
            WorkerStatus.CommittedWorkerStatus workerStatus,
            long admissionVersion,
            int totalLoad,
            int engineLoad,
            int engineCapacityUsed,
            long realKvUsed,
            long realKvAvailable,
            long engineFacingKvUsed,
            long engineFacingKvAvailable,
            long totalKv,
            long inflightHardKv,
            long inflightExpectedKv) {
    }

    /**
     * Project one request through the same Engine-facing capacity policy used
     * by the exact dispatch permit. This immutable-view result is advisory:
     * the permit acquisition remains the serialization point, but routing can
     * no longer prefer an endpoint which the captured state already proves
     * cannot accept this request.
     */
    public static boolean canAcquireEngineDispatchPermit(
            DecodeRoutingView view,
            EngineDispatchDemand demand,
            AdmissionCapacity capacity) {
        if (view == null || demand == null || capacity == null) {
            throw new IllegalArgumentException(
                    "Decode routing view, demand, and capacity are required");
        }
        return engineDispatchCapacityFits(
                view.engineCapacityUsed(),
                view.engineFacingKvAvailable(),
                view.engineFacingKvUsed(),
                view.totalKv(),
                demand,
                capacity.maxEngineRequests(),
                capacity.maxKvUsagePercent());
    }

    public DecodeRoutingView routingView() {
        admissionLock.lock();
        try {
            return routingViewLocked();
        } finally {
            admissionLock.unlock();
        }
    }

    /**
     * Reuse an immutable observation for bulk selection when both routing
     * generations are unchanged. This is deliberately separate from
     * {@link #routingView()}: winner authorization must always take the fresh
     * locked path above.
     */
    DecodeRoutingView routingViewSnapshot() {
        WorkerStatus.CommittedWorkerStatus committed =
                getStatus().committedWorkerStatus();
        long version = admissionVersion.get();
        RoutingViewCache cached = routingViewCache;
        if (cached != null
                && cached.workerStatus() == committed
                && cached.admissionVersion() == version) {
            return cached.routing();
        }
        admissionLock.lock();
        try {
            committed =
                    getStatus().committedWorkerStatus();
            version = admissionVersion.get();
            cached = routingViewCache;
            if (cached != null
                    && cached.workerStatus() == committed
                    && cached.admissionVersion() == version) {
                return cached.routing();
            }
            DecodeRoutingView routing = routingViewLocked(
                    committed, version);
            routingViewCache = new RoutingViewCache(
                    committed, version, routing);
            return routing;
        } finally {
            admissionLock.unlock();
        }
    }

    /** Caller holds {@link #admissionLock}. */
    private DecodeRoutingView routingViewLocked() {
        WorkerStatus.CommittedWorkerStatus committed =
                getStatus().committedWorkerStatus();
        return routingViewLocked(committed, admissionVersion.get());
    }

    /** Caller holds {@link #admissionLock}; both key values were read there. */
    private DecodeRoutingView routingViewLocked(
            WorkerStatus.CommittedWorkerStatus committed,
            long version) {
        WorkerStatus.EngineObservation fields = committed.fields();
        int inflight = inflightRequests.size();
        int queued = Math.max(0, Math.min(queuedPhaseCount.get(), inflight));
        int totalLoad = confirmedEngineOwnedCount + inflight;
        int engineLoad = confirmedEngineOwnedCount + Math.max(0, inflight - queued);
        int engineCapacityUsed = engineLoad + activeEngineDispatchPermitCount;
        long reportedUsed = fields.totalKvCacheTokens() > 0
                ? Math.max(0L, fields.totalKvCacheTokens()
                        - fields.availableKvCacheTokens())
                : 0L;
        long hardInflight = inflightKvReservedTotal.get();
        long expectedInflight = inflightExpectedKvReservedTotal.get();
        long used = saturatedAddNonNegative(
                saturatedAddNonNegative(reportedUsed, expectedInflight),
                saturatedAddNonNegative(
                        priorityPreemptionHeldExpectedKv.get(),
                        engineFenceHeldExpectedKv.get()));
        long available = Math.max(0L, fields.availableKvCacheTokens()
                - hardInflight
                - priorityPreemptionHeldKv.get()
                - engineFenceHeldKv.get());
        long engineFacingUsed = engineFacingKvUsed(fields);
        long engineFacingAvailable = engineFacingKvAvailable(fields);
        return new DecodeRoutingView(
                committed,
                version,
                totalLoad,
                engineLoad,
                engineCapacityUsed,
                used,
                available,
                engineFacingUsed,
                engineFacingAvailable,
                fields.totalKvCacheTokens(),
                hardInflight,
                expectedInflight);
    }

    /** Immutable point-in-time view of one layered-registry entry. */
    public record ConfirmedTaskView(long requestId,
                                    int priority,
                                    long kvTokens,
                                    DecodeTaskPhase phase,
                                    boolean priorityKnown,
                                    long reservationToken,
                                    boolean claimedForPreemption) {
    }

    // ==================== Priority-preemption transaction ====================

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

    /**
     * Atomically claim Engine-visible victims and reserve the incoming demand
     * as provisional capacity.  Victim accounting is intentionally untouched:
     * Cancel ACCEPTED is only an intent acknowledgement.
     */
    public PreemptionBeginResult beginPriorityPreemption(
            long attemptToken,
            List<ReservationHandle> victims,
            long incomingRequestId,
            long incomingKvTokens,
            long incomingExpectedKvTokens,
            int incomingPriority,
            AdmissionCapacity capacity) {
        if (attemptToken <= 0 || victims == null || victims.isEmpty()) {
            throw new IllegalArgumentException("attempt token and victims are required");
        }
        GenerationPin generationPin = tryPinGeneration();
        if (generationPin == null) {
            return PreemptionBeginResult.ENDPOINT_RETIRED;
        }
        try (generationPin) {
            return beginPriorityPreemptionPinned(
                    attemptToken,
                    victims,
                    incomingRequestId,
                    incomingKvTokens,
                    incomingExpectedKvTokens,
                    incomingPriority,
                    capacity);
        }
    }

    private PreemptionBeginResult beginPriorityPreemptionPinned(
            long attemptToken,
            List<ReservationHandle> victims,
            long incomingRequestId,
            long incomingKvTokens,
            long incomingExpectedKvTokens,
            int incomingPriority,
            AdmissionCapacity capacity) {
        admissionLock.lock();
        try {
            if (preemptionAttempts.containsKey(attemptToken)) {
                return PreemptionBeginResult.ATTEMPT_ALREADY_EXISTS;
            }
            if (!requestIdAvailableForReservationLocked(incomingRequestId)) {
                return PreemptionBeginResult.INCOMING_ALREADY_RESERVED;
            }

            Map<Long, ClaimOwner> owners = new HashMap<>();
            Map<Long, ReservationHandle> exactVictims = new HashMap<>();
            long freedSlots = 0L;
            long freedHardKv = 0L;
            long freedExpectedUsage = 0L;
            for (ReservationHandle victim : victims) {
                if (victim == null
                        || victim.endpointGenerationId()
                                != getStatus().getGenerationId()
                        || victim.requestId() == incomingRequestId
                        || owners.containsKey(victim.requestId())) {
                    return PreemptionBeginResult.VICTIM_GONE;
                }
                long victimId = victim.requestId();
                if (preemptionClaims.containsKey(victimId)
                        || engineFenceProtections.containsKey(victimId)
                        || hasExactIncomingAttemptLocked(victim)) {
                    return PreemptionBeginResult.VICTIM_ALREADY_CLAIMED;
                }
                RequestInflight shadow = inflightRequests.get(victimId);
                ConfirmedTask confirmed = trackedConfirmed.get(victimId);
                boolean exactShadow = isExactReservation(shadow, victim);
                boolean exactConfirmed = confirmed != null
                        && confirmed.reservationToken()
                                == victim.reservationToken();
                if (exactShadow && exactConfirmed) {
                    return PreemptionBeginResult.VICTIM_GONE;
                }
                EngineDispatchPermit dispatchPermit = shadow == null
                        ? null : shadow.dispatchPermit();
                if (dispatchPermit != null) {
                    return PreemptionBeginResult.VICTIM_GONE;
                }
                if (exactShadow && !shadow.queued()) {
                    if (shadow.priority() <= 0 || shadow.priority() >= incomingPriority) {
                        return PreemptionBeginResult.INVALID_PRIORITY;
                    }
                    owners.put(victimId, ClaimOwner.SHADOW_IN_FLIGHT);
                    exactVictims.put(victimId, victim);
                    freedSlots++;
                    freedHardKv = saturatedAddNonNegative(
                            freedHardKv, shadow.kvTokens());
                    freedExpectedUsage = saturatedAddNonNegative(
                            freedExpectedUsage, shadow.expectedKvTokens());
                } else if (exactConfirmed && confirmed.phase().isEngineConfirmed()) {
                    if (confirmed.priority() <= 0 || confirmed.priority() >= incomingPriority) {
                        return PreemptionBeginResult.INVALID_PRIORITY;
                    }
                    owners.put(victimId, ClaimOwner.ENGINE_CONFIRMED);
                    exactVictims.put(victimId, victim);
                    freedSlots++;
                    freedHardKv = saturatedAddNonNegative(
                            freedHardKv, confirmed.kvTokens());
                    freedExpectedUsage = saturatedAddNonNegative(
                            freedExpectedUsage, confirmed.kvTokens());
                } else {
                    return PreemptionBeginResult.VICTIM_GONE;
                }
            }
            if (projectedEvictionCapacityFitsLocked(
                    capacity, incomingKvTokens, incomingExpectedKvTokens,
                    0L, 0L, 0L)) {
                return PreemptionBeginResult.INFEASIBLE;
            }
            if (!projectedEvictionCapacityFitsLocked(
                    capacity, incomingKvTokens, incomingExpectedKvTokens,
                    freedSlots, freedHardKv, freedExpectedUsage)) {
                return PreemptionBeginResult.INFEASIBLE;
            }

            // Provisional incoming ownership closes the free-pool race while
            // Cancel runs.  It is not visible to the prefill queue yet.
            ReservationHandle incomingReservation = reserveLocked(
                    incomingRequestId, incomingKvTokens,
                    incomingExpectedKvTokens, incomingPriority);
            for (ReservationHandle victim : victims) {
                long victimId = victim.requestId();
                RequestInflight shadow = inflightRequests.get(victimId);
                ConfirmedTask confirmed = trackedConfirmed.get(victimId);
                long hardKv = shadow != null ? shadow.kvTokens()
                        : confirmed != null ? confirmed.kvTokens() : 0;
                long expectedKv = shadow != null ? shadow.expectedKvTokens() : hardKv;
                preemptionClaims.put(victimId,
                        new PreemptionClaim(
                                attemptToken, owners.get(victimId),
                                victim.reservationToken(),
                                hardKv, expectedKv));
            }
            preemptionAttempts.put(attemptToken,
                    new EndpointPreemptionAttempt(
                            incomingRequestId,
                            incomingReservation.reservationToken(),
                            exactVictims));
            admissionVersion.incrementAndGet();
            return PreemptionBeginResult.SUCCESS;
        } finally {
            admissionLock.unlock();
        }
    }

    /** CLAIMED -> CANCEL_IN_FLIGHT; must complete before any outbound RPC. */
    public boolean markPriorityCancelInFlight(long attemptToken) {
        admissionLock.lock();
        try {
            EndpointPreemptionAttempt attempt = preemptionAttempts.get(attemptToken);
            if (attempt == null) {
                return false;
            }
            for (ReservationHandle victim
                    : attempt.remainingVictims.values()) {
                long victimId = victim.requestId();
                PreemptionClaim claim = preemptionClaims.get(victimId);
                if (claim == null || claim.attemptToken != attemptToken
                        || claim.reservationToken
                                != victim.reservationToken()
                        || !claim.phase.canStartCancel()) {
                    return false;
                }
            }
            for (ReservationHandle victim
                    : attempt.remainingVictims.values()) {
                preemptionClaims.get(victim.requestId()).phase =
                        PreemptionCancelPhase.CANCEL_IN_FLIGHT;
            }
            admissionVersion.incrementAndGet();
            return true;
        } finally {
            admissionLock.unlock();
        }
    }

    /** Cancel ACCEPTED: retain every byte/slot and only advance control state. */
    public boolean markPriorityCancelAccepted(long attemptToken, long requestId) {
        return transitionClaim(attemptToken, requestId,
                PreemptionCancelPhase.CANCEL_IN_FLIGHT,
                PreemptionCancelPhase.CANCEL_REQUESTED);
    }

    public boolean markPriorityCancelNotFound(long attemptToken, long requestId) {
        return transitionClaim(attemptToken, requestId,
                PreemptionCancelPhase.CANCEL_IN_FLIGHT,
                PreemptionCancelPhase.NOT_FOUND_STALE);
    }

    public boolean markPriorityCancelUnknown(long attemptToken, long requestId) {
        admissionLock.lock();
        try {
            PreemptionClaim claim = preemptionClaims.get(requestId);
            if (claim == null || claim.attemptToken != attemptToken
                    || !claim.phase.canRecordUnknown()) {
                return false;
            }
            claim.phase = PreemptionCancelPhase.CANCEL_UNKNOWN;
            admissionVersion.incrementAndGet();
            return true;
        } finally {
            admissionLock.unlock();
        }
    }

    /**
     * Settle an exact typed Prefill CANCELED proof. Ordinary Decode terminal
     * status uses the same resource transaction but keeps its own outcome
     * classification; duplicate or stale observations are total no-ops.
     */
    public boolean settlePriorityCanceled(
            long attemptToken,
            ReservationHandle reservation) {
        if (reservation == null
                || reservation.endpointGenerationId()
                        != getStatus().getGenerationId()) {
            return false;
        }
        long requestId = reservation.requestId();
        boolean settled = false;
        boolean capacityChanged = false;
        admissionLock.lock();
        try {
            PreemptionClaim claim = preemptionClaims.get(requestId);
            if (claim == null || claim.attemptToken != attemptToken
                    || claim.reservationToken
                            != reservation.reservationToken()
                    || !claim.phase.acceptsPriorityTerminal()) {
                return false;
            }
            settled = settlePriorityClaimTerminalLocked(
                    attemptToken, reservation, claim);
            capacityChanged = settled;
        } finally {
            admissionLock.unlock();
        }
        if (capacityChanged) {
            notifyEngineDispatchCapacityListeners();
        }
        return settled;
    }

    /**
     * Settle an engine {@code TOMBSTONED} acknowledgement.
     *
     * <p>TOMBSTONED is stronger than NOT_FOUND: the addressed request was
     * absent and the engine atomically installed a late-enqueue fence. It is
     * therefore an authoritative terminal proof and may release the same
     * accounting as typed CANCELED without waiting for WorkerStatus.</p>
     */
    public boolean settlePriorityTombstoned(
            long attemptToken,
            ReservationHandle reservation) {
        if (reservation == null
                || reservation.endpointGenerationId()
                        != getStatus().getGenerationId()) {
            return false;
        }
        long requestId = reservation.requestId();
        boolean settled = false;
        boolean capacityChanged = false;
        admissionLock.lock();
        try {
            PreemptionClaim claim = preemptionClaims.get(requestId);
            if (claim == null || claim.attemptToken != attemptToken
                    || claim.reservationToken
                            != reservation.reservationToken()
                    || claim.engineFenceTransferred
                    || !claim.phase.acceptsTombstone()) {
                return false;
            }
            settled = settlePriorityClaimTerminalLocked(
                    attemptToken, reservation, claim);
            capacityChanged = settled;
        } finally {
            admissionLock.unlock();
        }
        if (capacityChanged) {
            notifyEngineDispatchCapacityListeners();
        }
        return settled;
    }

    /**
     * Transfer a NOT_FOUND priority claim to the scheduler's post-delivery
     * Engine fence without releasing any victim accounting.
     *
     * <p>The original attempt token becomes the exact endpoint-side fence
     * generation. In particular, {@code kvHeldAfterWorkerRelease} remains
     * charged when Decode has already stopped reporting an ENGINE_CONFIRMED
     * victim; dropping that synthetic hold here would oversell KV before the
     * new Cancel observes TOMBSTONED or another authoritative terminal.</p>
     */
    public boolean transferPriorityNotFoundClaimToEngineFence(
            long attemptToken,
            long requestId) {
        admissionLock.lock();
        try {
            PreemptionClaim claim = preemptionClaims.get(requestId);
            if (claim == null || claim.attemptToken != attemptToken
                    || claim.engineFenceTransferred
                    || claim.phase != PreemptionCancelPhase.NOT_FOUND_STALE) {
                return false;
            }
            claim.engineFenceTransferred = true;
            admissionVersion.incrementAndGet();
            return true;
        } finally {
            admissionLock.unlock();
        }
    }

    /** Authoritative terminal settlement for an exact transferred fence generation. */
    public boolean settleEngineFenceClaim(
            long attemptToken,
            ReservationHandle reservation) {
        if (reservation == null
                || reservation.endpointGenerationId()
                        != getStatus().getGenerationId()) {
            return false;
        }
        long requestId = reservation.requestId();
        boolean settled = false;
        boolean capacityChanged = false;
        admissionLock.lock();
        try {
            PreemptionClaim claim = preemptionClaims.get(requestId);
            if (claim == null || claim.attemptToken != attemptToken
                        || claim.reservationToken
                                != reservation.reservationToken()
                        || !claim.engineFenceTransferred) {
                return false;
            }
            settled = settlePriorityClaimTerminalLocked(
                    attemptToken, reservation, claim);
            capacityChanged = settled;
        } finally {
            admissionLock.unlock();
        }
        if (capacityChanged) {
            notifyEngineDispatchCapacityListeners();
        }
        return settled;
    }

    /** Called with {@link #admissionLock} held. */
    private boolean settlePriorityClaimTerminalLocked(
            long attemptToken,
            ReservationHandle reservation,
            PreemptionClaim claim) {
        long requestId = reservation.requestId();
        long reservationToken = reservation.reservationToken();
        if (preemptionClaims.get(requestId) != claim
                || claim.attemptToken != attemptToken
                || claim.reservationToken != reservationToken) {
            return false;
        }

        RequestInflight shadow = inflightRequests.get(requestId);
        ConfirmedTask confirmed = trackedConfirmed.get(requestId);
        EngineDispatchPermit dispatchPermit = shadow == null
                ? null : shadow.dispatchPermit();
        EngineFenceProtection protection =
                engineFenceProtections.get(requestId);
        if ((shadow != null && !isExactReservation(shadow, reservation))
                || (confirmed != null
                    && confirmed.reservationToken() != reservationToken)
                || (dispatchPermit != null
                    && !isExactReservation(
                            dispatchPermit.reservation, reservation))
                || (protection != null
                    && protection.reservationToken != reservationToken)
                || hasExactIncomingAttemptLocked(reservation)) {
            return false;
        }
        EndpointPreemptionAttempt attempt =
                preemptionAttempts.get(attemptToken);
        if (attempt != null
                && !reservation.equals(
                        attempt.remainingVictims.get(requestId))) {
            return false;
        }

        boolean syntheticSlotRemoved = protection != null
                && protection.syntheticHeld;
        if (protection != null
                && engineFenceProtections.remove(requestId, protection)) {
            releaseEngineFenceSyntheticHoldLocked(protection);
        }
        if (dispatchPermit != null) {
            removeEngineDispatchPermitLocked(requestId);
        }
        if (confirmed != null) {
            trackedConfirmed.remove(requestId, confirmed);
        }
        if (claim.owner == ClaimOwner.ENGINE_CONFIRMED
                && !syntheticSlotRemoved) {
            confirmedEngineOwnedCount = Math.max(0, confirmedEngineOwnedCount - 1);
        }
        if (shadow != null
                && inflightRequests.remove(requestId, shadow)) {
            removeEngineLifecycleReservationLocked(shadow);
            removeQueuedPhaseLocked(requestId, shadow);
            inflightKvReservedTotal.addAndGet(-shadow.kvTokens());
            inflightExpectedKvReservedTotal.addAndGet(
                    -shadow.expectedKvTokens());
        }
        removeEngineLifecycleReservationExactLocked(reservationToken);
        releaseHeldKv(claim);
        preemptionClaims.remove(requestId, claim);
        if (attempt != null) {
            attempt.remainingVictims.remove(requestId, reservation);
        }
        rememberSettledLocked(requestId, System.currentTimeMillis());
        admissionVersion.incrementAndGet();
        return true;
    }

    /** Mark the full incoming reservation committed after every victim settles. */
    public boolean commitPriorityPreemption(long attemptToken) {
        admissionLock.lock();
        try {
            EndpointPreemptionAttempt attempt = preemptionAttempts.get(attemptToken);
            RequestInflight incoming = attempt == null
                    ? null : inflightRequests.get(attempt.incomingRequestId);
            if (attempt == null || incoming == null
                    || incoming.reservationToken()
                            != attempt.incomingReservationToken) {
                return false;
            }
            if (!attempt.remainingVictims.isEmpty()) {
                return false;
            }
            preemptionAttempts.remove(attemptToken);
            admissionVersion.incrementAndGet();
            return true;
        } finally {
            admissionLock.unlock();
        }
    }

    /**
     * Abort the incoming attempt.  Provisional incoming ownership is released;
     * successfully canceled victims become ordinary free capacity, while
     * NOT_FOUND/unknown claims retain their accounting and reconciliation
     * fence.
     */
    public void abortPriorityPreemption(long attemptToken) {
        boolean capacityChanged = false;
        admissionLock.lock();
        try {
            EndpointPreemptionAttempt attempt = preemptionAttempts.remove(attemptToken);
            if (attempt == null) {
                return;
            }
            releaseLocked(attempt.incomingRequestId);
            for (ReservationHandle victim
                    : attempt.remainingVictims.values()) {
                long victimId = victim.requestId();
                PreemptionClaim claim = preemptionClaims.get(victimId);
                if (claim == null || claim.attemptToken != attemptToken
                        || claim.reservationToken
                                != victim.reservationToken()) {
                    continue;
                }
                if (claim.phase.isLocallyReleasable()) {
                    releaseHeldKv(claim);
                    preemptionClaims.remove(victimId);
                }
            }
            admissionVersion.incrementAndGet();
            capacityChanged = true;
        } finally {
            admissionLock.unlock();
        }
        if (capacityChanged) {
            notifyEngineDispatchCapacityListeners();
        }
    }

    /** Fresh active status is the only path that reopens a NOT_FOUND_STALE victim. */
    public boolean reconcilePriorityVictimActive(
            long attemptToken,
            ReservationHandle reservation) {
        if (reservation == null
                || reservation.endpointGenerationId()
                        != getStatus().getGenerationId()) {
            return false;
        }
        long requestId = reservation.requestId();
        boolean reconciled = false;
        admissionLock.lock();
        try {
            PreemptionClaim claim = preemptionClaims.get(requestId);
            if (claim == null
                    || claim.attemptToken != attemptToken
                    || claim.reservationToken
                            != reservation.reservationToken()
                    || claim.engineFenceTransferred
                    || claim.phase != PreemptionCancelPhase.NOT_FOUND_STALE) {
                return false;
            }
            releaseHeldKv(claim);
            preemptionClaims.remove(requestId);
            admissionVersion.incrementAndGet();
            reconciled = true;
        } finally {
            admissionLock.unlock();
        }
        if (reconciled) {
            notifyEngineDispatchCapacityListeners();
        }
        return reconciled;
    }

    /**
     * Reconcile a one-shot ordinary Decode terminal after Cancel NOT_FOUND or
     * a transport-unknown ACK. Neither outcome is typed priority completion,
     * so the ordinary terminal resumes the pre-existing completion path.
     */
    public boolean reconcilePriorityVictimFinished(
            long attemptToken,
            ReservationHandle reservation) {
        if (reservation == null
                || reservation.endpointGenerationId()
                        != getStatus().getGenerationId()) {
            return false;
        }
        long requestId = reservation.requestId();
        boolean reconciled = false;
        boolean capacityChanged = false;
        admissionLock.lock();
        try {
            PreemptionClaim claim = preemptionClaims.get(requestId);
            if (claim == null
                    || claim.attemptToken != attemptToken
                    || claim.reservationToken
                            != reservation.reservationToken()
                    || claim.engineFenceTransferred
                    || !claim.phase.requiresOrdinaryReconciliation()) {
                return false;
            }
            // An ordinary finished sample is authoritative. Unlike an
            // ambiguous priority ACK, it must clear the generic fence rather
            // than transfer accounting into it.
            releaseHeldKv(claim);
            preemptionClaims.remove(requestId);
            if (claim.reservationToken > 0L) {
                settleAuthoritativeTerminalLocked(
                        new AuthoritativeTerminalProof(
                                this,
                                reservation,
                                null,
                                AuthoritativeTerminalOwner.WORKER_STATUS),
                        System.currentTimeMillis());
            } else {
                settleUntrackedWorkerTerminalLocked(requestId);
            }
            admissionVersion.incrementAndGet();
            reconciled = true;
            capacityChanged = true;
        } finally {
            admissionLock.unlock();
        }
        if (capacityChanged) {
            notifyEngineDispatchCapacityListeners();
        }
        return reconciled;
    }

    private boolean transitionClaim(long attemptToken, long requestId,
                                    PreemptionCancelPhase expected,
                                    PreemptionCancelPhase next) {
        admissionLock.lock();
        try {
            PreemptionClaim claim = preemptionClaims.get(requestId);
            if (claim == null
                    || claim.attemptToken != attemptToken
                    || claim.phase != expected) {
                return false;
            }
            claim.phase = next;
            admissionVersion.incrementAndGet();
            return true;
        } finally {
            admissionLock.unlock();
        }
    }

    /** Accepted-not-running layer size (Phase 5 gauge). */
    public int getAcceptedLayerCount() {
        int count = 0;
        for (ConfirmedTask task : trackedConfirmed.values()) {
            if (task.phase() == DecodeTaskPhase.ACCEPTED_NOT_RUNNING) {
                count++;
            }
        }
        return count;
    }

    /** True running layer size (Phase 5 gauge). */
    public int getRunningLayerCount() {
        int count = 0;
        for (ConfirmedTask task : trackedConfirmed.values()) {
            if (task.phase() == DecodeTaskPhase.RUNNING) {
                count++;
            }
        }
        return count;
    }

    /** Current diagnostic generation of the layered admission projection. */
    public long admissionVersion() {
        return admissionVersion.get();
    }

    @Override
    public EndpointStatusReduction applyPreparedStatus(
            WorkerStatus ws,
            WorkerStatus.PreparedStatus prepared) {
        requireStatusGeneration(ws);
        WorkerStatus.StatusObservation observation = prepared.observation();
        List<WorkerStatusFact> facts;
        boolean placementCapacityImproved;
        admissionLock.lock();
        try {
            DecodeRoutingView before = routingViewLocked();
            facts = doCalibrate(
                    observation.engine(), observation.finishedTasks());
            if (!observation.alive()) {
                beginRetirement();
            }
            ws.publishPreparedStatus(prepared);
            placementCapacityImproved = placementCapacityImproved(
                    before, routingViewLocked());
        } catch (RuntimeException | Error failure) {
            beginRetirement();
            throw failure;
        } finally {
            admissionLock.unlock();
        }
        notifyEngineDispatchCapacityListeners();
        if (placementCapacityImproved) {
            signalPlacementCapacityChanged();
        }
        return new StatusReduction(this, facts);
    }

    private static boolean placementCapacityImproved(
            DecodeRoutingView before,
            DecodeRoutingView after) {
        return after.engineLoad() < before.engineLoad()
                || after.realKvAvailable() > before.realKvAvailable()
                || after.realKvUsed() < before.realKvUsed();
    }

    private void signalPlacementCapacityChanged() {
        WorkerStatus.TopologySnapshot topology =
                getStatus().topologySnapshot();
        placementAvailability.capacityChanged(
                RoleType.DECODE, topology.group());
    }

    @Override
    public EndpointStatusReduction initializeFromPreparedStatus(
            WorkerStatus ws,
            WorkerStatus.StatusObservation observation) {
        requireStatusGeneration(ws);
        List<WorkerStatusFact> facts;
        admissionLock.lock();
        try {
            facts = doCalibrate(
                    observation.engine(), observation.finishedTasks());
        } finally {
            admissionLock.unlock();
        }
        if (!facts.isEmpty()) {
            throw new IllegalStateException(
                    "Private Decode candidate produced locally-owned status facts");
        }
        return new StatusReduction(this, List.of());
    }

    @Override
    public EndpointStatusReduction observeStatusHeartbeat(
            WorkerStatus ws,
            WorkerStatus.StatusObservation observation) {
        requireStatusGeneration(ws);
        if (observation.owner() != ws) {
            throw new IllegalArgumentException(
                    "Status observation belongs to another Decode generation");
        }
        List<WorkerStatusFact> facts = new ArrayList<>(
                observation.runningTasks().size());
        admissionLock.lock();
        try {
            for (WorkerStatus.TaskObservation task
                    : observation.runningTasks().values()) {
                ReservationHandle active = workerStatusHandleLocked(
                        task.requestId());
                if (active != null) {
                    facts.add(WorkerStatusFact.active(active));
                }
            }
        } finally {
            admissionLock.unlock();
        }
        return new StatusReduction(this, facts);
    }

    private List<WorkerStatusFact> doCalibrate(
            WorkerStatus.EngineObservation engine,
            Map<String, WorkerStatus.TaskObservation> finishedTasks) {
        admissionVersion.incrementAndGet();
        List<WorkerStatusFact> facts = new ArrayList<>();

        // Build one authoritative Decode view. Claimed victims that merely
        // disappear are held synthetically. An explicit Decode finished task
        // is a separate authoritative terminal outcome: it settles the exact
        // claim without reclassifying that outcome as priority CANCELED.
        Set<Long> confirmedNow = new HashSet<>();
        Set<Long> terminalNow = new HashSet<>();
        for (WorkerStatus.TaskObservation task : finishedTasks.values()) {
            terminalNow.add(task.requestId());
        }
        int actualConfirmed = 0;
        long now = System.currentTimeMillis();
        for (WorkerStatus.TaskObservation task
                : engine.runningTaskList().values()) {
            TaskPhase phase = task.phase();
            long requestId = task.requestId();
            if (terminalNow.contains(requestId)
                    || settledTombstones.containsKey(requestId)) {
                continue;
            }
            if (phase == TaskPhase.KV_ALLOCATED || phase == TaskPhase.RUNNING) {
                actualConfirmed++;
                RequestInflight removed = inflightRequests.remove(requestId);
                removeEngineLifecycleReservationLocked(removed);
                if (removed != null) {
                    removeEngineDispatchPermitLocked(removed);
                    removeQueuedPhaseLocked(requestId, removed);
                    inflightKvReservedTotal.addAndGet(-removed.kvTokens());
                    inflightExpectedKvReservedTotal.addAndGet(
                            -removed.expectedKvTokens());
                }
                PreemptionClaim claim = preemptionClaims.get(requestId);
                if (claim != null
                        && !claim.engineFenceTransferred
                        && claim.phase == PreemptionCancelPhase.NOT_FOUND_STALE
                        && !preemptionAttempts.containsKey(claim.attemptToken)) {
                    releaseHeldKv(claim);
                    preemptionClaims.remove(requestId);
                    claim = null;
                }
                if (claim != null) {
                    claim.owner = ClaimOwner.ENGINE_CONFIRMED;
                    releaseHeldKv(claim);
                }
                confirmedNow.add(requestId);
                trackConfirmed(task, phase, removed, now);
                observeEngineFenceConfirmedLocked(requestId, removed);
                ReservationHandle accepted = workerStatusHandleLocked(requestId);
                if (accepted != null) {
                    facts.add(WorkerStatusFact.accepted(accepted));
                }
            } else {
                ReservationHandle active = workerStatusHandleLocked(requestId);
                if (active != null) {
                    facts.add(WorkerStatusFact.active(active));
                }
            }
        }

        // Terminal proof must be captured before absent-task pruning removes
        // the exact ConfirmedTask identity. Endpoint settlement happens here;
        // the downstream scheduler receives only the immutable result.
        for (WorkerStatus.TaskObservation task
                : finishedTasks.values()) {
            long requestId = task.requestId();
            if (settledTombstones.containsKey(requestId)) {
                continue;
            }
            ReservationHandle terminal = workerStatusHandleLocked(requestId);
            confirmedNow.remove(requestId);
            PreemptionClaim claim = preemptionClaims.get(requestId);
            if (claim != null) {
                if (terminal != null
                        && settlePriorityClaimTerminalLocked(
                                claim.attemptToken, terminal, claim)) {
                    facts.add(WorkerStatusFact.terminal(
                            terminal, task.errorCode()));
                } else {
                    logger.error(
                            "Decode terminal did not match its exact priority claim: "
                                    + "request_id={} generation={}",
                            requestId, getStatus().getGenerationId());
                }
                continue;
            }
            if (terminal != null) {
                facts.add(WorkerStatusFact.terminal(
                        terminal, task.errorCode()));
            }
            AuthoritativeTerminalProof proof = terminal == null
                    ? workerStatusTerminalProofLocked(requestId)
                    : new AuthoritativeTerminalProof(
                            this,
                            terminal,
                            null,
                            AuthoritativeTerminalOwner.WORKER_STATUS);
            if (proof != null) {
                settleAuthoritativeTerminalLocked(proof, now);
            } else {
                settleUntrackedWorkerTerminalLocked(requestId);
            }
        }

        int syntheticallyHeldSlots = 0;
        for (Map.Entry<Long, PreemptionClaim> entry : preemptionClaims.entrySet()) {
            PreemptionClaim claim = entry.getValue();
            if (claim.owner == ClaimOwner.ENGINE_CONFIRMED
                    && !confirmedNow.contains(entry.getKey())) {
                syntheticallyHeldSlots++;
                holdReleasedKv(claim);
            }
        }

        // Preserve one union accounting owner for a confirmed request that
        // temporarily vanished. Priority claims take precedence while live;
        // the generic fence takes over only when no priority owner exists.
        // This replaces the prior removeIf traversal, so the generic protection
        // adds no extra per-status collection or wrapper allocation.
        java.util.Iterator<Map.Entry<Long, ConfirmedTask>> confirmedIt =
                trackedConfirmed.entrySet().iterator();
        while (confirmedIt.hasNext()) {
            Map.Entry<Long, ConfirmedTask> entry = confirmedIt.next();
            long requestId = entry.getKey();
            if (confirmedNow.contains(requestId)) {
                continue;
            }
            EngineFenceProtection protection = engineFenceProtections.get(requestId);
            PreemptionClaim priorityClaim = preemptionClaims.get(requestId);
            long confirmedReservationToken = entry.getValue().reservationToken();
            if (protection != null
                    && protection.reservationToken != confirmedReservationToken) {
                protection = null;
            }
            boolean exactPriorityOwner = isPriorityAccountingOwner(priorityClaim)
                    && priorityClaim.reservationToken == confirmedReservationToken;
            if (exactPriorityOwner) {
                if (protection != null) {
                    releaseEngineFenceSyntheticHoldLocked(protection, true);
                }
                continue;
            }
            if (protection != null) {
                protection.confirmedOwner = true;
                ensureEngineFenceSyntheticHoldLocked(protection, true);
                continue;
            }
            confirmedIt.remove();
            if (!hasAnyRequestOwnerLocked(requestId)) {
                rememberSettledLocked(requestId, now);
            }
        }
        this.confirmedEngineOwnedCount = actualConfirmed + syntheticallyHeldSlots
                + engineFenceHeldSlotCount;

        return facts;
    }

    /** Caller holds {@link #admissionLock}; request-id-only facts are forbidden. */
    private ReservationHandle workerStatusHandleLocked(long requestId) {
        ConfirmedTask confirmed = trackedConfirmed.get(requestId);
        RequestInflight shadow = inflightRequests.get(requestId);
        PreemptionClaim claim = preemptionClaims.get(requestId);
        long reservationToken = claim != null
                ? claim.reservationToken
                : confirmed != null
                    ? confirmed.reservationToken()
                    : shadow != null ? shadow.reservationToken() : 0L;
        if (reservationToken <= 0L) {
            return null;
        }
        return new ReservationHandle(
                getStatus().getGenerationId(),
                requestId,
                reservationToken);
    }

    /**
     * Register / refresh one engine-confirmed task in the layered registry:
     * {@code KV_ALLOCATED} → accepted layer, {@code RUNNING} → running layer.
     * Priority is inherited from the shadow entry removed this round; when
     * the WorkerStatus report precedes the reserve (or the shadow entry
     * expired) it is unknown here and falls back to the default. KV is approximated by
     * {@code TaskObservation.inputLength} — the engine does not report per-request
     * KV usage, so 0 stays 0.
     */
    private void trackConfirmed(
            WorkerStatus.TaskObservation task,
            TaskPhase phase,
            RequestInflight removed,
            long now) {
        DecodeTaskPhase layer = phase == TaskPhase.KV_ALLOCATED
                ? DecodeTaskPhase.ACCEPTED_NOT_RUNNING
                : DecodeTaskPhase.RUNNING;
        ConfirmedTask tracked = trackedConfirmed.get(task.requestId());
        boolean priorityKnown = removed != null;
        long reservationToken = removed == null ? 0L : removed.reservationToken();
        if (tracked == null
                || (reservationToken > 0L
                    && tracked.reservationToken() != reservationToken)
                || (!tracked.priorityKnown() && priorityKnown)) {
            int priority = removed != null ? removed.priority() : RequestInflight.DEFAULT_PRIORITY;
            long kvTokens = Math.max(0, task.inputLength());
            trackedConfirmed.put(task.requestId(),
                    new ConfirmedTask(
                            priority, kvTokens, layer, now,
                            priorityKnown, reservationToken));
        } else {
            tracked.refresh(layer, now);
        }
    }

    /** Publish one non-refreshing terminal fence while admissionLock is held. */
    private boolean rememberSettledLocked(long requestId, long settledAtMs) {
        return settledTombstones.putIfAbsent(requestId, settledAtMs) == null;
    }

    /** Called with {@link #admissionLock} held after a fresh active observation. */
    private void observeEngineFenceConfirmedLocked(long requestId, RequestInflight removed) {
        EngineFenceProtection protection = engineFenceProtections.get(requestId);
        if (protection == null) {
            return;
        }
        ConfirmedTask confirmed = trackedConfirmed.get(requestId);
        boolean exactRemoved = removed != null
                && removed.reservationToken() == protection.reservationToken;
        boolean exactConfirmed = confirmed != null
                && confirmed.reservationToken() == protection.reservationToken;
        if (!exactRemoved && !exactConfirmed) {
            return;
        }
        if (exactRemoved) {
            retainEngineFenceDemandLocked(
                    protection, removed.kvTokens(), removed.expectedKvTokens());
        }
        if (exactConfirmed) {
            retainEngineFenceDemandLocked(
                    protection, confirmed.kvTokens(), confirmed.kvTokens());
        }
        protection.confirmedOwner = true;
        // The fresh engine report owns both slot and KV again. Any prior
        // generic synthetic owner must disappear before the aggregate count is
        // recomputed for this status round.
        releaseEngineFenceSyntheticHoldLocked(protection, true);
    }

    /**
     * Install the generic synthetic owner without replacing an existing slot
     * count. When {@code slotAlreadyCounted} is false this is a shadow-to-
     * synthetic transfer, so move the slot into confirmedEngineOwnedCount before
     * the shadow map entry is removed.
     */
    private void ensureEngineFenceSyntheticHoldLocked(
            EngineFenceProtection protection,
            boolean slotAlreadyCounted) {
        if (protection.syntheticHeld) {
            return;
        }
        engineFenceHeldKv.addAndGet(protection.hardKvTokens);
        engineFenceHeldExpectedKv.addAndGet(protection.expectedKvTokens);
        engineFenceHeldSlotCount++;
        protection.syntheticHeld = true;
        if (!slotAlreadyCounted) {
            confirmedEngineOwnedCount++;
        }
    }

    /** Grow a conservative demand estimate without corrupting a live hold. */
    private void retainEngineFenceDemandLocked(
            EngineFenceProtection protection,
            long hardKvTokens,
            long expectedKvTokens) {
        long previousHardKv = protection.hardKvTokens;
        long previousExpectedKv = protection.expectedKvTokens;
        protection.retainAtLeast(hardKvTokens, expectedKvTokens);
        if (protection.syntheticHeld) {
            engineFenceHeldKv.addAndGet(protection.hardKvTokens - previousHardKv);
            engineFenceHeldExpectedKv.addAndGet(
                    protection.expectedKvTokens - previousExpectedKv);
        }
    }

    /** Called with {@link #admissionLock} held. */
    private void releaseEngineFenceSyntheticHoldLocked(EngineFenceProtection protection) {
        releaseEngineFenceSyntheticHoldLocked(protection, false);
    }

    /**
     * Drop a synthetic hold. Calibration passes {@code preservePublishedSlot}
     * when a fresh engine/priority owner replaces the slot later in the same
     * critical section; keeping the old volatile count published until the
     * final aggregate assignment prevents lock-free routing from observing a
     * transient zero-load gap.
     */
    private void releaseEngineFenceSyntheticHoldLocked(
            EngineFenceProtection protection,
            boolean preservePublishedSlot) {
        if (!protection.syntheticHeld) {
            return;
        }
        engineFenceHeldKv.addAndGet(-protection.hardKvTokens);
        engineFenceHeldExpectedKv.addAndGet(-protection.expectedKvTokens);
        engineFenceHeldSlotCount--;
        if (!preservePublishedSlot) {
            confirmedEngineOwnedCount = Math.max(0, confirmedEngineOwnedCount - 1);
        }
        protection.syntheticHeld = false;
    }

    /** Called with {@link #admissionLock} held. */
    private boolean clearEngineFenceProtectionLocked(long requestId) {
        EngineFenceProtection protection = engineFenceProtections.remove(requestId);
        if (protection == null) {
            return false;
        }
        releaseEngineFenceSyntheticHoldLocked(protection);
        return true;
    }

    private static boolean isPriorityAccountingOwner(PreemptionClaim claim) {
        return claim != null;
    }

    private void holdReleasedKv(PreemptionClaim claim) {
        requirePriorityPreemptionHoldLock();
        if (claim.kvHeldAfterWorkerRelease) {
            return;
        }
        long currentHardKv = priorityPreemptionHeldKv.get();
        long currentExpectedKv = priorityPreemptionHeldExpectedKv.get();
        requirePriorityPreemptionHoldInvariant(currentHardKv, currentExpectedKv);
        long nextHardKv;
        long nextExpectedKv;
        try {
            nextHardKv = Math.addExact(currentHardKv, claim.hardKvTokens);
            nextExpectedKv = Math.addExact(
                    currentExpectedKv, claim.expectedKvTokens);
        } catch (ArithmeticException overflow) {
            throw new IllegalStateException(
                    "Priority preemption KV hold counter overflow", overflow);
        }
        requirePriorityPreemptionHoldInvariant(nextHardKv, nextExpectedKv);

        // Preserve expected >= hard even for lock-free readers between writes.
        priorityPreemptionHeldExpectedKv.set(nextExpectedKv);
        priorityPreemptionHeldKv.set(nextHardKv);
        claim.kvHeldAfterWorkerRelease = true;
    }

    private void releaseHeldKv(PreemptionClaim claim) {
        requirePriorityPreemptionHoldLock();
        if (!claim.kvHeldAfterWorkerRelease) {
            return;
        }
        long currentHardKv = priorityPreemptionHeldKv.get();
        long currentExpectedKv = priorityPreemptionHeldExpectedKv.get();
        requirePriorityPreemptionHoldInvariant(currentHardKv, currentExpectedKv);
        if (currentHardKv < claim.hardKvTokens
                || currentExpectedKv < claim.expectedKvTokens) {
            throw new IllegalStateException(
                    "Priority preemption KV hold counter underflow: hard="
                            + currentHardKv + "-" + claim.hardKvTokens
                            + ", expected=" + currentExpectedKv + "-"
                            + claim.expectedKvTokens);
        }
        long nextHardKv = currentHardKv - claim.hardKvTokens;
        long nextExpectedKv = currentExpectedKv - claim.expectedKvTokens;
        requirePriorityPreemptionHoldInvariant(nextHardKv, nextExpectedKv);

        // Preserve expected >= hard even for lock-free readers between writes.
        priorityPreemptionHeldKv.set(nextHardKv);
        priorityPreemptionHeldExpectedKv.set(nextExpectedKv);
        claim.kvHeldAfterWorkerRelease = false;
    }

    private void requirePriorityPreemptionHoldLock() {
        if (!admissionLock.isHeldByCurrentThread()) {
            throw new IllegalStateException(
                    "Priority preemption KV hold mutation requires admissionLock");
        }
    }

    private static void requirePriorityPreemptionHoldInvariant(
            long hardKvTokens, long expectedKvTokens) {
        if (hardKvTokens < 0L || expectedKvTokens < hardKvTokens) {
            throw new IllegalStateException(
                    "Invalid priority preemption KV hold counters: hard="
                            + hardKvTokens + ", expected=" + expectedKvTokens);
        }
    }

    // ==================== KV Cache 三视图 ====================

    /**
     * Local inflight KV reservation (conservative estimate) not yet confirmed by the engine.
     * Sums {@code expectedKvTokens} (seqLen + maxNewTokens) to account for generation-phase
     * KV growth. Used for scoring / load balancing.
     * Backed by {@code inflightExpectedKvReservedTotal} counter — O(1) incremental maintenance.
     */
    private long inflightKvReserved() {
        return inflightExpectedKvReservedTotal.get();
    }

    /**
     * Local inflight KV reservation (hard demand) not yet confirmed by the engine.
     * Sums {@code kvTokens} (seqLen only) — the minimum KV needed for the prompt itself.
     * Used for hard-capacity filtering to ensure the prompt fits.
     * Backed by {@code inflightKvReservedTotal} counter — O(1) incremental maintenance.
     */
    public long inflightHardKvReserved() {
        return inflightKvReservedTotal.get();
    }

    /**
     * Real KV used: engine-reported used (total - available), local inflight
     * reservations, and expected demand retained by a synthetic EngineFence owner.
     */
    public long realKvUsed() {
        WorkerStatus.EngineObservation status =
                getStatus().committedEngineObservation();
        long totalCap = status.totalKvCacheTokens();
        long avail = status.availableKvCacheTokens();
        long reportedUsed = totalCap > 0 ? Math.max(0, totalCap - avail) : 0;
        return saturatedAddNonNegative(
                saturatedAddNonNegative(reportedUsed, inflightKvReserved()),
                saturatedAddNonNegative(
                        priorityPreemptionHeldExpectedKv.get(),
                        engineFenceHeldExpectedKv.get()));
    }

    /**
     * KV demand which may reach the engine now. Reservations parked in a
     * Prefill queue are soft placement hints: charging all of them against the
     * hard availability gate makes a long scheduler queue report every Decode
     * worker unavailable even though none of that work has been dispatched.
     */
    public long engineFacingKvUsed() {
        WorkerStatus.EngineObservation fields =
                getStatus().committedWorkerStatus().fields();
        return engineFacingKvUsed(fields);
    }

    private long engineFacingKvUsed(
            WorkerStatus.EngineObservation fields) {
        long totalCap = fields.totalKvCacheTokens();
        long avail = fields.availableKvCacheTokens();
        long reportedUsed = totalCap > 0 ? Math.max(0, totalCap - avail) : 0;
        long localEngineFacing = Math.max(0L,
                inflightKvReserved() - queuedExpectedKvReservedTotal.get())
                + engineDispatchPermitExpectedKvReservedTotal.get();
        return saturatedAddNonNegative(
                saturatedAddNonNegative(reportedUsed, localEngineFacing),
                saturatedAddNonNegative(
                        priorityPreemptionHeldExpectedKv.get(),
                        engineFenceHeldExpectedKv.get()));
    }

    /**
     * Real KV available: engine-reported available minus local shadow,
     * priority-fence, and generic EngineFence hard reservations.
     *
     * <p>Uses {@link #inflightHardKvReserved()} (prompt-only KV) rather than
     * {@link #inflightKvReserved()} (expected KV with generation) so that the
     * hard-capacity filter only checks whether the prompt itself fits, without
     * being overly aggressive due to other inflight requests' expected growth.
     *
     * <p><b>Approximate:</b> reads one committed WorkerStatus holder and local
     * counters non-atomically. Authoritative acquisition repeats the same math
     * under {@link #admissionLock}.
     */
    public long realKvAvailable() {
        WorkerStatus.EngineObservation fields =
                getStatus().committedWorkerStatus().fields();
        return Math.max(0, fields.availableKvCacheTokens()
                - inflightHardKvReserved()
                - priorityPreemptionHeldKv.get()
                - engineFenceHeldKv.get());
    }

    /** Hard prompt KV available to the next engine dispatch, excluding soft queued holds. */
    public long engineFacingKvAvailable() {
        WorkerStatus.EngineObservation fields =
                getStatus().committedWorkerStatus().fields();
        return engineFacingKvAvailable(fields);
    }

    private long engineFacingKvAvailable(
            WorkerStatus.EngineObservation fields) {
        long localEngineFacing = Math.max(0L,
                inflightHardKvReserved() - queuedHardKvReservedTotal.get())
                + engineDispatchPermitHardKvReservedTotal.get();
        return Math.max(0, fields.availableKvCacheTokens()
                - localEngineFacing
                - priorityPreemptionHeldKv.get()
                - engineFenceHeldKv.get());
    }

    // ==================== Metrics ====================

    /**
     * Report per-worker decode inflight metrics via the given reporter.
     * Called periodically by {@link org.flexlb.balance.scheduler.RequestScheduler}.
     */
    public void reportBatchMetrics(BatchSchedulerReporter reporter) {
        reporter.reportInflightRequestCount(RoleType.DECODE.name(), getIp(), getInflightCount());
        reporter.reportDecodeTotalLoad(getIp(), getTotalLoad());
        reporter.reportDecodeInflightKvReserved(getIp(), inflightKvReserved());
        reporter.reportDecodeInflightHardKvReserved(getIp(), inflightHardKvReserved());
        reporter.reportInflightMaxAgeMs(RoleType.DECODE.name(), getIp(),
                TtlEvictor.maxAgeMs(inflightRequests, System.currentTimeMillis()));
    }

    /** Report one consistent phase-split admission snapshot for this endpoint. */
    public void reportAdmissionMetrics(RequestSchedulerReporter reporter) {
        LayeredAdmissionView view = layeredAdmissionView();
        String endpoint = ipPort();
        reporter.reportDecodeReservedCount(endpoint, view.reserved().size());
        reporter.reportDecodeShadowKvReserved(
                endpoint, view.routing().inflightHardKv());
        reporter.reportDecodeRunningCount(endpoint, view.runningCount());
        reporter.reportDecodeAcceptedCount(endpoint, view.acceptedCount());
        reporter.reportDecodeEngineLoad(endpoint, view.routing().engineLoad());
    }

    /**
     * Real KV total capacity reported by the engine.
     */
    public long realKvTotal() {
        return getStatus().getTotalKvCacheTokens();
    }

    public int getInflightCount() {
        return inflightRequests.size();
    }

    /** Evict only endpoint orphans which have no live scheduler generation. */
    public int evictExpiredRequests(long ttlMs,
                                    LongPredicate schedulerOwnsRequest) {
        int evicted;
        boolean capacityChanged;
        admissionLock.lock();
        try {
            // A priority claim or generic EngineFence is a stronger accounting
            // owner than age-only cleanup. In particular, an ambiguous
            // ENGINE_MAY_HAVE_SEEN shadow remains charged until reconciliation.
            evicted = requestEvictor.evictExpired(
                    ttlMs, requestId -> !schedulerOwnsRequest.test(requestId)
                            && !preemptionClaims.containsKey(requestId)
                            && !engineFenceProtections.containsKey(requestId));
            long cutoff = System.currentTimeMillis() - ttlMs;
            int trackedPurged = 0;
            java.util.Iterator<Map.Entry<Long, ConfirmedTask>> trackedEvictIt =
                    trackedConfirmed.entrySet().iterator();
            while (trackedEvictIt.hasNext()) {
                Map.Entry<Long, ConfirmedTask> entry = trackedEvictIt.next();
                if (entry.getValue().lastSeenMs() < cutoff
                        && !schedulerOwnsRequest.test(entry.getKey())
                        && !preemptionClaims.containsKey(entry.getKey())
                        && !engineFenceProtections.containsKey(entry.getKey())) {
                    trackedEvictIt.remove();
                    trackedPurged++;
                }
            }
            if (trackedPurged > 0) {
                confirmedEngineOwnedCount = Math.max(
                        0, confirmedEngineOwnedCount - trackedPurged);
            }
            boolean settledTombstonesPurged = settledTombstones.entrySet()
                    .removeIf(entry -> entry.getValue() < cutoff);
            if (evicted > 0 || trackedPurged > 0 || settledTombstonesPurged) {
                admissionVersion.incrementAndGet();
            }
            capacityChanged = evicted > 0 || trackedPurged > 0
                    || settledTombstonesPurged;
        } finally {
            admissionLock.unlock();
        }
        if (capacityChanged) {
            notifyEngineDispatchCapacityListeners();
            signalPlacementCapacityChanged();
        }
        return evicted;
    }

    public int getTotalLoad() {
        return confirmedEngineOwnedCount + inflightRequests.size();
    }

    /**
     * Engine-facing load (N2): confirmed running/accepted requests plus
     * reserved entries that are <b>not</b> parked in a prefill queue. Queued
     * reservations remain in the full placement/priority view, but they must
     * not close the Decode concurrency gate while work is still waiting in
     * Prefill. {@link #getTotalLoad()} keeps the full shadow view for
     * observability and eviction planning.
     *
     * <p>O(1) formula (PR-C): {@code confirmedEngineOwnedCount
     * + max(0, inflightRequests.size() − queuedPhaseCount)}. The
     * {@link #queuedPhaseCount} counter is updated with every queued-phase
     * transition, so this admission read does not scan the queued set.
     *
     * <p><b>Torn-read safety:</b> the counter is read lock-free and may
     * transiently fall outside {@code [0, inflight]} while a lock-owned
     * transition publishes its two fields. Clamp that expected observation;
     * exact admission still revalidates both values under the lock.
     */
    public int getEngineLoad() {
        int inflight = inflightRequests.size();
        int queued = queuedPhaseCount.get();
        if (queued < 0 || queued > inflight) {
            queued = Math.max(0, Math.min(queued, inflight));
        }
        return confirmedEngineOwnedCount + Math.max(0, inflight - queued);
    }

    /** Exact outcome of moving one Decode reservation into the Prefill queue. */
    public enum MarkQueuedResult {
        MARKED,
        ALREADY_QUEUED,
        NOT_OWNED
    }

    /**
     * Move the exact reservation owned by a queue admission into the queued
     * phase. The generation pin makes a capture accepted before detach valid
     * through this transition; the reservation token fences request-id reuse.
     */
    public MarkQueuedResult markQueuedExact(
            GenerationPin generationPin,
            ReservationHandle reservation) {
        if (reservation == null) {
            throw new IllegalArgumentException(
                    "Decode reservation is required for queued transition");
        }
        boolean capacityChanged = false;
        MarkQueuedResult result;
        admissionLock.lock();
        try {
            requirePinnedGeneration(generationPin);
            if (reservation.endpointGenerationId()
                    != getStatus().getGenerationId()) {
                return MarkQueuedResult.NOT_OWNED;
            }
            long requestId = reservation.requestId();
            RequestInflight current = inflightRequests.get(requestId);
            if (!isExactReservation(current, reservation)) {
                return MarkQueuedResult.NOT_OWNED;
            }
            if (current.queued()) {
                return MarkQueuedResult.ALREADY_QUEUED;
            }
            addQueuedPhaseLocked(requestId, current);
            // Re-queueing begins a new dispatch round. Invalidate any
            // pre-delivery lease before publishing that transition.
            removeEngineDispatchPermitLocked(requestId);
            admissionVersion.incrementAndGet();
            capacityChanged = true;
            result = MarkQueuedResult.MARKED;
        } finally {
            admissionLock.unlock();
        }
        if (capacityChanged) {
            notifyEngineDispatchCapacityListeners();
        }
        return result;
    }

    private boolean addQueuedPhaseLocked(long requestId, RequestInflight reservation) {
        if (reservation == null || !reservation.markQueued()) {
            return false;
        }
        queuedPhaseCount.incrementAndGet();
        queuedHardKvReservedTotal.addAndGet(reservation.kvTokens());
        queuedExpectedKvReservedTotal.addAndGet(reservation.expectedKvTokens());
        return true;
    }

    private boolean removeQueuedPhaseLocked(long requestId, RequestInflight reservation) {
        if (reservation == null) {
            throw new IllegalStateException(
                    "queued Decode reservation missing for request " + requestId);
        }
        if (!reservation.clearQueued()) {
            return false;
        }
        queuedPhaseCount.decrementAndGet();
        queuedHardKvReservedTotal.addAndGet(-reservation.kvTokens());
        queuedExpectedKvReservedTotal.addAndGet(-reservation.expectedKvTokens());
        return true;
    }

    /** Outcome of acquiring a pre-delivery Decode slot. */
    public enum EngineDispatchPermitAcquireStatus {
        /** A token-fenced permit now owns one Decode hard-gate slot. */
        ACQUIRED,
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

    /** Outcome of transferring an acquired permit to engine lifecycle ownership. */
    public enum EngineDispatchPermitTransferStatus {
        TRANSFERRED,
        OWNERSHIP_LOST,
        ENDPOINT_RETIRED
    }

    /** Explicit result of {@link #acquireEngineDispatchPermit(long, long)}. */
    public record EngineDispatchPermitAcquisition(
            EngineDispatchPermitAcquireStatus status,
            ReservationHandle reservation,
            EngineDispatchPermit permit) {

        public EngineDispatchPermitAcquisition {
            if (status == null) {
                throw new IllegalArgumentException("permit acquisition status is required");
            }
            if ((status == EngineDispatchPermitAcquireStatus.ACQUIRED) != (permit != null)) {
                throw new IllegalArgumentException(
                        "only an ACQUIRED result may carry an engine dispatch permit");
            }
            if (status != EngineDispatchPermitAcquireStatus.ACQUIRED
                    && reservation != null) {
                throw new IllegalArgumentException(
                        "a rejected result cannot carry a reservation");
            }
        }
    }

    /**
     * Token-fenced reservation of one Decode concurrency slot.
     *
     * <p>{@link #transferToEngineLifecycle()} atomically converts the still-current queued
     * reservation into engine-facing ownership without checking capacity again.
     * {@link #release()} gives up only this token's temporary hard-gate slot;
     * the reservation stays queued. Both operations are idempotent with respect
     * to endpoint state, and an old request-id generation cannot affect a newer
     * permit.
     */
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
        private final long token;
        private final RequestInflight reservation;
        private Resolution resolution = Resolution.ACQUIRED;

        private EngineDispatchPermit(DecodeEndpoint endpoint,
                                     long requestId,
                                     long token,
                                     RequestInflight reservation) {
            this.endpoint = endpoint;
            this.requestId = requestId;
            this.token = token;
            this.reservation = reservation;
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
        public synchronized EngineDispatchPermitTransferStatus
                transferToEngineLifecycle() {
            if (resolution == Resolution.ENGINE_LIFECYCLE_OWNED) {
                return EngineDispatchPermitTransferStatus.TRANSFERRED;
            }
            if (resolution == Resolution.ENDPOINT_RETIRED) {
                return EngineDispatchPermitTransferStatus.ENDPOINT_RETIRED;
            }
            if (resolution != Resolution.ACQUIRED) {
                return EngineDispatchPermitTransferStatus.OWNERSHIP_LOST;
            }
            EngineDispatchPermitTransferStatus transfer =
                    endpoint.transferEngineDispatchPermitToLifecycle(this);
            if (transfer == EngineDispatchPermitTransferStatus.TRANSFERRED) {
                resolution = Resolution.ENGINE_LIFECYCLE_OWNED;
            } else if (transfer == EngineDispatchPermitTransferStatus.ENDPOINT_RETIRED) {
                resolution = Resolution.ENDPOINT_RETIRED;
            } else {
                resolution = Resolution.INVALIDATED;
            }
            return transfer;
        }

        /**
         * Release this acquired slot while leaving the same reservation queued.
         *
         * @return {@code true} only when this invocation removed the live permit;
         *         repeated or stale releases return {@code false}
         */
        public synchronized boolean release() {
            if (resolution != Resolution.ACQUIRED) {
                return false;
            }
            if (endpoint.releaseEngineDispatchPermit(this)) {
                resolution = Resolution.RELEASED;
                return true;
            }
            resolution = Resolution.INVALIDATED;
            return false;
        }

    }

    /**
     * Acquire one pre-delivery Decode slot while the reservation remains queued.
     * Decode concurrency and KV are validated and occupied under the same
     * admission lock, so concurrent acquisitions cannot oversell either gate.
     */
    public EngineDispatchPermitAcquisition acquireEngineDispatchPermit(
            long requestId, long concurrencyLimit) {
        return acquireEngineDispatchPermit(requestId, concurrencyLimit, -1L);
    }

    public EngineDispatchPermitAcquisition acquireEngineDispatchPermit(
            long requestId,
            long concurrencyLimit,
            long maxKvUsagePercent) {
        GenerationPin generationPin = tryPinGeneration();
        if (generationPin == null) {
            return rejectedEngineDispatchPermit(
                    EngineDispatchPermitAcquireStatus.ENDPOINT_RETIRED);
        }
        try (generationPin) {
            return acquireEngineDispatchPermitPinned(
                    requestId, concurrencyLimit, maxKvUsagePercent);
        }
    }

    /**
     * Atomically reserve a newly selected queued request and acquire its exact
     * pre-delivery permit. This is the Decode-side commit point used when a
     * Prefill queue head moves away from a binding that became full after the
     * request was enqueued.
     */
    public EngineDispatchPermitAcquisition
            tryAcquireQueuedEngineDispatchPermitPinned(
            GenerationPin pin,
            long requestId,
            long hardKvTokens,
            long expectedKvTokens,
            int priority,
            long concurrencyLimit,
            long maxKvUsagePercent) {
        admissionLock.lock();
        try {
            requirePinnedGeneration(pin);
            if (!requestIdAvailableForReservationLocked(requestId)) {
                return rejectedEngineDispatchPermit(
                        EngineDispatchPermitAcquireStatus.NOT_OWNED);
            }
            RequestInflight projected = new RequestInflight(
                    hardKvTokens,
                    expectedKvTokens,
                    priority,
                    0L);
            if (isEngineDispatchCapacityFullLocked(
                    projected, concurrencyLimit, maxKvUsagePercent)) {
                return rejectedEngineDispatchPermit(
                        EngineDispatchPermitAcquireStatus.CAPACITY_FULL);
            }

            ReservationHandle reservation = null;
            try {
                reservation = reserveLocked(
                        requestId,
                        hardKvTokens,
                        expectedKvTokens,
                        priority);
                RequestInflight exact = inflightRequests.get(requestId);
                addQueuedPhaseLocked(requestId, exact);
                EngineDispatchPermit permit = installEngineDispatchPermitLocked(
                        requestId, exact);
                return new EngineDispatchPermitAcquisition(
                        EngineDispatchPermitAcquireStatus.ACQUIRED,
                        reservation,
                        permit);
            } catch (Throwable failure) {
                if (reservation != null) {
                    try {
                        releaseLocked(requestId);
                    } catch (Throwable rollbackFailure) {
                        if (rollbackFailure != failure) {
                            failure.addSuppressed(rollbackFailure);
                        }
                    }
                }
                throw failure;
            }
        } finally {
            admissionLock.unlock();
        }
    }

    private EngineDispatchPermitAcquisition acquireEngineDispatchPermitPinned(
            long requestId,
            long concurrencyLimit,
            long maxKvUsagePercent) {
        admissionLock.lock();
        try {
            RequestInflight reservation = inflightRequests.get(requestId);
            if (reservation == null || preemptionClaims.containsKey(requestId)) {
                return rejectedEngineDispatchPermit(
                        EngineDispatchPermitAcquireStatus.NOT_OWNED);
            }
            if (!reservation.queued()) {
                return rejectedEngineDispatchPermit(
                        EngineDispatchPermitAcquireStatus.NOT_QUEUED);
            }
            if (reservation.dispatchPermit() != null) {
                return rejectedEngineDispatchPermit(
                        EngineDispatchPermitAcquireStatus.ALREADY_ACQUIRED);
            }
            if (isEngineDispatchCapacityFullLocked(
                    reservation, concurrencyLimit, maxKvUsagePercent)) {
                return rejectedEngineDispatchPermit(
                        EngineDispatchPermitAcquireStatus.CAPACITY_FULL);
            }

            EngineDispatchPermit permit = installEngineDispatchPermitLocked(
                    requestId, reservation);
            return new EngineDispatchPermitAcquisition(
                    EngineDispatchPermitAcquireStatus.ACQUIRED, null, permit);
        } finally {
            admissionLock.unlock();
        }
    }

    private static EngineDispatchPermitAcquisition rejectedEngineDispatchPermit(
            EngineDispatchPermitAcquireStatus status) {
        return new EngineDispatchPermitAcquisition(status, null, null);
    }

    /** Caller holds admissionLock and has validated exact queued ownership. */
    private EngineDispatchPermit installEngineDispatchPermitLocked(
            long requestId,
            RequestInflight reservation) {
        long token = nextEngineDispatchPermitTokenLocked();
        EngineDispatchPermit permit = new EngineDispatchPermit(
                this, requestId, token, reservation);
        reservation.installDispatchPermit(permit);
        activeEngineDispatchPermitCount++;
        engineDispatchPermitHardKvReservedTotal.addAndGet(
                reservation.kvTokens());
        engineDispatchPermitExpectedKvReservedTotal.addAndGet(
                reservation.expectedKvTokens());
        admissionVersion.incrementAndGet();
        return permit;
    }

    private EngineDispatchPermitTransferStatus transferEngineDispatchPermitToLifecycle(
            EngineDispatchPermit permit) {
        GenerationPin generationPin = tryPinGeneration();
        if (generationPin == null) {
            return EngineDispatchPermitTransferStatus.ENDPOINT_RETIRED;
        }
        try (generationPin) {
            return transferEngineDispatchPermitToLifecyclePinned(permit);
        }
    }

    private EngineDispatchPermitTransferStatus
            transferEngineDispatchPermitToLifecyclePinned(
            EngineDispatchPermit permit) {
        EngineDispatchPermitTransferStatus transferStatus;
        boolean capacityIncreased;
        admissionLock.lock();
        try {
            int usageBefore = engineDispatchHardGateUsageLocked();
            if (retiredEngineDispatchPermitTokens.remove(permit.token)) {
                return EngineDispatchPermitTransferStatus.ENDPOINT_RETIRED;
            }
            if (!isCurrentEngineDispatchPermitLocked(permit)) {
                return EngineDispatchPermitTransferStatus.OWNERSHIP_LOST;
            }
            if (inflightRequests.get(permit.requestId) != permit.reservation
                    || !permit.reservation.queued()
                    || preemptionClaims.containsKey(permit.requestId)) {
                removeEngineDispatchPermitLocked(permit.requestId);
                admissionVersion.incrementAndGet();
                transferStatus = EngineDispatchPermitTransferStatus.OWNERSHIP_LOST;
            } else {
                removeEngineDispatchPermitLocked(permit.requestId);
                // The identity and queued membership were checked while holding the
                // same lock. Transfer only changes ownership; it never re-reads the cap.
                removeQueuedPhaseLocked(permit.requestId, permit.reservation);
                engineLifecycleReservationTokens.add(
                        permit.reservation.reservationToken());
                admissionVersion.incrementAndGet();
                transferStatus = EngineDispatchPermitTransferStatus.TRANSFERRED;
            }
            capacityIncreased = engineDispatchHardGateUsageLocked() < usageBefore;
        } finally {
            admissionLock.unlock();
        }
        // Capacity listeners may acquire scheduler queue locks. Keep that
        // one-way notification strictly outside the endpoint admission lock.
        if (capacityIncreased) {
            notifyEngineDispatchCapacityListeners();
        }
        return transferStatus;
    }

    private boolean releaseEngineDispatchPermit(EngineDispatchPermit permit) {
        boolean released = false;
        admissionLock.lock();
        try {
            if (retiredEngineDispatchPermitTokens.remove(permit.token)) {
                return true;
            }
            if (!isCurrentEngineDispatchPermitLocked(permit)) {
                return false;
            }
            removeEngineDispatchPermitLocked(permit.requestId);
            admissionVersion.incrementAndGet();
            released = true;
        } finally {
            admissionLock.unlock();
        }
        if (released) {
            notifyEngineDispatchCapacityListeners();
            signalPlacementCapacityChanged();
        }
        return true;
    }

    private boolean isCurrentEngineDispatchPermitLocked(EngineDispatchPermit permit) {
        return permit.reservation.dispatchPermit() == permit
                && inflightRequests.get(permit.requestId) == permit.reservation;
    }

    private boolean removeEngineDispatchPermitLocked(long requestId) {
        RequestInflight reservation = inflightRequests.get(requestId);
        return reservation != null
                && removeEngineDispatchPermitLocked(reservation);
    }

    private boolean removeEngineDispatchPermitLocked(
            RequestInflight reservation) {
        EngineDispatchPermit removed = reservation.clearDispatchPermit();
        if (removed == null) {
            return false;
        }
        engineDispatchPermitHardKvReservedTotal.addAndGet(
                -removed.reservation.kvTokens());
        engineDispatchPermitExpectedKvReservedTotal.addAndGet(
                -removed.reservation.expectedKvTokens());
        decrementActiveEngineDispatchPermitCountLocked();
        return true;
    }

    /** Caller holds admissionLock. */
    private void decrementActiveEngineDispatchPermitCountLocked() {
        activeEngineDispatchPermitCount--;
        if (activeEngineDispatchPermitCount < 0) {
            throw new IllegalStateException("negative active Decode dispatch permit count");
        }
    }

    private long nextEngineDispatchPermitTokenLocked() {
        if (nextEngineDispatchPermitToken <= 0L
                || nextEngineDispatchPermitToken == Long.MAX_VALUE) {
            throw new IllegalStateException("Decode dispatch permit token space exhausted");
        }
        return nextEngineDispatchPermitToken++;
    }

    private long nextReservationTokenLocked() {
        if (nextReservationToken <= 0L
                || nextReservationToken == Long.MAX_VALUE) {
            throw new IllegalStateException(
                    "Decode reservation token space exhausted");
        }
        return nextReservationToken++;
    }

    private int engineDispatchHardGateUsageLocked() {
        int engineFacingInflight = Math.max(0,
                inflightRequests.size() - queuedPhaseCount.get());
        return confirmedEngineOwnedCount + engineFacingInflight
                + activeEngineDispatchPermitCount;
    }

    /** Exact capacity gate for a reservation which remains Prefill-queued. */
    private boolean queuedPlacementIsFullLocked(
            long hardKvTokens,
            long expectedKvTokens,
            AdmissionCapacity capacity) {
        // This method is used only by the preemptive placement path. Unlike
        // the non-preemptive soft queue hold, its exact queued reservation
        // must consume the placement slot immediately; otherwise one status
        // edge can publish an executor-width burst before the first batcher
        // acquires its delivery permit.
        long occupiedSlots = confirmedEngineOwnedCount
                + inflightRequests.size();
        if (capacity.maxEngineRequests() > 0L
                && occupiedSlots >= capacity.maxEngineRequests()) {
            return true;
        }

        WorkerStatus.EngineObservation status =
                getStatus().committedWorkerStatus().fields();
        long totalKv = status.totalKvCacheTokens();
        if (totalKv <= 0L) {
            return false;
        }
        long hardAvailable = Math.max(
                0L,
                status.availableKvCacheTokens()
                        - inflightKvReservedTotal.get()
                        - priorityPreemptionHeldKv.get()
                        - engineFenceHeldKv.get());
        if (hardKvTokens > hardAvailable) {
            return true;
        }
        if (capacity.maxKvUsagePercent() == 0L) {
            return false;
        }
        long reportedUsed = Math.max(
                0L, totalKv - status.availableKvCacheTokens());
        long expectedUsed = saturatedAddNonNegative(
                saturatedAddNonNegative(
                        reportedUsed, inflightExpectedKvReservedTotal.get()),
                saturatedAddNonNegative(
                        priorityPreemptionHeldExpectedKv.get(),
                        engineFenceHeldExpectedKv.get()));
        long projected = saturatedAddNonNegative(
                expectedUsed, expectedKvTokens);
        return (double) projected * 100.0
                > (double) capacity.maxKvUsagePercent() * (double) totalKv;
    }

    /** Caller holds admissionLock; this is the authoritative pre-admission gate. */
    private boolean isEngineDispatchCapacityFullLocked(
            RequestInflight candidate,
            long concurrencyLimit,
            long maxKvUsagePercent) {
        WorkerStatus.CommittedWorkerStatus committed =
                getStatus().committedWorkerStatus();
        return isEngineDispatchCapacityFullSnapshot(
                candidate,
                concurrencyLimit,
                maxKvUsagePercent,
                committed.fields());
    }

    /**
     * Authoritative post-eviction projection. The caller has already validated
     * every exact victim under {@link #admissionLock}; this method evaluates
     * current canonical owners minus those victims plus the incoming owner
     * before any claim, removal, or reservation is installed.
     */
    private boolean projectedEvictionCapacityFitsLocked(
            AdmissionCapacity capacity,
            long incomingHardKv,
            long incomingExpectedKv,
            long freedSlots,
            long freedHardKv,
            long freedExpectedUsage) {
        if (incomingHardKv < 0L || incomingExpectedKv < incomingHardKv
                || freedSlots < 0L || freedHardKv < 0L
                || freedExpectedUsage < 0L) {
            throw new IllegalArgumentException(
                    "Decode eviction capacity projection requires non-negative demand");
        }
        long currentSlots = engineDispatchHardGateUsageLocked();
        long projectedSlots = Math.max(0L, currentSlots - freedSlots) + 1L;
        if (capacity.maxEngineRequests() > 0L
                && projectedSlots > capacity.maxEngineRequests()) {
            return false;
        }

        WorkerStatus.EngineObservation fields =
                getStatus().committedWorkerStatus().fields();
        long totalKv = fields.totalKvCacheTokens();
        if (totalKv <= 0L) {
            return true;
        }
        long priorityHeldHardKv = priorityPreemptionHeldKv.get();
        long priorityHeldExpectedKv =
                priorityPreemptionHeldExpectedKv.get();
        requirePriorityPreemptionHoldInvariant(
                priorityHeldHardKv, priorityHeldExpectedKv);
        long currentHardCharges = saturatedAddNonNegative(
                saturatedAddNonNegative(
                        inflightKvReservedTotal.get(),
                        priorityHeldHardKv),
                engineFenceHeldKv.get());
        long projectedHardSupply = saturatedAddNonNegative(
                Math.max(0L, fields.availableKvCacheTokens()), freedHardKv);
        long projectedHardDemand = saturatedAddNonNegative(
                currentHardCharges, incomingHardKv);
        if (projectedHardDemand > projectedHardSupply) {
            return false;
        }
        if (capacity.maxKvUsagePercent() == 0L) {
            return true;
        }

        long reportedUsed = Math.max(0L,
                fields.totalKvCacheTokens() - fields.availableKvCacheTokens());
        long currentExpectedUsage = saturatedAddNonNegative(
                saturatedAddNonNegative(
                        reportedUsed, inflightExpectedKvReservedTotal.get()),
                saturatedAddNonNegative(
                        priorityHeldExpectedKv,
                        engineFenceHeldExpectedKv.get()));
        long usageAfterVictims = Math.max(
                0L, currentExpectedUsage - freedExpectedUsage);
        long projectedExpectedUsage = saturatedAddNonNegative(
                usageAfterVictims, incomingExpectedKv);
        return (double) projectedExpectedUsage * 100.0
                <= (double) capacity.maxKvUsagePercent() * (double) totalKv;
    }

    /**
     * Common O(1) gate math used by authoritative acquisition and the live
     * waiter hint. Acquired permits are already included in both the slot and
     * KV counters; {@code candidate} is still a queued soft reservation and is
     * projected exactly once here.
     */
    private boolean isEngineDispatchCapacityFullSnapshot(
            RequestInflight candidate,
            long concurrencyLimit,
            long maxKvUsagePercent,
            WorkerStatus.EngineObservation fields) {
        return !engineDispatchCapacityFits(
                getEngineLoad() + Math.max(0, activeEngineDispatchPermitCount),
                engineFacingKvAvailable(fields),
                engineFacingKvUsed(fields),
                fields.totalKvCacheTokens(),
                new EngineDispatchDemand(
                        candidate.kvTokens(), candidate.expectedKvTokens()),
                concurrencyLimit,
                maxKvUsagePercent);
    }

    /** Single policy kernel shared by selection projections and exact admission. */
    private static boolean engineDispatchCapacityFits(
            long occupiedSlots,
            long hardKvAvailable,
            long expectedKvUsed,
            long totalKv,
            EngineDispatchDemand demand,
            long concurrencyLimit,
            long maxKvUsagePercent) {
        if (concurrencyLimit > 0L && occupiedSlots >= concurrencyLimit) {
            return false;
        }
        // A negative percentage is the legacy concurrency-only overload.
        if (maxKvUsagePercent < 0L || totalKv <= 0L) {
            return true;
        }
        if (demand.hardKvTokens() > Math.max(0L, hardKvAvailable)) {
            return false;
        }
        if (maxKvUsagePercent == 0L) {
            return true;
        }
        long projectedExpectedKv = saturatedAddNonNegative(
                Math.max(0L, expectedKvUsed), demand.expectedKvTokens());
        return (double) projectedExpectedKv * 100.0
                <= (double) maxKvUsagePercent * (double) totalKv;
    }

    /** Saturating addition for non-negative admission counters. */
    private static long saturatedAddNonNegative(long left, long right) {
        if (left < 0 || right < 0) {
            throw new IllegalArgumentException("KV admission counters must be non-negative");
        }
        return left > Long.MAX_VALUE - right ? Long.MAX_VALUE : left + right;
    }

    /**
     * Lock-free live predicate for a WorkerBatcher capacity wait.
     *
     * <p>The queue condition invokes this method while holding its own lock, so
     * this method must never acquire {@link #admissionLock}. A torn snapshot can
     * only cause an extra authoritative acquisition attempt; every capacity or
     * ownership transition publishes a listener wake after its atomic/volatile
     * counters are updated. Returning {@code true} for retirement or ownership
     * loss lets the worker resume and obtain the exact typed acquisition result.
     */
    public boolean isEngineDispatchPermitAvailable(
            long requestId,
            long concurrencyLimit,
            long maxKvUsagePercent) {
        if (isGenerationRetiringOrRetired()) {
            return true;
        }
        RequestInflight candidate = inflightRequests.get(requestId);
        if (candidate == null
                || !candidate.queued()
                || candidate.dispatchPermit() != null) {
            return true;
        }
        WorkerStatus.CommittedWorkerStatus committed =
                getStatus().committedWorkerStatus();
        return !isEngineDispatchCapacityFullSnapshot(
                candidate,
                concurrencyLimit,
                maxKvUsagePercent,
                committed.fields());
    }

    @Override
    public OptionalLong getLoadMetric() {
        return OptionalLong.of(getTotalLoad());
    }

    /**
     * Mutable layered-registry entry for one engine-confirmed request
     * (Phase 5). Identity fields (priority / KV estimate) are fixed
     * at first sight; {@code phase} and {@code lastSeenMs} are volatile and
     * only mutated under {@link #admissionLock} by calibration.
     */
    static final class ConfirmedTask {

        private final int priority;
        private final long kvTokens;
        private final boolean priorityKnown;
        /** Zero only for Engine tasks never admitted through this endpoint. */
        private final long reservationToken;
        private volatile DecodeTaskPhase phase;
        private volatile long lastSeenMs;

        ConfirmedTask(int priority, long kvTokens,
                      DecodeTaskPhase layer, long now, boolean priorityKnown,
                      long reservationToken) {
            this.priority = priority;
            this.kvTokens = kvTokens;
            this.priorityKnown = priorityKnown;
            this.reservationToken = reservationToken;
            this.phase = layer;
            this.lastSeenMs = now;
        }

        int priority() { return priority; }
        long kvTokens() { return kvTokens; }
        boolean priorityKnown() { return priorityKnown; }
        long reservationToken() { return reservationToken; }
        DecodeTaskPhase phase() { return phase; }
        long lastSeenMs() { return lastSeenMs; }

        /** Refresh layer membership and liveness on every calibrate round. */
        void refresh(DecodeTaskPhase layer, long now) {
            this.phase = layer;
            this.lastSeenMs = now;
        }
    }

    /**
     * Mutable generic-fence owner. Every field is accessed only while holding
     * {@link #admissionLock}; routing reads only the aggregate atomics/volatile
     * counters, never these entries.
     */
    private static final class EngineFenceProtection {
        private final long reservationToken;
        private long hardKvTokens;
        private long expectedKvTokens;
        private boolean confirmedOwner;
        private boolean syntheticHeld;

        private EngineFenceProtection(long reservationToken,
                                      long hardKvTokens,
                                      long expectedKvTokens,
                                      boolean confirmedOwner) {
            if (reservationToken <= 0L) {
                throw new IllegalArgumentException(
                        "Engine fence requires an exact Decode reservation");
            }
            this.reservationToken = reservationToken;
            this.hardKvTokens = Math.max(0, hardKvTokens);
            this.expectedKvTokens = Math.max(this.hardKvTokens, expectedKvTokens);
            this.confirmedOwner = confirmedOwner;
        }

        private void retainAtLeast(long hardKvTokens, long expectedKvTokens) {
            this.hardKvTokens = Math.max(this.hardKvTokens, Math.max(0, hardKvTokens));
            this.expectedKvTokens = Math.max(
                    this.expectedKvTokens,
                    Math.max(this.hardKvTokens, expectedKvTokens));
        }
    }

    private enum ClaimOwner {
        SHADOW_IN_FLIGHT,
        ENGINE_CONFIRMED
    }

    private static final class PreemptionClaim {
        private final long attemptToken;
        private ClaimOwner owner;
        /** Exact reservation identity; zero only for an untracked Engine task. */
        private final long reservationToken;
        private final long hardKvTokens;
        private final long expectedKvTokens;
        private PreemptionCancelPhase phase = PreemptionCancelPhase.CLAIMED;
        /** Endpoint accounting moved from this attempt to the Engine fence. */
        private boolean engineFenceTransferred;
        private boolean kvHeldAfterWorkerRelease;

        private PreemptionClaim(long attemptToken, ClaimOwner owner,
                                long reservationToken,
                                long hardKvTokens, long expectedKvTokens) {
            if (hardKvTokens < 0L || expectedKvTokens < hardKvTokens) {
                throw new IllegalArgumentException(
                        "Priority preemption claim requires expected KV >= hard KV >= 0");
            }
            this.attemptToken = attemptToken;
            this.owner = owner;
            this.reservationToken = reservationToken;
            this.hardKvTokens = hardKvTokens;
            this.expectedKvTokens = expectedKvTokens;
        }
    }

    private static final class EndpointPreemptionAttempt {
        private final long incomingRequestId;
        private final long incomingReservationToken;
        private final Map<Long, ReservationHandle> remainingVictims;

        private EndpointPreemptionAttempt(
                long incomingRequestId,
                long incomingReservationToken,
                Map<Long, ReservationHandle> victims) {
            this.incomingRequestId = incomingRequestId;
            this.incomingReservationToken = incomingReservationToken;
            this.remainingVictims = new HashMap<>(victims);
        }
    }

}
