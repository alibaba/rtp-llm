package org.flexlb.balance.endpoint;

import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.balance.scheduler.EndpointEventProjector;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.config.RoutingConfig;
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
import java.util.function.Predicate;

/**
 * Decode-side endpoint with Auto-TPM shadow admission accounting.
 *
 * <p>{@link #decodeRequests} is the canonical per-request ownership table.
 * Calibration changes a shadow reservation into an engine-confirmed entry in
 * place, preserving the exact reservation token used by fences and terminal
 * reconciliation.
 *
 * <p>The same admission lock and version counter protect every
 * reserve/release/calibrate transition, including when Auto-TPM is disabled.
 * Keeping one ownership protocol avoids mode-dependent state paths.
 */
public class DecodeEndpoint extends WorkerEndpoint {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");
    private static final Comparator<ReservationHandle> RETIREMENT_ORDER =
            Comparator.comparingLong(ReservationHandle::endpointGenerationId)
                    .thenComparingLong(ReservationHandle::requestId)
                    .thenComparingLong(ReservationHandle::reservationToken);

    private final EndpointEventProjector endpointEvents;
    private final PlacementAvailability placementAvailability;
    private final ConcurrentHashMap<Long, DecodeRequestState> decodeRequests = new ConcurrentHashMap<>();
    private final AtomicLong inflightKvReservedTotal = new AtomicLong(0);
    private final AtomicLong inflightExpectedKvReservedTotal = new AtomicLong(0);
    /**
     * Engine-confirmed request ownership: KV_ALLOCATED plus RUNNING, including
     * synthetic fenced slots. This is deliberately not the Engine's physical
     * running concurrency; confirmed entries provide the phase split.
     */
    private volatile int confirmedEngineOwnedCount;

    /** Number of reservation-phase entries in {@link #decodeRequests}. */
    private final AtomicInteger reservedRequestCount = new AtomicInteger();

    private final Map<Long, EndpointPreemptionAttempt> preemptionAttempts = new HashMap<>();

    /** KV that Decode has reported free but the Prefill CANCELED fence has not settled yet. */
    private final AtomicLong priorityPreemptionHeldKv = new AtomicLong();
    /** Expected-demand counterpart; invariant: expected hold >= hard hold >= 0. */
    private final AtomicLong priorityPreemptionHeldExpectedKv = new AtomicLong();

    /**
     * O(1) mirror of reservations whose request is still waiting in a Prefill
     * queue. Such reservations protect KV, but do not consume Decode engine
     * concurrency until delivery obtains an {@link EngineDispatchPermit}.
     * Queue publication increments the counter; dispatch, release, and status
     * reconciliation decrement it under {@link #admissionLock}.
     */
    private final AtomicInteger queuedPhaseCount = new AtomicInteger(0);

    /** Prompt-only KV held by reservations which are still Prefill-queued. */
    private final AtomicLong queuedHardKvReservedTotal = new AtomicLong(0);

    /** Expected KV held by reservations which are still Prefill-queued. */
    private final AtomicLong queuedExpectedKvReservedTotal = new AtomicLong(0);

    /**
     * Hard prompt KV already committed to acquired pre-delivery permits.
     * Permit identities carry the request generation and monotonic reservation
     * token, so a stale release cannot affect request-id reuse.
     */
    private final AtomicLong engineDispatchPermitHardKvReservedTotal = new AtomicLong();
    /** Expected KV already committed to acquired pre-delivery permits. */
    private final AtomicLong engineDispatchPermitExpectedKvReservedTotal = new AtomicLong();
    /** Mutated under admissionLock; volatile for the lock-free waiter predicate. */
    private volatile int activeEngineDispatchPermitCount;
    /** Guarded by {@link #admissionLock}; zero is never issued. */
    private long nextReservationToken = 1L;
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
    private volatile DecodeRoutingView routingViewCache;

    /**
     * Canonical generation-local ownership for one request. A request is
     * either a shadow reservation or engine-confirmed; calibration mutates the
     * same entry instead of moving ownership between parallel registries.
     */
    static final class DecodeRequestState {
        static final int DEFAULT_PRIORITY = 0;

        private long kvTokens;
        private long expectedKvTokens;
        private final long createdAtMs;
        private int priority;
        private final long reservationToken;
        private boolean priorityKnown = true;
        /** Shadow, confirmed, or null when only protocol/history remains. */
        private volatile DecodeTaskPhase phase = DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN;
        private long lastSeenAtMs;
        private boolean queued;
        /** True after dispatch crosses the Master rollback boundary. */
        private boolean engineLifecycleOwned;
        private EngineDispatchPermit dispatchPermit;
        /** Current protocol owners for this exact request generation. */
        private PreemptionClaim preemptionClaim;
        /** Non-zero while stale WorkerStatus must not resurrect this request id. */
        private long settledAtMs;

        DecodeRequestState(long kvTokens, long expectedKvTokens,
                        int priority, long reservationToken) {
            if (reservationToken < 0L) {
                throw new IllegalArgumentException(
                        "reservationToken must be non-negative");
            }
            this.kvTokens = kvTokens;
            this.expectedKvTokens = expectedKvTokens;
            this.createdAtMs = System.currentTimeMillis();
            this.priority = priority;
            this.reservationToken = reservationToken;
        }

        long kvTokens() { return kvTokens; }
        long expectedKvTokens() { return expectedKvTokens; }
        CapacityRelease capacityRelease() {
            return new CapacityRelease(1L, kvTokens, expectedKvTokens);
        }
        long createdAtMs() { return createdAtMs; }
        int priority() { return priority; }
        long reservationToken() { return reservationToken; }
        boolean queued() { return queued; }
        boolean ownsRequest() { return phase != null; }
        boolean confirmed() { return phase != null
                && phase != DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN; }
        DecodeTaskPhase phase() { return phase; }
        boolean priorityKnown() { return priorityKnown; }
        long lastSeenAtMs() { return lastSeenAtMs; }

        void confirm(
                long engineKvTokens,
                DecodeTaskPhase phase,
                long observedAtMs) {
            kvTokens = Math.max(0L, engineKvTokens);
            expectedKvTokens = kvTokens;
            this.phase = java.util.Objects.requireNonNull(phase, "phase");
            lastSeenAtMs = observedAtMs;
            queued = false;
            engineLifecycleOwned = false;
            dispatchPermit = null;
        }

        static DecodeRequestState untrackedConfirmed(
                long engineKvTokens,
                DecodeTaskPhase phase,
                long observedAtMs) {
            DecodeRequestState state = new DecodeRequestState(
                    0L, 0L, DEFAULT_PRIORITY, 0L);
            state.priorityKnown = false;
            state.confirm(engineKvTokens, phase, observedAtMs);
            return state;
        }

        void refresh(DecodeTaskPhase phase, long observedAtMs) {
            if (!confirmed()) {
                throw new IllegalStateException(
                        "cannot refresh an unconfirmed Decode request");
            }
            this.phase = java.util.Objects.requireNonNull(phase, "phase");
            lastSeenAtMs = observedAtMs;
        }

        long lastSeenMs() { return lastSeenAtMs; }

        boolean markQueued() {
            if (queued) {
                return false;
            }
            queued = true;
            return true;
        }

        boolean clearQueued() {
            if (!queued) {
                return false;
            }
            queued = false;
            return true;
        }

        EngineDispatchPermit dispatchPermit() { return dispatchPermit; }

        void installDispatchPermit(EngineDispatchPermit permit) {
            if (dispatchPermit != null) {
                throw new IllegalStateException(
                        "Decode reservation already owns a dispatch permit");
            }
            dispatchPermit = java.util.Objects.requireNonNull(permit, "permit");
        }

        EngineDispatchPermit clearDispatchPermit() {
            EngineDispatchPermit current = dispatchPermit;
            dispatchPermit = null;
            return current;
        }

        long releasableKvTokens() { return kvTokens; }

        boolean hasProtocolOwner() { return preemptionClaim != null; }

        void clearRequestOwnership() {
            phase = null;
        }
    }

    private DecodeRequestState requestState(long requestId) {
        return decodeRequests.get(requestId);
    }

    private DecodeRequestState ownedRequest(long requestId) {
        DecodeRequestState state = requestState(requestId);
        return state != null && state.ownsRequest() ? state : null;
    }

    private PreemptionClaim preemptionClaim(long requestId) {
        DecodeRequestState state = requestState(requestId);
        return state == null ? null : state.preemptionClaim;
    }

    private boolean removePreemptionClaimLocked(
            long requestId, PreemptionClaim expected) {
        DecodeRequestState state = requestState(requestId);
        if (state == null || state.preemptionClaim != expected) {
            return false;
        }
        state.preemptionClaim = null;
        pruneRequestStateLocked(requestId, state);
        return true;
    }

    /** Drop a protocol-only shell after its last exact owner is gone. */
    private void pruneRequestStateLocked(long requestId, DecodeRequestState state) {
        if (state != null && !state.ownsRequest()
                && !state.hasProtocolOwner() && state.settledAtMs == 0L) {
            decodeRequests.remove(requestId, state);
        }
    }

    private DecodeRequestState shadowReservation(long requestId) {
        DecodeRequestState state = requestState(requestId);
        return state != null
                && state.phase() == DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN
                ? state : null;
    }

    private DecodeRequestState confirmedRequest(long requestId) {
        DecodeRequestState state = requestState(requestId);
        return state != null && state.confirmed() ? state : null;
    }

    /** Caller holds {@link #admissionLock}. */
    private boolean removeShadowExactLocked(
            long requestId, DecodeRequestState expected) {
        if (expected == null || expected.confirmed()
                || requestState(requestId) != expected
                || !expected.ownsRequest()) {
            return false;
        }
        reservedRequestCount.decrementAndGet();
        clearShadowAccountingLocked(requestId, expected);
        expected.clearRequestOwnership();
        pruneRequestStateLocked(requestId, expected);
        return true;
    }

    /** Remove the counters/capabilities owned by one reservation-phase entry. */
    private void clearShadowAccountingLocked(
            long requestId, DecodeRequestState reservation) {
        removeEngineDispatchPermitLocked(reservation);
        removeEngineLifecycleReservationLocked(reservation);
        removeQueuedPhaseLocked(requestId, reservation);
        inflightKvReservedTotal.addAndGet(-reservation.kvTokens());
        inflightExpectedKvReservedTotal.addAndGet(
                -reservation.expectedKvTokens());
    }

    /** Caller holds {@link #admissionLock}. */
    private boolean removeConfirmedExactLocked(
            long requestId, DecodeRequestState expected) {
        if (expected == null || !expected.confirmed()
                || requestState(requestId) != expected) {
            return false;
        }
        expected.clearRequestOwnership();
        pruneRequestStateLocked(requestId, expected);
        return true;
    }

    public DecodeEndpoint(
            WorkerStatus status,
            EndpointEventProjector endpointEvents) {
        this(status, endpointEvents, new PlacementAvailability());
    }

    DecodeEndpoint(
            WorkerStatus status,
            EndpointEventProjector endpointEvents,
            PlacementAvailability placementAvailability) {
        super(status);
        this.endpointEvents = java.util.Objects.requireNonNull(
                endpointEvents, "endpointEvents");
        this.placementAvailability = java.util.Objects.requireNonNull(
                placementAvailability, "placementAvailability");
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
            long remainingCapacity = saturatedAddNonNegative(capacity, Math.max(0L, released - used));
            return remainingUsed > remainingCapacity
                    ? saturatedAddNonNegative(remainingUsed - remainingCapacity, incoming)
                    : Math.max(0L, incoming - (remainingCapacity - remainingUsed));
        }
    }

    /** Physical supply and local charges remain separate so existing KV debt survives projection. */
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

    /** Capacity reclaimed from the same ownership scope as the evaluated observation. */
    public record CapacityRelease(long requests, long hardKvTokens, long expectedKvTokens) {
        public static final CapacityRelease NONE = new CapacityRelease(0L, 0L, 0L);

        public CapacityRelease {
            if (requests < 0L || hardKvTokens < 0L || expectedKvTokens < hardKvTokens) {
                throw new IllegalArgumentException("invalid Decode capacity release");
            }
        }

        public CapacityRelease plus(CapacityRelease other) {
            return new CapacityRelease(saturatedAddNonNegative(requests, other.requests),
                    saturatedAddNonNegative(hardKvTokens, other.hardKvTokens),
                    saturatedAddNonNegative(expectedKvTokens, other.expectedKvTokens));
        }
    }

    public record CapacityDeficit(long requests, long hardKvTokens, long expectedKvTokens) {
        public boolean fits() { return requests == 0L && !needsKv(); }
        public boolean needsKv() { return hardKvTokens > 0L || expectedKvTokens > 0L; }
        public long kvTokens() { return Math.max(hardKvTokens, expectedKvTokens); }
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

    /** Stop this exact endpoint generation and retire all generation-local ownership. */
    @Override
    protected void closeEndpoint() {
        List<ReservationHandle> retiredReservations;
        admissionLock.lock();
        try {
            for (DecodeRequestState reservation : decodeRequests.values()) {
                EngineDispatchPermit permit = reservation.dispatchPermit();
                if (permit != null) {
                    permit.retiredByEndpoint = true;
                }
            }
            retiredReservations = retireGenerationOwnershipLocked();
            admissionVersion.incrementAndGet();
        } finally {
            admissionLock.unlock();
        }
        try {
            endpointEvents.onDecodeGenerationRetired(
                    this, retiredReservations);
        } finally {
            notifyEngineDispatchCapacityListeners();
        }
    }

    /** Materialize every exact owner, then atomically drain this generation. */
    private List<ReservationHandle> retireGenerationOwnershipLocked() {
        Set<ReservationHandle> owners = new HashSet<>();
        long generationId = getStatus().getGenerationId();

        decodeRequests.forEach((requestId, reservation) ->
                addRetiredOwner(
                        owners, generationId, requestId,
                        reservation.reservationToken()));
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
        decodeRequests.values().forEach(DecodeRequestState::clearDispatchPermit);
        activeEngineDispatchPermitCount = 0;
        engineDispatchPermitHardKvReservedTotal.set(0L);
        engineDispatchPermitExpectedKvReservedTotal.set(0L);

        decodeRequests.clear();
        reservedRequestCount.set(0);
        inflightKvReservedTotal.set(0L);
        inflightExpectedKvReservedTotal.set(0L);
        confirmedEngineOwnedCount = 0;

        queuedPhaseCount.set(0);
        queuedHardKvReservedTotal.set(0L);
        queuedExpectedKvReservedTotal.set(0L);

        preemptionAttempts.clear();
        priorityPreemptionHeldKv.set(0L);
        priorityPreemptionHeldExpectedKv.set(0L);

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

    public boolean isReservationAccepted(ReservationHandle handle) {
        if (handle == null) {
            return false;
        }
        admissionLock.lock();
        try {
            DecodeRequestState state = confirmedRequest(handle.requestId());
            return handle.endpointGenerationId() == getStatus().getGenerationId()
                    && state != null && state.reservationToken() == handle.reservationToken();
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
        DecodeRequestState newReservation =
                new DecodeRequestState(
                        kvTokens, expectedKvTokens, priority, reservationToken);
        ReservationHandle handle = new ReservationHandle(
                getStatus().getGenerationId(), requestId, reservationToken);
        if (decodeRequests.putIfAbsent(requestId, newReservation) != null) {
            throw new IllegalStateException(
                    "Decode reservation appeared while admissionLock was held: "
                            + requestId);
        }
        reservedRequestCount.incrementAndGet();
        inflightKvReservedTotal.addAndGet(kvTokens);
        inflightExpectedKvReservedTotal.addAndGet(expectedKvTokens);
        admissionVersion.incrementAndGet();
        return handle;
    }

    /**
     * Reserve placement before Engine-facing delivery, or report that this request id
     * is still fenced by the exact endpoint generation. The fence is a
     * transient placement blocker: WorkerStatus cannot distinguish a reused
     * request id until its settlement tombstone expires.
     */
    public ReservationHandle tryReservePlacementPinned(
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
            addQueuedPhaseLocked(requestId, shadowReservation(requestId));
            return reservation;
        } finally {
            admissionLock.unlock();
        }
    }

    /**
     * Reserve placement only if this exact generation can still accept
     * the request. A null result mutates nothing.
     */
    public ReservationHandle tryReservePlacementPinned(
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
            addQueuedPhaseLocked(requestId, shadowReservation(requestId));
            return reservation;
        } finally {
            admissionLock.unlock();
        }
    }

    /** Release an aborted attempt's incoming shadow. Caller holds admissionLock. */
    private boolean releaseLocked(long requestId) {
        DecodeRequestState removed = shadowReservation(requestId);
        boolean changed = removeShadowExactLocked(requestId, removed);
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

    /** Publish one real Decode capacity release to local and global waiters. */
    private void publishCapacityRelease() {
        notifyEngineDispatchCapacityListeners();
        signalPlacementCapacityChanged();
    }

    /**
     * Release one exact reservation that never crossed into Engine ownership.
     *
     * <p>A missing or replaced handle is an idempotent success. If the exact
     * identity is still present but has an Engine lifecycle or
     * priority-protocol owner, local release is a caller invariant violation:
     * that owner must consume an authoritative terminal instead.</p>
     */
    public void releaseReservationExact(ReservationHandle reservation) {
        releaseLocalReservationExact(reservation);
    }

    private boolean releaseLocalReservationExact(
            ReservationHandle reservation) {
        if (reservation == null) {
            throw new IllegalArgumentException(
                    "Decode reservation is required for release");
        }
        if (reservation.endpointGenerationId()
                != getStatus().getGenerationId()) {
            return false;
        }
        boolean capacityChanged = false;
        admissionLock.lock();
        try {
            long requestId = reservation.requestId();
            DecodeRequestState state = requestState(requestId);
            DecodeRequestState current = state != null && state.ownsRequest()
                    ? state : null;
            boolean exactRequest = isExactReservation(current, reservation);
            boolean exactState = isExactReservation(state, reservation);
            PreemptionClaim claim = state == null
                    ? null : state.preemptionClaim;
            EngineDispatchPermit dispatchPermit = current == null
                    || current.confirmed()
                    ? null : current.dispatchPermit();

            boolean exactClaim = exactState && claim != null;
            boolean exactAttempt = hasExactIncomingAttemptLocked(reservation);
            boolean exactDispatchPermit = dispatchPermit != null
                    && isExactReservation(
                            dispatchPermit.reservation, reservation);
            boolean exactEngineLifecycle =
                    hasEngineLifecycleReservationExactLocked(reservation);

            if (!exactRequest) {
                if (exactClaim
                        || exactAttempt || exactDispatchPermit
                        || exactEngineLifecycle) {
                    throw localReleaseInvariant(reservation,
                            "exact ownership already crossed local shadow release");
                }
                return false;
            }
            if (current.confirmed() || exactEngineLifecycle
                    || exactClaim || exactAttempt) {
                throw localReleaseInvariant(reservation,
                        "exact ownership is held by Engine/protocol lifecycle");
            }

            if (!exactDispatchPermit && dispatchPermit != null) {
                throw localReleaseInvariant(reservation,
                        "request id has a dispatch permit for another reservation");
            }
            if (!removeShadowExactLocked(requestId, current)) {
                throw localReleaseInvariant(reservation,
                        "exact shadow changed while admissionLock was held");
            }
            admissionVersion.incrementAndGet();
            capacityChanged = true;
        } finally {
            admissionLock.unlock();
        }
        if (capacityChanged) {
            publishCapacityRelease();
        }
        return capacityChanged;
    }

    /**
     * Expire one exact local request lease without asserting an Engine terminal.
     * The last physical KV sample remains unchanged; only local reservations,
     * dispatch permits, and priority protocol ownership are released. A bounded
     * tombstone prevents delayed request-id-only WorkerStatus from recreating the
     * expired generation. Duplicate and stale handles are total no-ops.
     */
    public boolean expireReservationExact(ReservationHandle reservation) {
        if (reservation == null
                || reservation.endpointGenerationId()
                        != getStatus().getGenerationId()) {
            return false;
        }
        boolean expired = false;
        admissionLock.lock();
        try {
            long requestId = reservation.requestId();
            DecodeRequestState state = requestState(requestId);
            if (!isExactReservation(state, reservation)
                    || (!state.ownsRequest() && !state.hasProtocolOwner()
                        && !hasExactIncomingAttemptLocked(reservation))) {
                return false;
            }

            // An expired incoming request cannot retain an admission attempt.
            // Other victims keep ambiguous Engine ownership until their own
            // status or inactivity deadline settles it.
            java.util.Iterator<Map.Entry<Long, EndpointPreemptionAttempt>> attempts =
                    preemptionAttempts.entrySet().iterator();
            while (attempts.hasNext()) {
                Map.Entry<Long, EndpointPreemptionAttempt> entry = attempts.next();
                EndpointPreemptionAttempt attempt = entry.getValue();
                if (attempt.incomingRequestId != requestId
                        || attempt.incomingReservationToken
                                != reservation.reservationToken()) {
                    continue;
                }
                attempts.remove();
                for (ReservationHandle victim : attempt.remainingVictims.values()) {
                    PreemptionClaim victimClaim = exactPreemptionClaimLocked(
                            entry.getKey(), victim);
                    if (victimClaim != null && victimClaim.phase.isLocallyReleasable()) {
                        releaseHeldKv(victimClaim);
                        removePreemptionClaimLocked(victim.requestId(), victimClaim);
                    }
                }
            }

            PreemptionClaim claim = state.preemptionClaim;
            boolean confirmedSlot = state.confirmed()
                    || claim != null && claim.owner == ClaimOwner.ENGINE_CONFIRMED;
            if (claim != null) {
                EndpointPreemptionAttempt attempt = preemptionAttempts.get(claim.attemptToken);
                if (attempt != null) {
                    attempt.remainingVictims.remove(requestId, reservation);
                }
                releaseHeldKv(claim);
                removePreemptionClaimLocked(requestId, claim);
            }
            if (state.confirmed()) {
                removeConfirmedExactLocked(requestId, state);
            } else if (state.ownsRequest()) {
                removeShadowExactLocked(requestId, state);
            }
            if (confirmedSlot) {
                confirmedEngineOwnedCount = Math.max(0, confirmedEngineOwnedCount - 1);
            }
            // Use a history-only entry, so no old exact token remains live.
            decodeRequests.remove(requestId, state);
            rememberSettledLocked(requestId, System.currentTimeMillis());
            admissionVersion.incrementAndGet();
            expired = true;
        } finally {
            admissionLock.unlock();
        }
        if (expired) {
            publishCapacityRelease();
        }
        return expired;
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
            DecodeRequestState shadow = shadowReservation(requestId);
            if (!isExactReservation(shadow, reservation)
                    || hasEngineLifecycleReservationExactLocked(reservation)) {
                return false;
            }
            if (shadow.preemptionClaim != null
                    || hasExactIncomingAttemptLocked(reservation)) {
                return false;
            }
            EngineDispatchPermit dispatchPermit = shadow.dispatchPermit();
            if (dispatchPermit != null
                    && !isExactReservation(
                            dispatchPermit.reservation, reservation)) {
                return false;
            }
            if (!removeShadowExactLocked(requestId, shadow)) {
                return false;
            }
            if (!decodeRequests.containsKey(requestId)) {
                rememberSettledLocked(requestId, System.currentTimeMillis());
            }
            admissionVersion.incrementAndGet();
            released = true;
        } finally {
            admissionLock.unlock();
        }
        if (released) {
            publishCapacityRelease();
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
            DecodeRequestState current,
            ReservationHandle reservation) {
        return current != null
                && current.reservationToken()
                        == reservation.reservationToken();
    }

    /** Caller holds {@link #admissionLock}. */
    private boolean hasEngineLifecycleReservationExactLocked(
            ReservationHandle reservation) {
        DecodeRequestState current = shadowReservation(
                reservation.requestId());
        return isExactReservation(current, reservation)
                && current.engineLifecycleOwned;
    }

    private static IllegalStateException localReleaseInvariant(
            ReservationHandle reservation,
            String detail) {
        return new IllegalStateException(
                "Illegal Decode local release: requestId="
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
            DecodeRequestState current = shadowReservation(requestId);
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
            DecodeRequestState state = requestState(requestId);
            boolean exactState = isExactReservation(state, reservation);
            DecodeRequestState confirmed = confirmedRequest(requestId);
            if (confirmed != null
                    && exactState) {
                return DispatchRejectionSettlement.ENGINE_ACCEPTED;
            }

            PreemptionClaim claim = state == null
                    ? null : state.preemptionClaim;
            if (claim != null && exactState) {
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

                DecodeRequestState shadow = shadowReservation(requestId);
                EngineDispatchPermit dispatchPermit = shadow == null
                        ? null : shadow.dispatchPermit();
                boolean exactShadow = isExactReservation(shadow, reservation);
                boolean exactDispatchPermit = dispatchPermit != null
                        && isExactReservation(
                                dispatchPermit.reservation, reservation);
                boolean exactEngineLifecycle =
                        hasEngineLifecycleReservationExactLocked(reservation);
                if (!exactShadow && !exactDispatchPermit
                        && !exactEngineLifecycle) {
                    return DispatchRejectionSettlement.STALE;
                }

                capacityChanged = settleAuthoritativeTerminalLocked(
                        reservation, true, System.currentTimeMillis());
                if (capacityChanged) {
                    admissionVersion.incrementAndGet();
                }
                result = DispatchRejectionSettlement.RELEASED;
            }
        } finally {
            admissionLock.unlock();
        }
        if (capacityChanged) {
            publishCapacityRelease();
        }
        return result;
    }

    /** Caller holds {@link #admissionLock}. */
    private boolean settleAuthoritativeTerminalLocked(
            ReservationHandle reservation,
            boolean retainTombstone,
            long settledAtMs) {
        long requestId = reservation.requestId();
        DecodeRequestState state = requestState(requestId);
        boolean exactState = isExactReservation(state, reservation);

        PreemptionClaim claim = state == null ? null : state.preemptionClaim;
        if (claim != null && exactState) {
            throw terminalInvariant(reservation,
                    "priority claim must settle before generic terminal ownership");
        }
        if (hasExactIncomingAttemptLocked(reservation)) {
            throw terminalInvariant(reservation,
                    "priority attempt still owns the exact incoming reservation");
        }

        boolean changed = false;
        DecodeRequestState request = state != null && state.ownsRequest()
                ? state : null;
        EngineDispatchPermit dispatchPermit = request == null
                || request.confirmed() ? null : request.dispatchPermit();
        if (dispatchPermit != null
                && isExactReservation(
                        dispatchPermit.reservation, reservation)) {
            changed = removeEngineDispatchPermitLocked(requestId) || changed;
        }

        if (isExactReservation(request, reservation)) {
            if (request.confirmed()
                    && removeConfirmedExactLocked(requestId, request)) {
                confirmedEngineOwnedCount = Math.max(
                        0, confirmedEngineOwnedCount - 1);
                changed = true;
            } else if (!request.confirmed()
                    && removeShadowExactLocked(requestId, request)) {
                changed = true;
            }
        }
        if (!decodeRequests.containsKey(requestId) && retainTombstone) {
            changed = rememberSettledLocked(requestId, settledAtMs) || changed;
        }

        if (hasExactOwnerLocked(reservation)) {
            throw terminalInvariant(reservation,
                    "exact accounting remains after authoritative settlement");
        }
        return changed;
    }

    /**
     * Remove only one Engine task that never had a local reservation identity.
     * A same-id shadow/fence necessarily belongs to another generation and is
     * deliberately preserved.
     */
    private boolean settleUntrackedWorkerTerminalLocked(long requestId) {
        DecodeRequestState confirmed = confirmedRequest(requestId);
        if (confirmed == null || confirmed.reservationToken() > 0L
                || !removeConfirmedExactLocked(requestId, confirmed)) {
            return false;
        }
        confirmedEngineOwnedCount = Math.max(0, confirmedEngineOwnedCount - 1);
        return true;
    }

    /** Caller holds {@link #admissionLock}. */
    private boolean removeEngineLifecycleReservationLocked(
            DecodeRequestState reservation) {
        if (reservation == null || !reservation.engineLifecycleOwned) {
            return false;
        }
        reservation.engineLifecycleOwned = false;
        return true;
    }

    /**
     * WorkerStatus identifies work only by request id. Reuse is therefore
     * forbidden while this endpoint generation still owns that id or retains
     * an ambiguity tombstone for it.
     */
    private boolean requestIdAvailableForReservationLocked(long requestId) {
        if (decodeRequests.containsKey(requestId)) {
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
        DecodeRequestState state = requestState(requestId);
        return isExactReservation(state, reservation)
                || hasExactIncomingAttemptLocked(reservation);
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

    /**
     * Atomically replace a fully validated set of exact Master-queued
     * reservations with one incoming reservation. Any stale victim or
     * incoming-id conflict leaves every owner untouched.
     */
    public boolean tryEvictLocalReservationsAndReserveIncoming(
            List<ReservationHandle> victims,
            long incomingRequestId, long kvTokens, long expectedKvTokens,
            int priority,
            AdmissionCapacity capacity) {
        GenerationPin generationPin = tryPinGeneration();
        if (generationPin == null) {
            return false;
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

    private boolean
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
                return false;
            }
            Set<Long> uniqueVictims = new HashSet<>(victims.size());
            CapacityRelease released = CapacityRelease.NONE;
            for (ReservationHandle victim : victims) {
                if (victim == null
                        || victim.endpointGenerationId()
                                != getStatus().getGenerationId()
                        || victim.requestId() == incomingRequestId
                        || !uniqueVictims.add(victim.requestId())) {
                    return false;
                }
                DecodeRequestState held = shadowReservation(victim.requestId());
                EngineDispatchPermit permit = held == null
                        ? null : held.dispatchPermit();
                if (!isExactReservation(held, victim)
                        || !held.queued()
                        || hasEngineLifecycleReservationExactLocked(victim)
                        || held.hasProtocolOwner()
                        || hasExactIncomingAttemptLocked(victim)
                        || permit != null) {
                    return false;
                }
                released = released.plus(held.capacityRelease());
            }
            if (projectedEvictionCapacityFitsLocked(
                    capacity, kvTokens, expectedKvTokens, CapacityRelease.NONE)) {
                return false;
            }
            if (!projectedEvictionCapacityFitsLocked(
                    capacity, kvTokens, expectedKvTokens, released)) {
                return false;
            }

            for (ReservationHandle victim : victims) {
                DecodeRequestState exact = shadowReservation(victim.requestId());
                if (!removeShadowExactLocked(victim.requestId(), exact)) {
                    throw localReleaseInvariant(
                            victim,
                            "validated victim changed while admissionLock was held");
                }
            }
            reserveLocked(
                    incomingRequestId,
                    kvTokens,
                    expectedKvTokens,
                    priority);
            committed = true;
            return true;
        } finally {
            admissionLock.unlock();
            if (committed) {
                publishCapacityRelease();
            }
        }
    }

    /** Consistent reserved/confirmed ownership view captured under admissionLock. */
    public LayeredAdmissionView layeredAdmissionView() {
        admissionLock.lock();
        try {
            Map<Long, DecodeRequestView> reserved = new HashMap<>(
                    reservedRequestCount.get());
            List<DecodeRequestView> confirmed = new java.util.ArrayList<>(
                    Math.max(0, confirmedEngineOwnedCount));
            decodeRequests.forEach((requestId, task) -> {
                if (!task.ownsRequest()) {
                    return;
                }
                boolean protectedRequest = task.hasProtocolOwner();
                if (task.confirmed()) {
                    confirmed.add(new DecodeRequestView(
                            requestId, task.priority(), task.kvTokens(),
                            task.kvTokens(), task.phase(), task.priorityKnown(),
                            task.reservationToken(), false, protectedRequest));
                } else {
                    reserved.put(requestId, new DecodeRequestView(
                            requestId, task.priority(), task.releasableKvTokens(),
                            task.expectedKvTokens(),
                            task.queued()
                                    ? DecodeTaskPhase.MASTER_QUEUED_NOT_DISPATCHED
                                    : DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN,
                            true,
                            task.reservationToken(), task.queued(),
                            protectedRequest));
                }
            });
            return new LayeredAdmissionView(routingViewLocked(),
                    Map.copyOf(reserved), List.copyOf(confirmed),
                    queuedPhaseCount.get(), activeEngineDispatchPermitCount);
        } finally {
            admissionLock.unlock();
        }
    }

    /** Advisory endpoint admission revision captured by queue placement. */
    public long placementVersion() {
        return admissionVersion.get();
    }

    /** Atomic reserved/confirmed/queued ownership tuple. */
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

    /**
     * One on-demand routing projection captured under the canonical admission
     * lock. It is not a second owner: every value is derived from the live
     * registries and the one committed WorkerStatus holder at capture time.
     */
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
     * generations are unchanged. {@code admissionVersion} is the coherence
     * token for both committed Engine observations and locally owned
     * reservations, so the cached fast path need not reread and compare the
     * larger status snapshots for every candidate. This is deliberately
     * separate from
     * {@link #routingView()}: winner authorization must always take the fresh
     * locked path above.
     */
    DecodeRoutingView routingViewSnapshot(String address) {
        long version = admissionVersion.get();
        WorkerStatus status = getStatus();
        WorkerStatus.TopologySnapshot topology = status.topologySnapshot();
        DecodeRoutingView cached = routingViewCache;
        if (routingViewMatches(cached, address, version, topology)) {
            return cached;
        }
        admissionLock.lock();
        try {
            WorkerStatus.CommittedWorkerStatus committed =
                    status.committedWorkerStatus();
            topology = status.topologySnapshot();
            version = admissionVersion.get();
            cached = routingViewCache;
            if (routingViewMatches(cached, address, version, topology)) {
                return cached;
            }
            DecodeRoutingView routing = routingViewLocked(
                    address, topology, committed, version);
            routingViewCache = routing;
            return routing;
        } finally {
            admissionLock.unlock();
        }
    }

    private static boolean routingViewMatches(
            DecodeRoutingView cached,
            String address,
            long version,
            WorkerStatus.TopologySnapshot topology) {
        return cached != null
                && cached.address().equals(address)
                && cached.admissionVersion() == version
                && cached.topology() == topology;
    }

    /** Caller holds {@link #admissionLock}. */
    private DecodeRoutingView routingViewLocked() {
        WorkerStatus status = getStatus();
        return routingViewLocked(
                ipPort(),
                status.topologySnapshot(),
                status.committedWorkerStatus(),
                admissionVersion.get());
    }

    /** Caller holds {@link #admissionLock}; both key values were read there. */
    private DecodeRoutingView routingViewLocked(
            String address,
            WorkerStatus.TopologySnapshot topology,
            WorkerStatus.CommittedWorkerStatus committed,
            long version) {
        WorkerStatus.EngineObservation fields = committed.fields();
        int inflight = reservedRequestCount.get();
        int queued = Math.max(0, Math.min(queuedPhaseCount.get(), inflight));
        int totalLoad = confirmedEngineOwnedCount + inflight;
        int engineLoad = confirmedEngineOwnedCount + Math.max(0, inflight - queued);
        long reportedUsed = fields.totalKvCacheTokens() > 0
                ? Math.max(0L, fields.totalKvCacheTokens()
                        - fields.availableKvCacheTokens())
                : 0L;
        long hardInflight = inflightKvReservedTotal.get();
        long expectedInflight = inflightExpectedKvReservedTotal.get();
        long used = saturatedAddNonNegative(
                saturatedAddNonNegative(reportedUsed, expectedInflight),
                priorityPreemptionHeldExpectedKv.get());
        long heldHard = priorityPreemptionHeldKv.get();
        long placementHard = saturatedAddNonNegative(hardInflight, heldHard);
        CapacityUsage placementUsage = new CapacityUsage(totalLoad,
                Math.max(0L, fields.totalKvCacheTokens()), Math.max(0L, fields.availableKvCacheTokens()),
                placementHard, used);
        CapacityUsage dispatchUsage = dispatchCapacityUsage(fields);
        return new DecodeRoutingView(
                address,
                getStatus().getGenerationId(),
                topology,
                committed,
                version,
                totalLoad,
                engineLoad,
                placementUsage,
                dispatchUsage,
                hardInflight,
                expectedInflight);
    }

    /** Immutable point-in-time view shared by shadow and confirmed ownership. */
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
            CapacityRelease released = CapacityRelease.NONE;
            for (ReservationHandle victim : victims) {
                if (victim == null
                        || victim.endpointGenerationId()
                                != getStatus().getGenerationId()
                        || victim.requestId() == incomingRequestId
                        || owners.containsKey(victim.requestId())) {
                    return PreemptionBeginResult.VICTIM_GONE;
                }
                long victimId = victim.requestId();
                DecodeRequestState victimState = requestState(victimId);
                if (victimState != null && victimState.hasProtocolOwner()
                        || hasExactIncomingAttemptLocked(victim)) {
                    return PreemptionBeginResult.VICTIM_ALREADY_CLAIMED;
                }
                DecodeRequestState request = ownedRequest(victimId);
                if (!isExactReservation(request, victim)) {
                    return PreemptionBeginResult.VICTIM_GONE;
                }
                EngineDispatchPermit dispatchPermit = request.confirmed()
                        ? null : request.dispatchPermit();
                if (dispatchPermit != null) {
                    return PreemptionBeginResult.VICTIM_GONE;
                }
                if (!request.confirmed() && !request.queued()) {
                    if (request.priority() <= 0
                            || request.priority() >= incomingPriority) {
                        return PreemptionBeginResult.INVALID_PRIORITY;
                    }
                    owners.put(victimId, ClaimOwner.SHADOW_IN_FLIGHT);
                    exactVictims.put(victimId, victim);

                } else if (request.confirmed()
                        && request.phase().isEngineConfirmed()) {
                    if (request.priority() <= 0
                            || request.priority() >= incomingPriority) {
                        return PreemptionBeginResult.INVALID_PRIORITY;
                    }
                    owners.put(victimId, ClaimOwner.ENGINE_CONFIRMED);
                    exactVictims.put(victimId, victim);

                } else {
                    return PreemptionBeginResult.VICTIM_GONE;
                }
                released = released.plus(request.capacityRelease());
            }
            if (projectedEvictionCapacityFitsLocked(
                    capacity, incomingKvTokens, incomingExpectedKvTokens,
                    CapacityRelease.NONE)) {
                return PreemptionBeginResult.INFEASIBLE;
            }
            if (!projectedEvictionCapacityFitsLocked(
                    capacity, incomingKvTokens, incomingExpectedKvTokens, released)) {
                return PreemptionBeginResult.INFEASIBLE;
            }

            // Allocate every victim claim before installing any incoming or
            // protocol ownership. The endpoint lock keeps these exact states
            // stable through the subsequent allocation-free installation.
            Map<Long, PreemptionClaim> preparedClaims = new HashMap<>();
            for (ReservationHandle victim : victims) {
                long victimId = victim.requestId();
                DecodeRequestState request = ownedRequest(victimId);
                long hardKv = request.kvTokens();
                long expectedKv = request.confirmed()
                        ? hardKv : request.expectedKvTokens();
                preparedClaims.put(
                        victimId,
                        new PreemptionClaim(
                                attemptToken,
                                owners.get(victimId),
                                hardKv,
                                expectedKv));
            }

            // Provisional incoming ownership closes the free-pool race while
            // Cancel runs.  It is not visible to the prefill queue yet.
            ReservationHandle incomingReservation = reserveLocked(
                    incomingRequestId, incomingKvTokens,
                    incomingExpectedKvTokens, incomingPriority);
            EndpointPreemptionAttempt preparedAttempt = null;
            try {
                preparedAttempt = new EndpointPreemptionAttempt(
                        incomingRequestId,
                        incomingReservation.reservationToken(),
                        exactVictims);
                EndpointPreemptionAttempt previous = preemptionAttempts.put(
                        attemptToken, preparedAttempt);
                if (previous != null) {
                    throw new IllegalStateException(
                            "priority attempt appeared while admissionLock was held");
                }
                for (Map.Entry<Long, PreemptionClaim> claim
                        : preparedClaims.entrySet()) {
                    DecodeRequestState request = ownedRequest(claim.getKey());
                    if (request == null || request.preemptionClaim != null) {
                        throw new IllegalStateException(
                                "validated priority victim changed before claim installation");
                    }
                    request.preemptionClaim = claim.getValue();
                }
            } catch (RuntimeException | Error installationFailure) {
                if (preparedAttempt != null) {
                    preemptionAttempts.remove(attemptToken, preparedAttempt);
                }
                for (Map.Entry<Long, PreemptionClaim> claim
                        : preparedClaims.entrySet()) {
                    removePreemptionClaimLocked(claim.getKey(), claim.getValue());
                }
                DecodeRequestState incoming =
                        shadowReservation(incomingRequestId);
                if (isExactReservation(incoming, incomingReservation)) {
                    removeShadowExactLocked(incomingRequestId, incoming);
                }
                throw installationFailure;
            }
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
                PreemptionClaim claim = exactPreemptionClaimLocked(
                        attemptToken, victim);
                if (claim == null
                        || !claim.phase.canTransitionTo(
                                PreemptionCancelPhase.CANCEL_IN_FLIGHT)) {
                    return false;
                }
            }
            for (ReservationHandle victim
                    : attempt.remainingVictims.values()) {
                preemptionClaim(victim.requestId()).phase =
                        PreemptionCancelPhase.CANCEL_IN_FLIGHT;
            }
            admissionVersion.incrementAndGet();
            return true;
        } finally {
            admissionLock.unlock();
        }
    }

    /**
     * Record the exact response to an in-flight priority Cancel.
     *
     * <p>Only response phases are accepted here. ACCEPTED and NOT_FOUND must
     * observe {@link PreemptionCancelPhase#CANCEL_IN_FLIGHT}; an unknown
     * transport result may also follow an ACCEPTED response whose subsequent
     * control transition could not be completed. No ownership is released.</p>
     */
    public boolean recordPriorityCancelPhase(
            long attemptToken,
            long requestId,
            PreemptionCancelPhase next) {
        if (next != PreemptionCancelPhase.CANCEL_REQUESTED
                && next != PreemptionCancelPhase.NOT_FOUND_STALE
                && next != PreemptionCancelPhase.CANCEL_UNKNOWN) {
            throw new IllegalArgumentException(
                    "Unsupported priority Cancel response phase: " + next);
        }
        admissionLock.lock();
        try {
            PreemptionClaim claim = preemptionClaim(requestId);
            if (claim == null || claim.attemptToken != attemptToken
                    || !claim.phase.canTransitionTo(next)) {
                return false;
            }
            claim.phase = next;
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
        return settlePriorityClaimTerminal(
                attemptToken,
                reservation,
                claim -> claim.phase.acceptsPriorityTerminal());
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
        return settlePriorityClaimTerminal(
                attemptToken,
                reservation,
                claim -> claim.phase.acceptsTombstone());
    }

    private boolean settlePriorityClaimTerminal(
            long attemptToken,
            ReservationHandle reservation,
            Predicate<PreemptionClaim> acceptsProof) {
        if (reservation == null
                || reservation.endpointGenerationId()
                        != getStatus().getGenerationId()) {
            return false;
        }
        boolean settled = false;
        boolean capacityChanged = false;
        admissionLock.lock();
        try {
            PreemptionClaim claim = exactPreemptionClaimLocked(
                    attemptToken, reservation);
            if (claim == null || !acceptsProof.test(claim)) {
                return false;
            }
            settled = settlePriorityClaimTerminalLocked(
                    attemptToken, reservation, claim);
            capacityChanged = settled;
        } finally {
            admissionLock.unlock();
        }
        if (capacityChanged) {
            publishCapacityRelease();
        }
        return settled;
    }

    /** Called with {@link #admissionLock} held. */
    private boolean settlePriorityClaimTerminalLocked(
            long attemptToken,
            ReservationHandle reservation,
            PreemptionClaim claim) {
        long requestId = reservation.requestId();
        DecodeRequestState state = requestState(requestId);
        if (exactPreemptionClaimLocked(attemptToken, reservation) != claim) {
            return false;
        }

        DecodeRequestState request = state.ownsRequest() ? state : null;
        EngineDispatchPermit dispatchPermit = request == null
                || request.confirmed() ? null : request.dispatchPermit();
        if ((request != null && !isExactReservation(request, reservation))
                || (dispatchPermit != null
                    && !isExactReservation(
                            dispatchPermit.reservation, reservation))
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

        if (dispatchPermit != null) {
            removeEngineDispatchPermitLocked(requestId);
        }
        if (request != null && request.confirmed()) {
            removeConfirmedExactLocked(requestId, request);
        }
        if (claim.owner == ClaimOwner.ENGINE_CONFIRMED) {
            confirmedEngineOwnedCount = Math.max(0, confirmedEngineOwnedCount - 1);
        }
        if (request != null && !request.confirmed()) {
            removeShadowExactLocked(requestId, request);
        }
        releaseHeldKv(claim);
        removePreemptionClaimLocked(requestId, claim);
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
            DecodeRequestState incoming = attempt == null
                    ? null : shadowReservation(attempt.incomingRequestId);
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
                PreemptionClaim claim = exactPreemptionClaimLocked(
                        attemptToken, victim);
                if (claim == null) {
                    continue;
                }
                if (claim.phase.isLocallyReleasable()) {
                    releaseHeldKv(claim);
                    removePreemptionClaimLocked(victimId, claim);
                }
            }
            admissionVersion.incrementAndGet();
            capacityChanged = true;
        } finally {
            admissionLock.unlock();
        }
        if (capacityChanged) {
            publishCapacityRelease();
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
            PreemptionClaim claim = exactPreemptionClaimLocked(
                    attemptToken, reservation);
            if (claim == null
                    || claim.phase != PreemptionCancelPhase.NOT_FOUND_STALE) {
                return false;
            }
            releaseHeldKv(claim);
            removePreemptionClaimLocked(requestId, claim);
            admissionVersion.incrementAndGet();
            reconciled = true;
        } finally {
            admissionLock.unlock();
        }
        if (reconciled) {
            publishCapacityRelease();
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
            PreemptionClaim claim = exactPreemptionClaimLocked(
                    attemptToken, reservation);
            if (claim == null
                    || !claim.phase.requiresOrdinaryReconciliation()) {
                return false;
            }
            // An ordinary finished sample settles the exact priority owner.
            releaseHeldKv(claim);
            removePreemptionClaimLocked(requestId, claim);
            if (reservation.reservationToken() > 0L) {
                settleAuthoritativeTerminalLocked(
                        reservation, false, System.currentTimeMillis());
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
            publishCapacityRelease();
        }
        return reconciled;
    }

    private PreemptionClaim exactPreemptionClaimLocked(
            long attemptToken, ReservationHandle reservation) {
        DecodeRequestState state = requestState(reservation.requestId());
        PreemptionClaim claim = state == null ? null : state.preemptionClaim;
        return claim != null
                && claim.attemptToken == attemptToken
                && isExactReservation(state, reservation) ? claim : null;
    }

    @Override
    public Runnable applyPreparedStatus(
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
        return () -> endpointEvents.onDecodeStatus(this, facts);
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
                RoleType.DECODE, topology.group(), ipPort());
    }

    @Override
    public Runnable initializeFromPreparedStatus(
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
        return () -> { };
    }

    @Override
    public Runnable observeStatusHeartbeat(
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
        return () -> endpointEvents.onDecodeStatus(this, facts);
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
            DecodeRequestState current = requestState(requestId);
            if (terminalNow.contains(requestId)
                    || current != null && current.settledAtMs != 0L) {
                continue;
            }
            if (phase == TaskPhase.KV_ALLOCATED || phase == TaskPhase.RUNNING) {
                actualConfirmed++;
                DecodeRequestState removed = shadowReservation(requestId);
                if (removed != null) {
                    clearShadowAccountingLocked(requestId, removed);
                    reservedRequestCount.decrementAndGet();
                }
                PreemptionClaim claim = preemptionClaim(requestId);
                if (claim != null
                        && claim.phase == PreemptionCancelPhase.NOT_FOUND_STALE
                        && !preemptionAttempts.containsKey(claim.attemptToken)) {
                    releaseHeldKv(claim);
                    removePreemptionClaimLocked(requestId, claim);
                    claim = null;
                }
                if (claim != null) {
                    claim.owner = ClaimOwner.ENGINE_CONFIRMED;
                    releaseHeldKv(claim);
                }
                confirmedNow.add(requestId);
                trackConfirmed(task, phase, now);
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
        // the exact DecodeRequestState identity. Endpoint settlement happens here;
        // the downstream scheduler receives only the immutable result.
        for (WorkerStatus.TaskObservation task
                : finishedTasks.values()) {
            long requestId = task.requestId();
            DecodeRequestState current = requestState(requestId);
            if (current != null && current.settledAtMs != 0L) {
                continue;
            }
            ReservationHandle terminal = workerStatusHandleLocked(requestId);
            confirmedNow.remove(requestId);
            PreemptionClaim claim = preemptionClaim(requestId);
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
            if (terminal != null) {
                settleAuthoritativeTerminalLocked(terminal, false, now);
            } else {
                settleUntrackedWorkerTerminalLocked(requestId);
            }
        }

        int syntheticallyHeldSlots = 0;
        for (Map.Entry<Long, DecodeRequestState> entry : decodeRequests.entrySet()) {
            PreemptionClaim claim = entry.getValue().preemptionClaim;
            if (claim == null) {
                continue;
            }
            if (claim.owner == ClaimOwner.ENGINE_CONFIRMED
                    && !confirmedNow.contains(entry.getKey())) {
                syntheticallyHeldSlots++;
                holdReleasedKv(claim);
            }
        }

        // Priority claims retain confirmed accounting until their exact
        // settlement or local request expiration. Ordinary absent tasks are pruned.
        java.util.Iterator<Map.Entry<Long, DecodeRequestState>> confirmedIt =
                decodeRequests.entrySet().iterator();
        while (confirmedIt.hasNext()) {
            Map.Entry<Long, DecodeRequestState> entry = confirmedIt.next();
            if (!entry.getValue().ownsRequest()
                    || !entry.getValue().confirmed()) {
                continue;
            }
            long requestId = entry.getKey();
            if (confirmedNow.contains(requestId)) {
                continue;
            }
            if (entry.getValue().preemptionClaim != null) {
                continue;
            }
            confirmedIt.remove();
            if (!decodeRequests.containsKey(requestId)) {
                rememberSettledLocked(requestId, now);
            }
        }
        this.confirmedEngineOwnedCount = actualConfirmed + syntheticallyHeldSlots;

        return facts;
    }

    /** Caller holds {@link #admissionLock}; request-id-only facts are forbidden. */
    private ReservationHandle workerStatusHandleLocked(long requestId) {
        DecodeRequestState state = requestState(requestId);
        long reservationToken = state == null ? 0L : state.reservationToken();
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
            long now) {
        DecodeTaskPhase layer = phase == TaskPhase.KV_ALLOCATED
                ? DecodeTaskPhase.ACCEPTED_NOT_RUNNING
                : DecodeTaskPhase.RUNNING;
        DecodeRequestState tracked = requestState(task.requestId());
        if (tracked == null) {
            decodeRequests.put(task.requestId(),
                    DecodeRequestState.untrackedConfirmed(
                            task.inputLength(), layer, now));
        } else if (!tracked.ownsRequest()) {
            tracked.confirm(task.inputLength(), layer, now);
        } else if (tracked.confirmed()) {
            tracked.refresh(layer, now);
        } else {
            tracked.confirm(task.inputLength(), layer, now);
        }
    }

    /** Publish one non-refreshing terminal fence while admissionLock is held. */
    private boolean rememberSettledLocked(long requestId, long settledAtMs) {
        DecodeRequestState state = requestState(requestId);
        if (state == null) {
            state = new DecodeRequestState(
                    0L, 0L, DecodeRequestState.DEFAULT_PRIORITY, 0L);
            state.clearRequestOwnership();
            decodeRequests.put(requestId, state);
        }
        if (state.settledAtMs != 0L) {
            return false;
        }
        state.settledAtMs = settledAtMs;
        return true;
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
    private long inflightHardKvReserved() {
        return inflightKvReservedTotal.get();
    }

    /**
     * KV demand which may reach the engine now. Reservations parked in a
     * Prefill queue are soft placement hints: charging all of them against the
     * hard availability gate makes a long scheduler queue report every Decode
     * worker unavailable even though none of that work has been dispatched.
     */
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
                priorityPreemptionHeldExpectedKv.get());
    }

    /**
     * Real KV available: engine-reported available minus local shadow,
     * and priority preemption hard reservations.
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
                - priorityPreemptionHeldKv.get());
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
                inflightMaxAgeMs(System.currentTimeMillis()));
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
        return reservedRequestCount.get();
    }

    /** Evict only endpoint orphans which have no live scheduler generation. */
    public int evictExpiredRequests(long ttlMs,
                                    LongPredicate schedulerOwnsRequest) {
        int evicted;
        boolean capacityChanged;
        admissionLock.lock();
        try {
            // Scheduler-owned requests expire through their exact local lease.
            // This pass only sweeps endpoint orphans.
            evicted = evictExpiredInflightLocked(
                    ttlMs, schedulerOwnsRequest);
            long cutoff = System.currentTimeMillis() - ttlMs;
            int trackedPurged = 0;
            java.util.Iterator<Map.Entry<Long, DecodeRequestState>> trackedEvictIt =
                    decodeRequests.entrySet().iterator();
            while (trackedEvictIt.hasNext()) {
                Map.Entry<Long, DecodeRequestState> entry = trackedEvictIt.next();
                if (entry.getValue().confirmed()
                        && entry.getValue().lastSeenMs() < cutoff
                        && !schedulerOwnsRequest.test(entry.getKey())
                        && entry.getValue().preemptionClaim == null) {
                    trackedEvictIt.remove();
                    trackedPurged++;
                }
            }
            if (trackedPurged > 0) {
                confirmedEngineOwnedCount = Math.max(
                        0, confirmedEngineOwnedCount - trackedPurged);
            }
            boolean settledTombstonesPurged = decodeRequests.entrySet()
                    .removeIf(entry -> !entry.getValue().ownsRequest()
                            && !entry.getValue().hasProtocolOwner()
                            && entry.getValue().settledAtMs != 0L
                            && entry.getValue().settledAtMs < cutoff);
            if (evicted > 0 || trackedPurged > 0 || settledTombstonesPurged) {
                admissionVersion.incrementAndGet();
            }
            capacityChanged = evicted > 0 || trackedPurged > 0
                    || settledTombstonesPurged;
        } finally {
            admissionLock.unlock();
        }
        if (capacityChanged) {
            publishCapacityRelease();
        }
        return evicted;
    }

    /** Caller holds {@link #admissionLock}. */
    private int evictExpiredInflightLocked(
            long ttlMs, LongPredicate schedulerOwnsRequest) {
        long nowMs = System.currentTimeMillis();
        int evicted = 0;
        for (Map.Entry<Long, DecodeRequestState> entry
                : decodeRequests.entrySet()) {
            long requestId = entry.getKey();
            DecodeRequestState request = entry.getValue();
            if (!request.ownsRequest()
                    || nowMs - request.createdAtMs() <= ttlMs
                    || schedulerOwnsRequest.test(requestId)
                    || request.preemptionClaim != null
                    || request.confirmed()
                    || !removeShadowExactLocked(requestId, request)) {
                continue;
            }
            evicted++;
        }
        return evicted;
    }

    private long inflightMaxAgeMs(long nowMs) {
        long oldest = Long.MAX_VALUE;
        for (DecodeRequestState request : decodeRequests.values()) {
            if (request.ownsRequest() && !request.confirmed()) {
                oldest = Math.min(oldest, request.createdAtMs());
            }
        }
        return oldest == Long.MAX_VALUE
                ? 0L : Math.max(0L, nowMs - oldest);
    }

    private int getTotalLoad() {
        return confirmedEngineOwnedCount + reservedRequestCount.get();
    }

    /**
     * Engine-facing load: confirmed running/accepted requests plus
     * reserved entries that are <b>not</b> parked in a prefill queue. Queued
     * reservations remain in the full placement/priority view, but they must
     * not close the Decode concurrency gate while work is still waiting in
     * Prefill. {@link #getTotalLoad()} keeps the full shadow view for
     * observability and eviction planning.
     *
     * <p>The O(1) formula is {@code confirmedEngineOwnedCount
     * + max(0, reservedRequestCount − queuedPhaseCount)}. The
     * {@link #queuedPhaseCount} counter is updated with every queued-phase
     * transition, so this admission read does not scan the queued set.
     *
     * <p><b>Torn-read safety:</b> the counter is read lock-free and may
     * transiently fall outside {@code [0, inflight]} while a lock-owned
     * transition publishes its two fields. Clamp that expected observation;
     * exact admission still revalidates both values under the lock.
     */
    private int getEngineLoad() {
        int inflight = reservedRequestCount.get();
        int queued = queuedPhaseCount.get();
        if (queued < 0 || queued > inflight) {
            queued = Math.max(0, Math.min(queued, inflight));
        }
        return confirmedEngineOwnedCount + Math.max(0, inflight - queued);
    }

    /**
     * Move the exact reservation owned by a queue admission into the queued
     * phase. The generation pin makes a capture accepted before detach valid
     * through this transition; the reservation token fences request-id reuse.
     */
    public boolean markQueuedExact(
            GenerationPin generationPin,
            ReservationHandle reservation) {
        if (reservation == null) {
            throw new IllegalArgumentException(
                    "Decode reservation is required for queued transition");
        }
        boolean capacityChanged = false;
        admissionLock.lock();
        try {
            requirePinnedGeneration(generationPin);
            if (reservation.endpointGenerationId()
                    != getStatus().getGenerationId()) {
                return false;
            }
            long requestId = reservation.requestId();
            DecodeRequestState current = shadowReservation(requestId);
            if (!isExactReservation(current, reservation)) {
                return false;
            }
            if (current.queued()) {
                return true;
            }
            addQueuedPhaseLocked(requestId, current);
            // Re-queueing begins a new dispatch round. Invalidate any
            // pre-delivery lease before publishing that transition.
            removeEngineDispatchPermitLocked(requestId);
            admissionVersion.incrementAndGet();
            capacityChanged = true;
        } finally {
            admissionLock.unlock();
        }
        if (capacityChanged) {
            publishCapacityRelease();
        }
        return true;
    }

    private boolean addQueuedPhaseLocked(long requestId, DecodeRequestState reservation) {
        if (reservation == null || !reservation.markQueued()) {
            return false;
        }
        queuedPhaseCount.incrementAndGet();
        queuedHardKvReservedTotal.addAndGet(reservation.kvTokens());
        queuedExpectedKvReservedTotal.addAndGet(reservation.expectedKvTokens());
        return true;
    }

    private boolean removeQueuedPhaseLocked(long requestId, DecodeRequestState reservation) {
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

    /** Outcome of transferring an acquired permit to engine lifecycle ownership. */
    public enum EngineDispatchPermitTransferStatus {
        TRANSFERRED,
        OWNERSHIP_LOST,
        ENDPOINT_RETIRED
    }

    /** Explicit result of {@link #acquireEngineDispatchPermit(ReservationHandle, AdmissionCapacity)}. */
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

    /**
     * Exact-reservation capability for one Decode delivery handoff.
     *
     * <p>{@link #transferToEngineLifecycle()} atomically converts the still-current queued
     * reservation into engine-facing ownership without checking capacity again.
     * An Engine observation which already confirmed this same reservation also
     * completes the transfer; a replacement reservation never does.
     * When acquired after Engine acceptance, the permit only carries that
     * reservation identity and does not occupy an additional capacity slot.
     * {@link #release()} gives up only this exact temporary hard-gate slot;
     * the reservation stays queued. Both operations are idempotent with respect
     * to endpoint state. Object identity and the reservation generation prevent
     * an old request-id generation from affecting a newer permit.
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
        private final DecodeRequestState reservation;
        /** Guarded by the endpoint admission lock. */
        private boolean retiredByEndpoint;
        private Resolution resolution = Resolution.ACQUIRED;

        private EngineDispatchPermit(DecodeEndpoint endpoint,
                                     long requestId,
                                     DecodeRequestState reservation) {
            this.endpoint = endpoint;
            this.requestId = requestId;
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
     * Acquire one pre-delivery Decode slot for an exact queued reservation.
     * Decode concurrency and KV are validated and occupied under the same
     * admission lock, so concurrent acquisitions cannot oversell either gate.
     * A prior Engine acceptance produces an identity-only permit without
     * occupying additional capacity.
     */
    public EngineDispatchPermitAcquisition acquireEngineDispatchPermit(
            ReservationHandle handle,
            AdmissionCapacity capacity) {
        java.util.Objects.requireNonNull(handle, "handle");
        java.util.Objects.requireNonNull(capacity, "capacity");
        GenerationPin generationPin = tryPinGeneration();
        if (generationPin == null) {
            return withoutEngineDispatchPermit(
                    EngineDispatchPermitAcquireStatus.ENDPOINT_RETIRED);
        }
        try (generationPin) {
            return acquireEngineDispatchPermitPinned(
                    handle, capacity);
        }
    }

    private EngineDispatchPermitAcquisition acquireEngineDispatchPermitPinned(
            ReservationHandle handle,
            AdmissionCapacity capacity) {
        admissionLock.lock();
        try {
            DecodeRequestState reservation = requestState(handle.requestId());
            if (handle.endpointGenerationId() != getStatus().getGenerationId()
                    || !isExactReservation(reservation, handle)
                    || !reservation.ownsRequest() || reservation.preemptionClaim != null) {
                return withoutEngineDispatchPermit(
                        EngineDispatchPermitAcquireStatus.NOT_OWNED);
            }
            if (reservation.confirmed()) {
                return new EngineDispatchPermitAcquisition(
                        EngineDispatchPermitAcquireStatus.ALREADY_ACCEPTED,
                        new EngineDispatchPermit(this, handle.requestId(), reservation));
            }
            if (!reservation.queued()) {
                return withoutEngineDispatchPermit(
                        EngineDispatchPermitAcquireStatus.NOT_QUEUED);
            }
            if (reservation.dispatchPermit() != null) {
                return withoutEngineDispatchPermit(
                        EngineDispatchPermitAcquireStatus.ALREADY_ACQUIRED);
            }
            if (isEngineDispatchCapacityFullLocked(
                    reservation, capacity)) {
                return withoutEngineDispatchPermit(
                        EngineDispatchPermitAcquireStatus.CAPACITY_FULL);
            }

            EngineDispatchPermit permit = installEngineDispatchPermitLocked(
                    handle.requestId(), reservation);
            return new EngineDispatchPermitAcquisition(
                    EngineDispatchPermitAcquireStatus.ACQUIRED, permit);
        } finally {
            admissionLock.unlock();
        }
    }

    private static EngineDispatchPermitAcquisition withoutEngineDispatchPermit(
            EngineDispatchPermitAcquireStatus status) {
        return new EngineDispatchPermitAcquisition(status, null);
    }

    /** Caller holds admissionLock and has validated exact queued ownership. */
    private EngineDispatchPermit installEngineDispatchPermitLocked(
            long requestId,
            DecodeRequestState reservation) {
        EngineDispatchPermit permit = new EngineDispatchPermit(
                this, requestId, reservation);
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
            // Engine status may consume the acquired permit before publication.
            // Only the same canonical reservation can satisfy this handoff.
            DecodeRequestState current = requestState(permit.requestId);
            if (current == permit.reservation && current.confirmed()
                    && current.preemptionClaim == null) {
                return EngineDispatchPermitTransferStatus.TRANSFERRED;
            }
            int usageBefore = engineDispatchHardGateUsageLocked();
            if (!isCurrentEngineDispatchPermitLocked(permit)) {
                return EngineDispatchPermitTransferStatus.OWNERSHIP_LOST;
            }
            if (shadowReservation(permit.requestId) != permit.reservation
                    || !permit.reservation.queued()
                    || permit.reservation.preemptionClaim != null) {
                removeEngineDispatchPermitLocked(permit.requestId);
                admissionVersion.incrementAndGet();
                transferStatus = EngineDispatchPermitTransferStatus.OWNERSHIP_LOST;
            } else {
                removeEngineDispatchPermitLocked(permit.requestId);
                // The identity and queued membership were checked while holding the
                // same lock. Transfer only changes ownership; it never re-reads the cap.
                removeQueuedPhaseLocked(permit.requestId, permit.reservation);
                permit.reservation.engineLifecycleOwned = true;
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
            if (permit.retiredByEndpoint) {
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
            publishCapacityRelease();
        }
        return true;
    }

    private boolean isCurrentEngineDispatchPermitLocked(EngineDispatchPermit permit) {
        return permit.reservation.dispatchPermit() == permit
                && shadowReservation(permit.requestId) == permit.reservation;
    }

    private boolean removeEngineDispatchPermitLocked(long requestId) {
        DecodeRequestState reservation = shadowReservation(requestId);
        return reservation != null
                && removeEngineDispatchPermitLocked(reservation);
    }

    private boolean removeEngineDispatchPermitLocked(
            DecodeRequestState reservation) {
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
                reservedRequestCount.get() - queuedPhaseCount.get());
        return confirmedEngineOwnedCount + engineFacingInflight
                + activeEngineDispatchPermitCount;
    }

    /** Preemptive placement accounts for every queued and Engine-owned reservation. */
    private boolean queuedPlacementIsFullLocked(
            long hardKvTokens, long expectedKvTokens, AdmissionCapacity capacity) {
        return !capacity.evaluate(routingViewLocked().placementUsage(), hardKvTokens, expectedKvTokens).fits();
    }

    /** Caller holds admissionLock; this is the authoritative pre-admission gate. */
    private boolean isEngineDispatchCapacityFullLocked(
            DecodeRequestState candidate,
            AdmissionCapacity capacity) {
        WorkerStatus.CommittedWorkerStatus committed =
                getStatus().committedWorkerStatus();
        return isEngineDispatchCapacityFullSnapshot(
                candidate,
                capacity,
                committed.fields());
    }

    /**
     * Authoritative post-eviction projection. The caller has already validated
     * every exact victim under {@link #admissionLock}; this method evaluates
     * current canonical owners minus those victims plus the incoming owner
     * before any claim, removal, or reservation is installed.
     */
    private boolean projectedEvictionCapacityFitsLocked(
            AdmissionCapacity capacity, long hardKvTokens, long expectedKvTokens, CapacityRelease released) {
        return capacity.evaluate(routingViewLocked().placementUsage(), hardKvTokens, expectedKvTokens, released).fits();
    }

    /**
     * Common O(1) gate math used by authoritative acquisition and the live
     * waiter hint. Acquired permits are already included in both the slot and
     * KV counters; {@code candidate} is still a queued soft reservation and is
     * projected exactly once here.
     */
    private boolean isEngineDispatchCapacityFullSnapshot(
            DecodeRequestState candidate,
            AdmissionCapacity capacity,
            WorkerStatus.EngineObservation fields) {
        return !capacity.evaluate(dispatchCapacityUsage(fields), candidate.kvTokens(), candidate.expectedKvTokens()).fits();
    }

    /** Capture the Engine-facing scope; queued soft reservations consume no dispatch capacity. */
    private CapacityUsage dispatchCapacityUsage(WorkerStatus.EngineObservation fields) {
        long heldHard = priorityPreemptionHeldKv.get();
        long dispatchHard = saturatedAddNonNegative(
                saturatedAddNonNegative(Math.max(0L, inflightHardKvReserved() - queuedHardKvReservedTotal.get()),
                        engineDispatchPermitHardKvReservedTotal.get()), heldHard);
        return new CapacityUsage(
                getEngineLoad() + Math.max(0, activeEngineDispatchPermitCount),
                Math.max(0L, fields.totalKvCacheTokens()), Math.max(0L, fields.availableKvCacheTokens()),
                dispatchHard, engineFacingKvUsed(fields));
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
            AdmissionCapacity capacity) {
        if (isGenerationRetiringOrRetired()) {
            return true;
        }
        DecodeRequestState candidate = shadowReservation(requestId);
        if (candidate == null
                || !candidate.queued()
                || candidate.dispatchPermit() != null) {
            return true;
        }
        WorkerStatus.CommittedWorkerStatus committed =
                getStatus().committedWorkerStatus();
        return !isEngineDispatchCapacityFullSnapshot(
                candidate,
                capacity,
                committed.fields());
    }

    @Override
    public OptionalLong getLoadMetric() {
        return OptionalLong.of(getTotalLoad());
    }

    private enum ClaimOwner {
        SHADOW_IN_FLIGHT,
        ENGINE_CONFIRMED
    }

    private static final class PreemptionClaim {
        private final long attemptToken;
        private ClaimOwner owner;
        private final long hardKvTokens;
        private final long expectedKvTokens;
        private PreemptionCancelPhase phase = PreemptionCancelPhase.CLAIMED;
        private boolean kvHeldAfterWorkerRelease;

        private PreemptionClaim(long attemptToken, ClaimOwner owner,
                                long hardKvTokens, long expectedKvTokens) {
            if (hardKvTokens < 0L || expectedKvTokens < hardKvTokens) {
                throw new IllegalArgumentException(
                        "Priority preemption claim requires expected KV >= hard KV >= 0");
            }
            this.attemptToken = attemptToken;
            this.owner = owner;
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
