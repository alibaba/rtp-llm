package org.flexlb.balance.autotpm;

import org.flexlb.util.Logger;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * Master-side tracker for per-endpoint decode reservations.
 *
 * <p>Maintains the Master's view of each decode endpoint's reservation state
 * (the 3-state model: {@link DecodeAdmissionState}). This is separate from
 * {@link org.flexlb.balance.endpoint.DecodeEndpoint}'s inflight tracking —
 * that class tracks raw KV/slot usage for load balancing and calibration,
 * while this tracker adds the priority-aware eviction layer.
 *
 * <h2>Concurrency</h2>
 * All maps are {@link ConcurrentHashMap}. Individual operations (reserve,
 * release, markAccepted, markRunning) are atomic per-request. Candidate
 * queries take a snapshot and return a consistent list, but the caller should
 * treat results as a point-in-time view — the committer handles races by
 * using removeIfPresent semantics.
 *
 * <h2>State transitions</h2>
 * <pre>
 * RESERVED_NOT_ACCEPTED → ACCEPTED_NOT_RUNNING → RUNNING
 *                                  ↓
 *                             (release on completion/cancel/error)
 * </pre>
 * Transitions are idempotent — re-marking a state that has already progressed
 * past the target is a no-op.
 */
public class DecodeAdmissionTracker {

    private static final org.slf4j.Logger log = org.slf4j.LoggerFactory.getLogger(DecodeAdmissionTracker.class);

    // Map: decodeEndpointKey (ip:port) → Map<requestId, DecodeReservation>
    private final ConcurrentHashMap<String, ConcurrentHashMap<Long, DecodeReservation>> reservationsByEndpoint =
            new ConcurrentHashMap<>();

    // ==================== Lifecycle: Register / Update / Release ====================

    /**
     * Register a new reservation for a request on a decode endpoint.
     *
     * @param endpointKey ip:port of the decode endpoint
     * @param reservation the reservation to register
     */
    public void reserve(String endpointKey, DecodeReservation reservation) {
        reservationsByEndpoint
                .computeIfAbsent(endpointKey, k -> new ConcurrentHashMap<>())
                .put(reservation.requestId(), reservation);
        log.debug("DecodeAdmissionTracker reserve: ep={} reqId={} pri={} kv={}",
                endpointKey, reservation.requestId(),
                reservation.priority(), reservation.kvTokensRequired());
    }

    /**
     * Transition a reservation from RESERVED_NOT_ACCEPTED to ACCEPTED_NOT_RUNNING.
     * Called when the Engine acknowledges the request (TaskPhase = KV_ALLOCATED).
     *
     * @param endpointKey ip:port of the decode endpoint
     * @param requestId   request ID
     */
    public void markAccepted(String endpointKey, long requestId) {
        DecodeReservation r = getReservation(endpointKey, requestId);
        if (r != null && r.state() == DecodeAdmissionState.RESERVED_NOT_ACCEPTED) {
            r.setState(DecodeAdmissionState.ACCEPTED_NOT_RUNNING);
            log.debug("DecodeAdmissionTracker markAccepted: ep={} reqId={}",
                    endpointKey, requestId);
        }
    }

    /**
     * Transition a reservation to RUNNING.
     * Called when the Engine starts actively decoding (TaskPhase = RUNNING).
     *
     * @param endpointKey ip:port of the decode endpoint
     * @param requestId   request ID
     */
    public void markRunning(String endpointKey, long requestId) {
        DecodeReservation r = getReservation(endpointKey, requestId);
        if (r != null && r.state() != DecodeAdmissionState.RUNNING) {
            r.setState(DecodeAdmissionState.RUNNING);
            r.setRunningSinceMs(System.currentTimeMillis());
            log.debug("DecodeAdmissionTracker markRunning: ep={} reqId={}",
                    endpointKey, requestId);
        }
    }

    /**
     * Remove a reservation (on completion/cancel/error).
     *
     * @param endpointKey ip:port of the decode endpoint
     * @param requestId   request ID
     * @return the removed reservation, or {@code null} if not found
     */
    public DecodeReservation release(String endpointKey, long requestId) {
        ConcurrentHashMap<Long, DecodeReservation> epMap =
                reservationsByEndpoint.get(endpointKey);
        if (epMap == null) {
            return null;
        }
        DecodeReservation removed = epMap.remove(requestId);
        if (removed != null) {
            log.debug("DecodeAdmissionTracker release: ep={} reqId={} state={}",
                    endpointKey, requestId, removed.state());
        }
        return removed;
    }

    /**
     * Remove and return a reservation. Used by the eviction committer —
     * only succeeds if the reservation is still present and evictable.
     *
     * @param endpointKey ip:port of the decode endpoint
     * @param requestId   request ID
     * @return the removed reservation if it was evictable, {@code null} otherwise
     */
    public DecodeReservation removeIfEvictable(String endpointKey, long requestId) {
        ConcurrentHashMap<Long, DecodeReservation> epMap =
                reservationsByEndpoint.get(endpointKey);
        if (epMap == null) {
            return null;
        }
        DecodeReservation r = epMap.get(requestId);
        if (r == null || !r.isEvictable()) {
            return null;
        }
        // CAS: only remove if the reference hasn't changed (no concurrent state mutation)
        if (epMap.remove(requestId, r)) {
            return r;
        }
        // Someone else mutated it — re-check
        r = epMap.get(requestId);
        if (r != null && r.isEvictable() && epMap.remove(requestId, r)) {
            return r;
        }
        return null;
    }

    // ==================== Capacity Queries ====================

    /**
     * Get available decode slots (capacity - reserved count).
     *
     * @param endpointKey   ip:port of the decode endpoint
     * @param totalCapacity total slot capacity (e.g. decodeConcurrencyLimit)
     * @return available slots (>= 0)
     */
    public int availableSlots(String endpointKey, int totalCapacity) {
        ConcurrentHashMap<Long, DecodeReservation> epMap =
                reservationsByEndpoint.get(endpointKey);
        int reserved = epMap != null ? epMap.size() : 0;
        return Math.max(0, totalCapacity - reserved);
    }

    /**
     * Get available KV (total - sum of reserved KV).
     *
     * @param endpointKey ip:port of the decode endpoint
     * @param totalKv     total KV cache tokens
     * @return available KV (>= 0)
     */
    public long availableKv(String endpointKey, long totalKv) {
        ConcurrentHashMap<Long, DecodeReservation> epMap =
                reservationsByEndpoint.get(endpointKey);
        if (epMap == null || epMap.isEmpty()) {
            return totalKv;
        }
        long reservedKv = epMap.values().stream()
                .mapToLong(DecodeReservation::kvTokensRequired)
                .sum();
        return Math.max(0, totalKv - reservedKv);
    }

    // ==================== Eviction Candidate Selection ====================

    /**
     * Find eviction candidates for slot shortage.
     *
     * <p>Filter: state != RUNNING, priority < incomingPriority (hard rule).
     * <p>Sort: priority ascending (lowest first), same priority earlier stage first
     *          (RESERVED_NOT_ACCEPTED before ACCEPTED_NOT_RUNNING), same priority same stage
     *          earlier arrival first (createdAtMs ascending, then requestId ascending).
     *
     * @param endpointKey       ip:port of the decode endpoint
     * @param incomingPriority  priority of the incoming request
     * @param neededSlots       number of slots needed
     * @return sorted list of eviction candidates (up to neededSlots)
     */
    public List<DecodeReservation> findSlotEvictionCandidates(
            String endpointKey, int incomingPriority, int neededSlots) {
        List<DecodeReservation> candidates = getEvictableCandidates(endpointKey, incomingPriority);
        if (candidates.isEmpty()) {
            return Collections.emptyList();
        }
        // Sort: priority asc, stage asc (earlier stage first), arrival asc, requestId asc
        candidates.sort(SLOT_EVICT_COMPARATOR);
        int selectCount = Math.min(neededSlots, candidates.size());
        return new ArrayList<>(candidates.subList(0, selectCount));
    }

    /**
     * Find eviction candidates for KV shortage.
     *
     * <p>Filter: state != RUNNING, priority < incomingPriority (hard rule).
     * <p>Sort: priority ascending (lowest first), same priority more KV released first
     *          (kvTokensRequired descending), same priority same KV earlier stage first,
     *          same priority same KV same stage earlier arrival first.
     *
     * <p>Greedy selection: take candidates in sorted order until the cumulative
     * KV released satisfies {@code neededKv}.
     *
     * @param endpointKey      ip:port of the decode endpoint
     * @param incomingPriority priority of the incoming request
     * @param neededKv        KV tokens needed
     * @return sorted list of eviction candidates (greedily selected to satisfy neededKv)
     */
    public List<DecodeReservation> findKvEvictionCandidates(
            String endpointKey, int incomingPriority, long neededKv) {
        List<DecodeReservation> candidates = getEvictableCandidates(endpointKey, incomingPriority);
        if (candidates.isEmpty()) {
            return Collections.emptyList();
        }
        // Sort: priority asc, KV desc (more released first), stage asc, arrival asc, requestId asc
        candidates.sort(KV_EVICT_COMPARATOR);
        // Greedy: select until cumulative KV >= neededKv
        List<DecodeReservation> selected = new ArrayList<>();
        long cumulative = 0;
        for (DecodeReservation r : candidates) {
            if (cumulative >= neededKv) {
                break;
            }
            selected.add(r);
            cumulative += r.kvTokensRequired();
        }
        return selected;
    }

    // =================--- Internal Helpers ---====================

    public DecodeReservation getReservation(String endpointKey, long requestId) {
        ConcurrentHashMap<Long, DecodeReservation> epMap =
                reservationsByEndpoint.get(endpointKey);
        return epMap != null ? epMap.get(requestId) : null;
    }

    /**
     * Get all evictable candidates for an endpoint that have strictly lower priority.
     */
    private List<DecodeReservation> getEvictableCandidates(String endpointKey, int incomingPriority) {
        ConcurrentHashMap<Long, DecodeReservation> epMap =
                reservationsByEndpoint.get(endpointKey);
        if (epMap == null || epMap.isEmpty()) {
            return Collections.emptyList();
        }
        return epMap.values().stream()
                .filter(DecodeReservation::isEvictable)           // state != RUNNING
                .filter(r -> r.priority() < incomingPriority)      // hard rule
                .collect(Collectors.toCollection(ArrayList::new));
    }

    /**
     * Get all reservations for an endpoint (for snapshot/inspection).
     */
    public Collection<DecodeReservation> getReservations(String endpointKey) {
        ConcurrentHashMap<Long, DecodeReservation> epMap =
                reservationsByEndpoint.get(endpointKey);
        return epMap != null ? Collections.unmodifiableCollection(epMap.values())
                : Collections.emptyList();
    }

    /**
     * Get the reservation for a specific request on an endpoint.
     */
    public DecodeReservation getReservation(String endpointKey, long requestId, boolean unused) {
        return getReservation(endpointKey, requestId);
    }

    /**
     * Total reservation count across all endpoints.
     */
    public int totalReservations() {
        return reservationsByEndpoint.values().stream()
                .mapToInt(Map::size)
                .sum();
    }

    // ==================== Comparators ====================

    /**
     * Stage ordinal for sorting: RESERVED_NOT_ACCEPTED = 0, ACCEPTED_NOT_RUNNING = 1.
     * (RUNNING is already filtered out.)
     */
    private static int stageOrdinal(DecodeAdmissionState state) {
        return switch (state) {
            case RESERVED_NOT_ACCEPTED -> 0;
            case ACCEPTED_NOT_RUNNING -> 1;
            case RUNNING -> 2; // never reaches here (filtered), but kept for completeness
        };
    }

    /**
     * Slot eviction comparator: priority asc, stage asc, arrival asc, requestId asc.
     */
    private static final Comparator<DecodeReservation> SLOT_EVICT_COMPARATOR =
            Comparator.comparingInt(DecodeReservation::priority)
                    .thenComparingInt(r -> stageOrdinal(r.state()))
                    .thenComparingLong(DecodeReservation::createdAtMs)
                    .thenComparingLong(DecodeReservation::requestId);

    /**
     * KV eviction comparator: priority asc, KV desc, stage asc, arrival asc, requestId asc.
     */
    private static final Comparator<DecodeReservation> KV_EVICT_COMPARATOR =
            Comparator.comparingInt(DecodeReservation::priority)
                    .thenComparing(Comparator.comparingLong(DecodeReservation::kvTokensRequired).reversed())
                    .thenComparingInt(r -> stageOrdinal(r.state()))
                    .thenComparingLong(DecodeReservation::createdAtMs)
                    .thenComparingLong(DecodeReservation::requestId);
}
