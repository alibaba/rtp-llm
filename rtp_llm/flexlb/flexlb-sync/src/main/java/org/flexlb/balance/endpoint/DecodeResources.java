package org.flexlb.balance.endpoint;

import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.enums.DecodeTaskPhase;

import java.util.Map;
import java.util.function.Predicate;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.math.LongMath.saturatedAdd;

/** Immutable Decode resource identities, capacity arithmetic, snapshots and observed facts. */
public final class DecodeResources {
    private DecodeResources() { }

    public record ReservationHandle(
            long endpointGenerationId,
            long requestId,
            long reservationToken) {

        public ReservationHandle {
            checkArgument(endpointGenerationId > 0L && reservationToken > 0L,
                    "Decode reservation identity must be positive");
        }
    }

    public enum ReleaseReason {
        LOCAL_ROLLBACK, COUNTERPART_FINISHED, NOT_SENT, REMOTE_CLEANUP, EXPIRED
    }

    public enum ReservationReleaseResult {
        RELEASED,
        ENGINE_ACCEPTED,
        STILL_OWNED,
        STALE,
        CONFLICT
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

    /** The caller has resolved request protocol; this operation changes only exact resource ownership. */
    public record PreemptionUpdate(Kind kind, ReservationHandle reservation) {
        public enum Kind { CANCEL_HANDED_OFF, CANCELED, REQUEST_FENCED, ACTIVE, FINISHED }

        public PreemptionUpdate {
            java.util.Objects.requireNonNull(kind, "kind");
            java.util.Objects.requireNonNull(reservation, "reservation");
        }
        public static PreemptionUpdate handedOff(ReservationHandle victim) { return new PreemptionUpdate(Kind.CANCEL_HANDED_OFF, victim); }
        public static PreemptionUpdate canceled(ReservationHandle victim) { return new PreemptionUpdate(Kind.CANCELED, victim); }
        public static PreemptionUpdate fenced(ReservationHandle victim) { return new PreemptionUpdate(Kind.REQUEST_FENCED, victim); }
        public static PreemptionUpdate active(ReservationHandle victim) { return new PreemptionUpdate(Kind.ACTIVE, victim); }
        public static PreemptionUpdate finished(ReservationHandle victim) { return new PreemptionUpdate(Kind.FINISHED, victim); }
        boolean releasesCapacity() { return kind != Kind.CANCEL_HANDED_OFF; }
    }

    /** Request status matched to its exact Decode reservation, published after ledger reconciliation. */
    public record DecodeRequestStatus(
            Kind kind,
            ReservationHandle reservation,
            long errorCode,
            boolean allocationObserved) {
        public DecodeRequestStatus {
            java.util.Objects.requireNonNull(kind, "kind");
            java.util.Objects.requireNonNull(reservation, "reservation");
            checkArgument(kind == Kind.TERMINAL || errorCode == 0L,
                    "only a terminal Decode request status may carry an error code");
        }

        public static DecodeRequestStatus active(ReservationHandle reservation) {
            return new DecodeRequestStatus(Kind.ACTIVE, reservation, 0L, false);
        }

        public static DecodeRequestStatus allocated(ReservationHandle reservation) {
            return new DecodeRequestStatus(Kind.ACTIVE, reservation, 0L, true);
        }

        public static DecodeRequestStatus terminal(
                ReservationHandle reservation, long errorCode) {
            return new DecodeRequestStatus(Kind.TERMINAL, reservation, errorCode, false);
        }

        public enum Kind {
            ACTIVE,
            TERMINAL
        }
    }

    /** One immutable request table and capacity view captured under the admission lock. */
    public record ResourceSnapshot(DecodeRoutingView routing,
                                   Map<Long, DecodeRequestView> requests,
                                   int queuedCount,
                                   int activeDispatchPermits) {
        public ResourceSnapshot {
            requests = Map.copyOf(requests);
        }

        public int reservedCount() {
            return requests.size() - confirmedCount();
        }

        public int confirmedCount() {
            return phaseCount(DecodeTaskPhase::isEngineConfirmed);
        }

        public int acceptedCount() {
            return phaseCount(phase -> phase == DecodeTaskPhase.ACCEPTED_NOT_RUNNING);
        }

        public int runningCount() {
            return phaseCount(phase -> phase == DecodeTaskPhase.RUNNING);
        }

        public int engineCapacityUsed() {
            return routing.engineCapacityUsed();
        }

        private int phaseCount(Predicate<DecodeTaskPhase> matches) {
            int count = 0;
            for (DecodeRequestView task : requests.values()) {
                if (matches.test(task.phase())) {
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
            long inflightHardKv) {

        public int engineCapacityUsed() { return Math.toIntExact(dispatchUsage.occupiedRequests()); }
        public long realKvUsed() { return placementUsage.expectedKvUsed(); }
        public long realKvAvailable() { return placementUsage.hardKvAvailable(); }
        public long totalKv() { return placementUsage.totalKvTokens(); }

        public DecodeRoutingView {
            java.util.Objects.requireNonNull(address, "address");
            java.util.Objects.requireNonNull(topology, "topology");
            java.util.Objects.requireNonNull(workerStatus, "workerStatus");
            checkArgument(generationId > 0L, "Decode routing view requires a positive generation");
        }
    }

    public record DecodeRequestView(long requestId,
                                    int priority,
                                    long kvTokens,
                                    long expectedKvTokens,
                                    DecodeTaskPhase phase,
                                    boolean priorityKnown,
                                    long reservationToken,
                                    boolean claimedForPreemption) {
        public CapacityRelease placementRelease() {
            return new CapacityRelease(1L, kvTokens, expectedKvTokens);
        }
    }

    /** Failure-only priority summary, shared until the admission revision changes. */
    public static final class AdmissionSummary {
        private final DecodeRoutingView routing;
        private final CapacityRelease[] placementOccupancy;
        private final CapacityRelease[] engineOccupancy;

        AdmissionSummary(DecodeRoutingView routing, CapacityRelease[] placementOccupancy,
                         CapacityRelease[] engineOccupancy) {
            this.routing = routing;
            this.placementOccupancy = placementOccupancy;
            this.engineOccupancy = engineOccupancy;
        }

        public DecodeRoutingView routing() { return routing; }
        public CapacityRelease placementOccupancy(int priority) { return placementOccupancy[priority]; }
        public CapacityRelease engineOccupancy(int priority) { return engineOccupancy[priority]; }
    }

    public record AdmissionCapacity(
            long maxEngineRequests,
            long maxKvUsagePercent) {

        public AdmissionCapacity {
            checkArgument(maxEngineRequests >= 0L
                    && maxKvUsagePercent > 0L
                    && maxKvUsagePercent <= RoutingConfig.PERCENTAGE_SCALE,
                    "Decode admission limits are outside their domain");
        }

        /** Use the same occupancy scope for the observation and every exact victim release. */
        public CapacityDeficit evaluate(CapacityUsage usage, long hardKvTokens, long expectedKvTokens,
                                        CapacityRelease release) {
            java.util.Objects.requireNonNull(usage, "usage");
            java.util.Objects.requireNonNull(release, "release");
            checkArgument(hardKvTokens >= 0L && expectedKvTokens >= hardKvTokens,
                    "Decode demand must satisfy expected >= hard >= 0");
            long requests = maxEngineRequests == 0L ? 0L
                    : shortfall(Math.max(0L, usage.occupiedRequests - release.requests),
                            1L, maxEngineRequests, 0L);
            if (usage.totalKvTokens == 0L) {
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
            checkArgument(totalKv >= 0L, "negative KV capacity");
            return totalKv / 100L * maxKvUsagePercent + totalKv % 100L * maxKvUsagePercent / 100L;
        }

        private static long shortfall(long used, long incoming, long capacity, long released) {
            long remainingUsed = Math.max(0L, used - released);
            long remainingCapacity = CapacityRelease.saturatedAddNonNegative(capacity, Math.max(0L, released - used));
            return remainingUsed > remainingCapacity
                    ? CapacityRelease.saturatedAddNonNegative(remainingUsed - remainingCapacity, incoming)
                    : Math.max(0L, incoming - (remainingCapacity - remainingUsed));
        }
    }

    public record CapacityUsage(long occupiedRequests, long totalKvTokens, long availableKvTokens,
                                long hardReservedKvTokens, long expectedKvUsed) {
        public CapacityUsage {
            checkArgument(occupiedRequests >= 0L
                    && totalKvTokens >= 0L
                    && availableKvTokens >= 0L
                    && hardReservedKvTokens >= 0L
                    && expectedKvUsed >= 0L,
                    "Decode occupancy must be non-negative");
        }

        public long hardKvAvailable() {
            return Math.max(0L, availableKvTokens - hardReservedKvTokens);
        }
    }

    public record CapacityRelease(long requests, long hardKvTokens, long expectedKvTokens) {
        public static final CapacityRelease NONE = new CapacityRelease(0L, 0L, 0L);

        public CapacityRelease {
            checkArgument(requests >= 0L && hardKvTokens >= 0L && expectedKvTokens >= hardKvTokens,
                    "invalid Decode capacity release");
        }

        static long saturatedAddNonNegative(long left, long right) {
            checkArgument(left >= 0 && right >= 0, "KV admission counters must be non-negative");
            return saturatedAdd(left, right);
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
}
