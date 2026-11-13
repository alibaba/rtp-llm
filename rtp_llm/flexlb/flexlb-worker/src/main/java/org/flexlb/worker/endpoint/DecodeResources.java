package org.flexlb.worker.endpoint;

import org.flexlb.config.routing.RoutingConfig;
import org.flexlb.worker.endpoint.DecodeEndpoint.RequestRegistration;
import org.flexlb.worker.status.DecodeTaskPhase;

import java.util.Map;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.math.LongMath.saturatedAdd;

/**
 * Immutable Decode resource identities, capacity arithmetic, snapshots and observed facts.
 */
public final class DecodeResources {
    private DecodeResources() {}

    /**
     * A cheap, exact resource query; it never grants a reservation or dispatch right.
     */
    public enum RegistrationState {
        NOT_REGISTERED,
        UNCONFIRMED,
        CONFIRMED;

        public boolean registered() {
            return this != NOT_REGISTERED;
        }

        public boolean confirmed() {
            return this == CONFIRMED;
        }

        public boolean unconfirmed() {
            return this == UNCONFIRMED;
        }
    }

    /**
     * Request status matched to its exact Decode reservation, published after ledger reconciliation.
     */
    public record DecodeRequestStatus(Kind kind, RequestRegistration reservation, long errorCode) {
        public DecodeRequestStatus {
            java.util.Objects.requireNonNull(kind, "kind");
            java.util.Objects.requireNonNull(reservation, "reservation");
            checkArgument(
                    kind == Kind.TERMINAL || errorCode == 0L,
                    "only a terminal Decode request status may carry an error code");
        }

        public static DecodeRequestStatus confirmed(RequestRegistration reservation) {
            return new DecodeRequestStatus(Kind.CONFIRMED, reservation, 0L);
        }

        public static DecodeRequestStatus terminal(
                RequestRegistration reservation, long errorCode) {
            return new DecodeRequestStatus(Kind.TERMINAL, reservation, errorCode);
        }

        public enum Kind {
            CONFIRMED,
            TERMINAL
        }
    }

    /**
     * One immutable request table and capacity view captured in one Endpoint lock.
     */
    public record ResourceSnapshot(
            DecodeEndpoint.SelectionSnapshot selection, Map<String, DecodeRequestView> requests) {
        public ResourceSnapshot {
            requests = Map.copyOf(requests);
        }
    }

    /**
     * Freezes endpoint identity, observed load and registered capacity for Decode selection.
     */
    public record DecodeRequestView(
            String requestId,
            int priority,
            long kvTokens,
            long expectedKvTokens,
            DecodeTaskPhase phase,
            RequestRegistration reservation) {
        public Capacity placementRelease() {
            return new Capacity(1L, kvTokens, expectedKvTokens);
        }
    }

    /**
     * Computes request and KV deficits against configured limits, accounting for exact releases.
     * A zero request limit is unlimited; the KV limit is expressed as a percentage.
     */
    public record AdmissionCapacity(long maxEngineRequests, long maxKvUsagePercent) {

        public AdmissionCapacity {
            checkArgument(
                    maxEngineRequests >= 0L
                            && maxKvUsagePercent > 0L
                            && maxKvUsagePercent <= RoutingConfig.PERCENTAGE_SCALE,
                    "Decode admission limits are outside their domain");
        }

        /**
         * Use the same occupancy scope for the observation and every exact request selected for
         * preemption release.
         */
        public Capacity evaluate(
                CapacityUsage usage, long hardKvTokens, long expectedKvTokens, Capacity release) {
            java.util.Objects.requireNonNull(usage, "usage");
            java.util.Objects.requireNonNull(release, "release");
            checkArgument(
                    hardKvTokens >= 0L && expectedKvTokens >= hardKvTokens,
                    "Decode demand must satisfy expected >= hard >= 0");
            long requests =
                    maxEngineRequests == 0L
                            ? 0L
                            : shortfall(
                                    Math.max(0L, usage.occupiedRequests - release.requests),
                                    1L,
                                    maxEngineRequests,
                                    0L);
            if (usage.totalKvTokens == 0L) {
                return requests == 0L ? Capacity.NONE : new Capacity(requests, 0L, 0L);
            }
            long hard =
                    shortfall(
                            usage.hardReservedKvTokens,
                            hardKvTokens,
                            usage.availableKvTokens,
                            release.hardKvTokens);
            long expected =
                    shortfall(
                            Math.max(0L, usage.expectedKvUsed - release.expectedKvTokens),
                            expectedKvTokens,
                            kvBudget(usage.totalKvTokens),
                            0L);
            return requests == 0L && hard == 0L && expected == 0L
                    ? Capacity.NONE
                    : new Capacity(requests, hard, expected);
        }

        /**
         * Integer arithmetic never rounds the configured KV budget up.
         */
        public long kvBudget(long totalKv) {
            checkArgument(totalKv >= 0L, "negative KV capacity");
            return totalKv / 100L * maxKvUsagePercent + totalKv % 100L * maxKvUsagePercent / 100L;
        }

        private static long shortfall(long used, long incoming, long capacity, long released) {
            long remainingUsed = Math.max(0L, used - released);
            long remainingCapacity =
                    Capacity.saturatedAddNonNegative(capacity, Math.max(0L, released - used));
            return remainingUsed > remainingCapacity
                    ? Capacity.saturatedAddNonNegative(remainingUsed - remainingCapacity, incoming)
                    : Math.max(0L, incoming - (remainingCapacity - remainingUsed));
        }
    }

    /**
     * Describes occupied requests and KV tokens used by Decode admission checks; all quantities are nonnegative.
     */
    public record CapacityUsage(
            long occupiedRequests,
            long totalKvTokens,
            long availableKvTokens,
            long hardReservedKvTokens,
            long expectedKvUsed) {
        public CapacityUsage {
            checkArgument(
                    occupiedRequests >= 0L
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

    /**
     * Request slots and KV token amounts used for occupancy, release estimates and capacity gaps.
     * Each dimension is nonnegative; hard and expected gaps are independent.
     */
    public record Capacity(long requests, long hardKvTokens, long expectedKvTokens) {
        public static final Capacity NONE = new Capacity(0L, 0L, 0L);

        public Capacity {
            checkArgument(
                    requests >= 0L && hardKvTokens >= 0L && expectedKvTokens >= 0L,
                    "Decode capacity must be non-negative");
        }

        public Capacity plus(Capacity other) {
            return new Capacity(
                    saturatedAdd(requests, other.requests),
                    saturatedAdd(hardKvTokens, other.hardKvTokens),
                    saturatedAdd(expectedKvTokens, other.expectedKvTokens));
        }

        public boolean isZero() {
            return requests == 0L && !hasKv();
        }

        public boolean hasKv() {
            return hardKvTokens > 0L || expectedKvTokens > 0L;
        }

        public long kvTokens() {
            return Math.max(hardKvTokens, expectedKvTokens);
        }

        static long saturatedAddNonNegative(long left, long right) {
            checkArgument(left >= 0 && right >= 0, "KV admission counters must be non-negative");
            return saturatedAdd(left, right);
        }
    }
}
