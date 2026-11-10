package org.flexlb.balance.endpoint;

import org.flexlb.dao.route.RoleType;

/**
 * Read-only node counts for monitoring.
 */
public interface EndpointStats {
    int discoveredCount();

    Counts counts(RoleType role);

    long blockSize(RoleType role);

    java.util.Map<String, Prefill> prefillStats();

    java.util.Map<String, Decode> decodeStats();

    record Counts(int discovered, int published) {}

    record DecodeCounts(
            int inflight, int totalLoad, long expectedKv, long hardKv, long oldestAgeMs) {}

    /**
     * Freezes Prefill request and batch changes together with scheduling and capacity notifications.
     */
    interface Prefill {
        String getIp();

        int queueDepth();

        org.flexlb.balance.endpoint.EndpointStats.QueueCounts queueCounters();

        org.flexlb.balance.endpoint.EndpointStats.PrefillCounts stats();
    }

    /**
     * Freezes Decode request changes and the placement and dispatch capacity notification flags.
     */
    interface Decode {
        String getIp();

        DecodeCounts stats();

        DecodeResources.ResourceSnapshot resourceSnapshot();
    }

    public record PrefillCounts(
            int locallyOwnedRequests,
            int individuallyOwnedRequests,
            int batchCount,
            long maxObservedAgeMs) {
        public PrefillCounts {
            com.google.common.base.Preconditions.checkArgument(
                    locallyOwnedRequests >= 0
                            && individuallyOwnedRequests >= 0
                            && batchCount >= 0
                            && maxObservedAgeMs >= 0L,
                    "Prefill state stats must be non-negative");
        }
    }

    /**
     * Diagnostic counters captured from one queue ownership revision.
     */
    public record QueueCounts(
            long version, int[] byPriority, long outstandingRequests, int batchSlots) {}
}
