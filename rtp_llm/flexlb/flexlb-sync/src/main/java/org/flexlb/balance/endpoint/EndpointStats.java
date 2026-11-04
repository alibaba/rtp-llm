package org.flexlb.balance.endpoint;

import org.flexlb.dao.route.RoleType;

/** Read-only node counts for monitoring. */
public interface EndpointStats {
    record Counts(int discovered, int published) {}

    int discoveredCount();

    Counts counts(RoleType role);

    long blockSize(RoleType role);

    record DecodeCounts(
            int inflight, int totalLoad, long expectedKv, long hardKv, long oldestAgeMs) {}

    interface Prefill {
        String getIp();

        int queueDepth();

        org.flexlb.balance.endpoint.EndpointStats.QueueCounts queueCounters();

        org.flexlb.balance.endpoint.EndpointStats.PrefillCounts stats();
    }

    interface Decode {
        String getIp();

        DecodeCounts stats();

        DecodeResources.ResourceSnapshot resourceSnapshot();
    }

    java.util.Map<String, Prefill> prefillStats();

    java.util.Map<String, Decode> decodeStats();

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

    /** Diagnostic counters captured from one queue ownership revision. */
    public record QueueCounts(
            long version, int[] byPriority, long outstandingRequests, int batchSlots) {}
}
