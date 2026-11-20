package org.flexlb.worker.endpoint;

import org.flexlb.config.service.RoleType;

/**
 * Read-only node counts for monitoring.
 */
public interface EndpointStats {
    int discoveredCount();

    Counts counts(RoleType role);

    long blockSize(RoleType role);

    <T extends WorkerEndpoint> java.util.List<T> endpoints(RoleType role, String group);

    record Counts(int discovered, int published) {}

    record DecodeCounts(
            int inflight,
            int totalLoad,
            long expectedKv,
            long hardKv,
            long oldestAgeMs,
            int accepted,
            int running) {}

    /**
     * Freezes Prefill request changes together with scheduling and capacity notifications.
     */
    interface Prefill {
        String getIp();

        org.flexlb.worker.endpoint.EndpointStats.PrefillCounts stats();
    }

    /**
     * Reads Decode request and capacity counts without copying request records.
     */
    interface Decode {
        String getIp();

        DecodeCounts stats();
    }

    /**
     * Non-negative Prefill request count and maximum observation age in milliseconds.
     */
    public record PrefillCounts(int locallyOwnedRequests, long maxObservedAgeMs) {
        public PrefillCounts {
            com.google.common.base.Preconditions.checkArgument(
                    locallyOwnedRequests >= 0 && maxObservedAgeMs >= 0L,
                    "Prefill state stats must be non-negative");
        }
    }
}
