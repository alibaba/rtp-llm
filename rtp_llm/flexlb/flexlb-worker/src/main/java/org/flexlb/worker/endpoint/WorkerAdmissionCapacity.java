package org.flexlb.worker.endpoint;

import java.util.Objects;

/**
 * Reports a worker's remaining request-count admission budget at one resource revision.
 * The generation identifies one WorkerStatus across all roles within this process.
 * The budget bounds scheduling retries; request-specific KV checks and registration still belong to the endpoint.
 */
public record WorkerAdmissionCapacity(
        CapacityKey key, long generation, long revision, long availableRequests) {
    public WorkerAdmissionCapacity {
        Objects.requireNonNull(key, "key");
        if (key.endpoint() == null || availableRequests < 0) {
            throw new IllegalArgumentException(
                    "worker identity and nonnegative capacity are required");
        }
    }
}
