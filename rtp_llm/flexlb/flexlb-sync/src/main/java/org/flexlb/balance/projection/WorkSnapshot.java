package org.flexlb.balance.projection;

import java.util.List;
import java.util.OptionalLong;

/**
 * Immutable view of Prefill work which has crossed an endpoint lifecycle boundary.
 *
 * <p>Known request and batch work is intentionally kept separate from
 * {@link #unknownRequestCount()}. A committed batch whose repack prediction is
 * unavailable retains its request identities and carries an empty
 * {@link BatchWork#remainingWorkMs()}. Neither form of unknown work may be
 * converted into fabricated milliseconds by a load projection.
 */
public record WorkSnapshot(
        long capturedAtMs,
        List<RequestWork> requests,
        List<BatchWork> batches,
        long unknownRequestCount) {

    public WorkSnapshot {
        requests = List.copyOf(requests);
        batches = List.copyOf(batches);
        if (unknownRequestCount < 0L) {
            throw new IllegalArgumentException(
                    "unknownRequestCount must be non-negative");
        }
    }

    /** Lifecycle phase visible to a projection. Only ENGINE_RUNNING consumes time. */
    public enum Phase {
        COMMITTED,
        ENGINE_QUEUED,
        ENGINE_RUNNING
    }

    /** One individually delivered request, identified by request id. */
    public record RequestWork(long requestId,
                              Phase phase,
                              long remainingWorkMs) {

        public RequestWork {
            if (remainingWorkMs < 0L) {
                throw new IllegalArgumentException(
                        "remaining request work must be non-negative");
            }
        }
    }

    /** One EnqueueBatch work unit, identified by batch id and its live members. */
    public record BatchWork(long batchId,
                            List<Long> requestIds,
                            Phase phase,
                            OptionalLong remainingWorkMs) {

        public BatchWork {
            requestIds = List.copyOf(requestIds);
            if (remainingWorkMs.isPresent()
                    && remainingWorkMs.getAsLong() < 0L) {
                throw new IllegalArgumentException(
                        "remaining batch work must be non-negative");
            }
        }

        /** Convenience constructor for a batch with a known work estimate. */
        public BatchWork(long batchId,
                         List<Long> requestIds,
                         Phase phase,
                         long remainingWorkMs) {
            this(batchId, requestIds, phase, OptionalLong.of(remainingWorkMs));
        }
    }

    public boolean hasUnknownWork() {
        if (unknownRequestCount > 0) {
            return true;
        }
        for (BatchWork batch : batches) {
            if (batch.remainingWorkMs().isEmpty()) {
                return true;
            }
        }
        return false;
    }

    /**
     * Complete committed duration, absent when any work unit lacks a duration.
     */
    public OptionalLong totalRemainingWorkMs() {
        return hasUnknownWork()
                ? OptionalLong.empty()
                : OptionalLong.of(knownRemainingWorkMs());
    }

    /**
     * Sum of work units whose duration is known. Callers that require a complete
     * endpoint total must use {@link #totalRemainingWorkMs()}.
     */
    public long knownRemainingWorkMs() {
        long total = 0L;
        for (RequestWork request : requests) {
            total = saturatedAdd(total, request.remainingWorkMs());
        }
        for (BatchWork batch : batches) {
            if (batch.remainingWorkMs().isPresent()) {
                total = saturatedAdd(
                        total, batch.remainingWorkMs().getAsLong());
            }
        }
        return total;
    }

    private static long saturatedAdd(long left, long right) {
        return left > Long.MAX_VALUE - right ? Long.MAX_VALUE : left + right;
    }
}
