package org.flexlb.balance.projection;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;
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
public final class WorkSnapshot {

    private static final long[] EMPTY_LONGS = new long[0];

    private final long capturedAtMs;
    private final List<RequestWork> requests;
    private final List<BatchWork> batches;
    private final long unknownRequestCount;
    private final long knownNonRunningWorkMs;
    private final long[] runningWorkMs;
    private final long[] requestIds;
    private final boolean unknownWork;

    public WorkSnapshot(
            long capturedAtMs,
            List<RequestWork> requests,
            List<BatchWork> batches,
            long unknownRequestCount) {
        this.capturedAtMs = capturedAtMs;
        this.requests = List.copyOf(requests);
        this.batches = List.copyOf(batches);
        if (unknownRequestCount < 0L) {
            throw new IllegalArgumentException(
                    "unknownRequestCount must be non-negative");
        }
        this.unknownRequestCount = unknownRequestCount;

        int runningCount = 0;
        int requestIdCount = this.requests.size();
        long nonRunningMs = 0L;
        boolean hasUnknown = unknownRequestCount > 0L;
        for (RequestWork request : this.requests) {
            if (request.phase() == Phase.ENGINE_RUNNING) {
                runningCount++;
            } else {
                nonRunningMs = saturatedAdd(
                        nonRunningMs, request.remainingWorkMs());
            }
        }
        for (BatchWork batch : this.batches) {
            requestIdCount = Math.addExact(
                    requestIdCount, batch.requestIds().size());
            if (batch.remainingWorkMs().isEmpty()) {
                hasUnknown = true;
            } else if (batch.phase() == Phase.ENGINE_RUNNING) {
                runningCount++;
            } else {
                nonRunningMs = saturatedAdd(
                        nonRunningMs, batch.remainingWorkMs().getAsLong());
            }
        }

        this.knownNonRunningWorkMs = nonRunningMs;
        this.runningWorkMs = runningCount == 0
                ? EMPTY_LONGS : new long[runningCount];
        this.requestIds = requestIdCount == 0
                ? EMPTY_LONGS : new long[requestIdCount];
        this.unknownWork = hasUnknown;
        int runningIndex = 0;
        int requestIdIndex = 0;
        for (RequestWork request : this.requests) {
            requestIds[requestIdIndex++] = request.requestId();
            if (request.phase() == Phase.ENGINE_RUNNING) {
                runningWorkMs[runningIndex++] = request.remainingWorkMs();
            }
        }
        for (BatchWork batch : this.batches) {
            for (long requestId : batch.requestIds()) {
                requestIds[requestIdIndex++] = requestId;
            }
            if (batch.phase() == Phase.ENGINE_RUNNING
                    && batch.remainingWorkMs().isPresent()) {
                runningWorkMs[runningIndex++] =
                        batch.remainingWorkMs().getAsLong();
            }
        }
        Arrays.sort(requestIds);
    }

    public long capturedAtMs() {
        return capturedAtMs;
    }

    public List<RequestWork> requests() {
        return requests;
    }

    public List<BatchWork> batches() {
        return batches;
    }

    public long unknownRequestCount() {
        return unknownRequestCount;
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
        return unknownWork;
    }

    public boolean containsRequest(long requestId) {
        return Arrays.binarySearch(requestIds, requestId) >= 0;
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
        long total = knownNonRunningWorkMs;
        for (long runningMs : runningWorkMs) {
            total = saturatedAdd(total, runningMs);
        }
        return total;
    }

    /** Known work rebased to a later planning clock without copying the snapshot. */
    public long knownRemainingWorkMsAt(long planningAtMs) {
        long elapsedMs = planningAtMs <= capturedAtMs
                ? 0L : planningAtMs - capturedAtMs;
        long total = knownNonRunningWorkMs;
        for (long runningMs : runningWorkMs) {
            long remaining = elapsedMs >= runningMs
                    ? 0L : runningMs - elapsedMs;
            total = saturatedAdd(total, remaining);
        }
        return total;
    }

    @Override
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }
        if (!(other instanceof WorkSnapshot that)) {
            return false;
        }
        return capturedAtMs == that.capturedAtMs
                && unknownRequestCount == that.unknownRequestCount
                && requests.equals(that.requests)
                && batches.equals(that.batches);
    }

    @Override
    public int hashCode() {
        return Objects.hash(
                capturedAtMs, requests, batches, unknownRequestCount);
    }

    @Override
    public String toString() {
        return "WorkSnapshot[capturedAtMs=" + capturedAtMs
                + ", requests=" + requests
                + ", batches=" + batches
                + ", unknownRequestCount=" + unknownRequestCount + ']';
    }

    private static long saturatedAdd(long left, long right) {
        return left > Long.MAX_VALUE - right ? Long.MAX_VALUE : left + right;
    }
}
