package org.flexlb.balance.projection;

import java.util.Arrays;
import java.util.List;
import java.util.OptionalLong;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * Frozen request identities and Prefill work accounting after endpoint admission.
 * Each running request or batch consumes elapsed time independently; a batch is counted once.
 * Unknown work preserves member identities and known durations, but makes the total unavailable.
 */
public final class WorkSnapshot {

    private static final long[] EMPTY_LONGS = new long[0];

    private final long capturedAtMs;
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
        requests = List.copyOf(requests);
        batches = List.copyOf(batches);
        checkArgument(unknownRequestCount >= 0L, "unknownRequestCount must be non-negative");

        int runningCount = 0;
        int requestIdCount = requests.size();
        long nonRunningMs = 0L;
        boolean hasUnknown = unknownRequestCount > 0L;
        for (RequestWork request : requests) {
            if (request.phase() == Phase.ENGINE_RUNNING) {
                runningCount++;
            } else {
                nonRunningMs = saturatedAdd(
                        nonRunningMs, request.remainingWorkMs());
            }
        }
        for (BatchWork batch : batches) {
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
        for (RequestWork request : requests) {
            requestIds[requestIdIndex++] = request.requestId();
            if (request.phase() == Phase.ENGINE_RUNNING) {
                runningWorkMs[runningIndex++] = request.remainingWorkMs();
            }
        }
        for (BatchWork batch : batches) {
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
            checkArgument(remainingWorkMs >= 0L, "remaining request work must be non-negative");
        }
    }

    /** One shared EnqueueBatch work unit and its live request identities. */
    public record BatchWork(List<Long> requestIds,
                            Phase phase,
                            OptionalLong remainingWorkMs) {

        public BatchWork {
            requestIds = List.copyOf(requestIds);
            checkArgument(!remainingWorkMs.isPresent() || remainingWorkMs.getAsLong() >= 0L,
                    "remaining batch work must be non-negative");
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
        return totalRemainingWorkMsAt(capturedAtMs);
    }

    /** Complete preceding work at a later clock; only observed running work consumes time. */
    public OptionalLong totalRemainingWorkMsAt(long observedAtMs) {
        return hasUnknownWork()
                ? OptionalLong.empty()
                : OptionalLong.of(knownRemainingWorkMsAt(observedAtMs));
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

    private static long saturatedAdd(long left, long right) {
        return left > Long.MAX_VALUE - right ? Long.MAX_VALUE : left + right;
    }
}
