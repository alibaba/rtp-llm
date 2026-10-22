package org.flexlb.dao.master;

import java.util.IdentityHashMap;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Per-generation diagnostic state. Only the newest STATUS and CACHE polls are retained.
 */
public final class EndpointTrace {
    private static final int MAX_ERROR_MESSAGE_CHARS = 192;
    private static final int MAX_ERROR_TYPE_CHARS = 128;
    private static final int MAX_CAUSE_DEPTH = 16;
    private static final int MAX_SUPPRESSED_ERRORS = 4;
    private static final int MAX_REASON_CHARS = 128;

    public enum Stage {
        CACHE_CLEANUP_BEGIN(true),
        CACHE_CLEANUP_END(true),
        CACHE_CLEANUP_FAILED(true),
        CACHE_INDEX_LOCK_ACQUIRED(true),
        CACHE_INDEX_LOCK_ATTEMPT(true),
        CACHE_INDEX_LOCK_RELEASED(true),
        CACHE_INDEX_UPDATE_BEGIN,
        CACHE_INDEX_UPDATE_END,
        CACHE_INDEX_UPDATE_FAILED,
        CACHE_MUTATION_LOCK_ACQUIRED,
        CACHE_MUTATION_LOCK_ATTEMPT,
        CACHE_MUTATION_LOCK_RELEASED,
        CACHE_RESPONSE_FAILED,
        CACHE_VERSION_UNCHANGED,
        CALLBACK_ENDED,
        CALLBACK_FAILED,
        CALLBACK_READY,
        CALLBACK_STARTED,
        CALLBACK_SUBMISSION_FAILED,
        CONVERSION_STARTED,
        CONVERSION_COMPLETED,
        CONVERSION_FAILED,
        DISCOVERY_POLL_SKIPPED,
        ENDPOINT_AWAIT_BEGIN(true),
        ENDPOINT_AWAIT_END(true),
        ENDPOINT_AWAIT_FAILED(true),
        ENDPOINT_GATE_CLOSED(true),
        ENDPOINT_PUBLISHED(true),
        ENDPOINT_WITHDRAWN(true),
        RETIREMENT_CHECK_LOCK_ACQUIRED,
        RETIREMENT_CHECK_LOCK_ATTEMPT,
        RETIREMENT_CHECK_LOCK_RELEASED,
        FINALIZE_FAILED(true),
        GENERATION_RETIRING(true),
        LEASE_RELEASED,
        RESPONSE_DISCARDED,
        RETIREMENT_CLEANUP_BEGIN(true),
        RETIREMENT_CLEANUP_END(true),
        RETIREMENT_CACHE_LOCK_ACQUIRED(true),
        RETIREMENT_CACHE_LOCK_ATTEMPT(true),
        RETIREMENT_CACHE_LOCK_RELEASED(true),
        RPC_COMPLETED,
        RPC_INITIATION_FAILED,
        RPC_STARTED,
        RUNNER_FAILED,
        RUNNER_REJECTED,
        RUNNER_STARTED,
        RUNNER_SUBMITTED,
        STATUS_APPLIED,
        STATUS_COMMITTED,
        STATUS_HEALTH_UPDATED,
        STATUS_LOCK_ACQUIRED,
        STATUS_LOCK_ATTEMPT,
        STATUS_LOCK_RELEASED,
        STATUS_PROJECTION_FAILED,
        STATUS_RECONCILIATION_FAILED,
        STATUS_REMOVED(true),
        STATUS_RESPONSE_FAILED,
        UNAVAILABLE_DECISION(true);

        private final boolean retirement;

        Stage() {
            this(false);
        }

        Stage(boolean retirement) {
            this.retirement = retirement;
        }
    }

    public record PollTimes(
            long rpcStartMs,
            long rpcResultMs,
            long convertStartMs,
            long convertEndMs,
            long callbackReadyMs,
            long callbackStartMs,
            long workerAttemptMs,
            long workerAcquiredMs,
            long healthMs,
            long commitMs,
            long workerReleasedMs,
            long leaseEndMs,
            long callbackEndMs,
            long runnerSubmitMs,
            long runnerStartMs,
            long cacheAttemptMs,
            long cacheAcquiredMs,
            long cacheReleasedMs) {
        private void append(StringBuilder text) {
            text.append("{rpc:");
            value(text, rpcStartMs);
            text.append('/');
            value(text, rpcResultMs);
            text.append(",conv:");
            value(text, convertStartMs);
            text.append('/');
            value(text, convertEndMs);
            text.append(",cb:");
            value(text, callbackReadyMs);
            text.append('/');
            value(text, callbackStartMs);
            text.append('/');
            value(text, callbackEndMs);
            text.append(",w:");
            value(text, workerAttemptMs);
            text.append('/');
            value(text, workerAcquiredMs);
            text.append('/');
            value(text, workerReleasedMs);
            text.append(",health:");
            value(text, healthMs);
            text.append(",commit:");
            value(text, commitMs);
            text.append(",lease:");
            value(text, leaseEndMs);
            text.append(",runner:");
            value(text, runnerSubmitMs);
            text.append('/');
            value(text, runnerStartMs);
            text.append(",cacheLock:");
            value(text, cacheAttemptMs);
            text.append('/');
            value(text, cacheAcquiredMs);
            text.append('/');
            value(text, cacheReleasedMs);
            text.append('}');
        }

        private static void value(StringBuilder text, long value) {
            if (value < 0L) {
                text.append('-');
            } else {
                text.append(value);
            }
        }
    }

    public record IndexWait(long ageUs, long totalWaitUs, long contendedHoldUs, long count) {}

    public record PollSnapshot(
            long sequence,
            long syncCycleId,
            String kind,
            long startedAtMs,
            String stage,
            boolean leaseReleased,
            String rpcResult,
            String failureStage,
            String error,
            PollTimes times,
            IndexWait indexWait,
            long skippedBusy,
            long rpcTimeoutMs,
            String outcome) {
        public String pollId() {
            return kind + "-" + sequence;
        }
    }

    public record Snapshot(
            long capturedAtMs,
            boolean retiring,
            String phase,
            String unavailableReason,
            long decidedAtMs,
            long lastSuccessAgeUs,
            long consecutiveFailures,
            long gateClosedAtMs,
            long cleanupWaitUs,
            long cleanupHoldUs,
            long scannedKeys,
            long removedMappings,
            String error,
            PollSnapshot statusPoll,
            PollSnapshot cachePoll) {
        public String compact() {
            var text = new StringBuilder().append("phase=").append(phase);
            if (retiring) {
                text.append(" reason=")
                        .append(unavailableReason)
                        .append(" decisionMs=")
                        .append(decidedAtMs)
                        .append(" lastSuccessAgeUs=")
                        .append(lastSuccessAgeUs)
                        .append(" failures=")
                        .append(consecutiveFailures)
                        .append(" gateMs=")
                        .append(gateClosedAtMs)
                        .append(" cleanupWaitUs=")
                        .append(cleanupWaitUs)
                        .append(" cleanupHoldUs=")
                        .append(cleanupHoldUs)
                        .append(" scanned=")
                        .append(scannedKeys)
                        .append(" removed=")
                        .append(removedMappings);
            }
            if (error != null) {
                text.append(" error=").append(error);
            }
            appendPoll(text, "statusPoll", statusPoll);
            appendPoll(text, "cachePoll", cachePoll);
            return text.toString();
        }

        private static void appendPoll(StringBuilder text, String label, PollSnapshot poll) {
            if (poll != null) {
                text.append(' ')
                        .append(label)
                        .append("={id:")
                        .append(poll.pollId())
                        .append(",cycle:")
                        .append(poll.syncCycleId())
                        .append(",startMs:")
                        .append(poll.startedAtMs())
                        .append(",stage:")
                        .append(poll.stage())
                        .append(",rpc:")
                        .append(poll.rpcResult())
                        .append(",rpcTimeoutMs:")
                        .append(poll.rpcTimeoutMs())
                        .append(",released:")
                        .append(poll.leaseReleased())
                        .append(",busySkips:")
                        .append(poll.skippedBusy());
                if (poll.error() != null) {
                    text.append(",failure:")
                            .append(poll.failureStage())
                            .append(",error:")
                            .append(poll.error());
                }
                if (poll.outcome() != null) {
                    text.append(",outcome:").append(poll.outcome());
                }
                var wait = poll.indexWait();
                if (wait.count() != 0) {
                    text.append(",gWaitUs:")
                            .append(wait.ageUs())
                            .append(",gTotalUs:")
                            .append(wait.totalWaitUs())
                            .append(",gHoldUs:")
                            .append(wait.contendedHoldUs());
                }
                text.append(",tMs:");
                poll.times().append(text);
                text.append('}');
            }
        }
    }

    private volatile Poll latestStatusPoll;
    private volatile Poll latestCachePoll;
    // WorkerStatus serializes poll creation with its health monitor.
    private long nextPollId;
    private final AtomicBoolean endpointPublished = new AtomicBoolean();
    private final AtomicLong gateClosedAtMs = new AtomicLong();
    private final AtomicReference<String> error = new AtomicReference<>();

    private volatile String phase = "DISCOVERED";
    private volatile String retirementPhase = "UNKNOWN";
    private volatile boolean retiring;
    private volatile String unavailableReason = "UNKNOWN";
    private volatile long decidedAtMs;
    private volatile long lastSuccessAgeUs = -1L;
    private volatile long consecutiveFailures = -1L;
    private volatile long cleanupAttemptNanos;
    private volatile long cleanupAcquiredNanos;
    private volatile long cleanupWaitUs = -1L;
    private volatile long cleanupHoldUs = -1L;
    private volatile long scannedKeys;
    private volatile long removedMappings;

    public void record(Stage stage) {
        if (stage == Stage.ENDPOINT_PUBLISHED && !endpointPublished.compareAndSet(false, true)) {
            return;
        }
        if (stage == Stage.ENDPOINT_GATE_CLOSED
                && !gateClosedAtMs.compareAndSet(0L, System.currentTimeMillis())) {
            return;
        }
        switch (stage) {
            case GENERATION_RETIRING -> retiring = true;
            case ENDPOINT_WITHDRAWN -> gateClosedAtMs.compareAndSet(0L, System.currentTimeMillis());
            case CACHE_INDEX_LOCK_ATTEMPT -> cleanupAttemptNanos = System.nanoTime();
            case CACHE_INDEX_LOCK_ACQUIRED -> {
                long now = System.nanoTime();
                long started = cleanupAttemptNanos;
                // Close the active interval before publishing its total to avoid double counting in
                // snapshots.
                cleanupAttemptNanos = 0L;
                cleanupWaitUs = Math.max(0L, cleanupWaitUs) + elapsedUs(started, now);
                cleanupAcquiredNanos = now;
            }
            case CACHE_INDEX_LOCK_RELEASED -> {
                long now = System.nanoTime();
                long started = cleanupAcquiredNanos;
                cleanupAcquiredNanos = 0L;
                cleanupHoldUs = Math.max(0L, cleanupHoldUs) + elapsedUs(started, now);
            }
            default -> {}
        }
        phase = stage.name();
        if (stage.retirement) {
            retirementPhase = stage.name();
        }
    }

    public void record(Stage stage, Throwable failure) {
        record(stage);
        if (failure != null && error.get() == null) {
            error.compareAndSet(null, errorSummary(failure));
        }
    }

    public void record(Stage stage, String reason) {
        error.set(bounded(reason, MAX_ERROR_MESSAGE_CHARS));
        record(stage);
    }

    public void recordUnavailable(String reason, long lastSuccessUs, long failures) {
        unavailableReason = bounded(reason, MAX_REASON_CHARS);
        decidedAtMs = System.currentTimeMillis();
        lastSuccessAgeUs =
                lastSuccessUs == 0L
                        ? -1L
                        : TimeUnit.NANOSECONDS.toMicros(System.nanoTime()) - lastSuccessUs;
        consecutiveFailures = failures;
        record(Stage.UNAVAILABLE_DECISION);
    }

    public String unavailableReason() {
        return unavailableReason;
    }

    public void recordCleanupCounts(long scanned, long removed) {
        scannedKeys = scanned;
        removedMappings = removed;
    }

    /** WorkerStatus holds its health monitor so only an admitted poll replaces the latest snapshot. */
    public Poll beginPoll(String kind, long cycle) {
        Poll pollTrace = new Poll(kind, ++nextPollId, cycle);
        switch (kind) {
            case "STATUS" -> latestStatusPoll = pollTrace;
            case "CACHE" -> latestCachePoll = pollTrace;
            default -> throw new IllegalArgumentException("Unknown poll kind: " + kind);
        }
        return pollTrace;
    }

    public static final class Poll {
        private final long id;
        private final long syncCycleId;
        private final String kind;
        private final long startedAtMs = System.currentTimeMillis();
        private final long startedNanos = System.nanoTime();
        private final AtomicLong skippedBusy = new AtomicLong();
        private final AtomicReference<Stage> firstFailureStage = new AtomicReference<>();
        private volatile String error = "DETAILS_PENDING";
        private volatile String stage = "LEASE_ACQUIRED";
        private volatile String rpcResult = "NOT_STARTED";
        private volatile String outcome;
        private volatile long rpcTimeoutMs;

        private volatile long runnerSubmittedNanos;
        private volatile long runnerStartedNanos;
        private volatile long rpcStartedNanos;
        private volatile long rpcCompletedNanos;
        private volatile long conversionStartedNanos;
        private volatile long conversionEndedNanos;
        private volatile long callbackReadyNanos;
        private volatile long callbackStartedNanos;
        private volatile long callbackEndedNanos;
        private volatile long workerLockAttemptNanos;
        private volatile long workerLockAcquiredNanos;
        private volatile long workerLockReleasedNanos;
        private volatile long healthUpdatedNanos;
        private volatile long statusCommittedNanos;
        private volatile long leaseReleasedNanos;
        private volatile long cacheLockAttemptNanos;
        private volatile long cacheLockAcquiredNanos;
        private volatile long cacheLockReleasedNanos;

        // These counters have one writer: this poll's cache callback.
        private volatile long indexWaitStartedNanos;
        private volatile long indexAcquiredNanos;
        private volatile long indexTotalWaitUs;
        private volatile long indexContendedHoldUs;
        private volatile long indexCount;

        private Poll(String kind, long id, long cycle) {
            this.kind = kind;
            this.id = id;
            this.syncCycleId = cycle;
        }

        public void record(Stage point) {
            record(point, (Throwable) null);
        }

        public Stage failureStage() {
            return firstFailureStage.get();
        }

        public boolean hasCallbackStarted() {
            return callbackStartedNanos != 0L;
        }

        public void record(Stage point, Throwable failure) {
            long now = System.nanoTime();
            switch (point) {
                case RUNNER_SUBMITTED -> runnerSubmittedNanos = now;
                case RUNNER_STARTED -> runnerStartedNanos = now;
                case RPC_STARTED -> {
                    rpcStartedNanos = now;
                    rpcResult = "PENDING";
                }
                case RPC_COMPLETED, RPC_INITIATION_FAILED -> {
                    rpcResult = failure == null ? "OK" : "ERROR";
                    rpcCompletedNanos = now;
                }
                case CONVERSION_STARTED -> conversionStartedNanos = now;
                case CONVERSION_COMPLETED, CONVERSION_FAILED -> conversionEndedNanos = now;
                case CALLBACK_READY -> callbackReadyNanos = now;
                case CALLBACK_STARTED -> callbackStartedNanos = now;
                case CALLBACK_ENDED -> callbackEndedNanos = now;
                case STATUS_LOCK_ATTEMPT -> workerLockAttemptNanos = now;
                case STATUS_LOCK_ACQUIRED -> workerLockAcquiredNanos = now;
                case STATUS_LOCK_RELEASED -> workerLockReleasedNanos = now;
                case STATUS_HEALTH_UPDATED -> healthUpdatedNanos = now;
                case STATUS_COMMITTED, CACHE_INDEX_UPDATE_END -> statusCommittedNanos = now;
                case LEASE_RELEASED -> leaseReleasedNanos = now;
                case CACHE_MUTATION_LOCK_ATTEMPT -> cacheLockAttemptNanos = now;
                case CACHE_MUTATION_LOCK_ACQUIRED -> cacheLockAcquiredNanos = now;
                case CACHE_MUTATION_LOCK_RELEASED -> cacheLockReleasedNanos = now;
                case CACHE_INDEX_LOCK_ATTEMPT -> {
                    indexWaitStartedNanos = now;
                    indexCount++;
                }
                case CACHE_INDEX_LOCK_ACQUIRED -> {
                    indexAcquiredNanos = now;
                    if (indexWaitStartedNanos != 0L) {
                        indexTotalWaitUs += elapsedUs(indexWaitStartedNanos, now);
                    }
                    indexWaitStartedNanos = 0L;
                }
                case CACHE_INDEX_LOCK_RELEASED -> {
                    if (indexAcquiredNanos != 0L) {
                        indexContendedHoldUs += elapsedUs(indexAcquiredNanos, now);
                    }
                }
                default -> {}
            }
            stage = point.name();
            if (failure != null && firstFailureStage.compareAndSet(null, point)) {
                error = errorSummary(failure);
            }
        }

        public void record(Stage point, String reason) {
            outcome = point.name() + ":" + bounded(reason, MAX_REASON_CHARS);
            record(point);
        }

        public void recordRpcStarted(long timeoutMs) {
            rpcTimeoutMs = timeoutMs;
            record(Stage.RPC_STARTED);
        }

        public void skippedBusy() {
            skippedBusy.incrementAndGet();
        }

        private PollSnapshot snapshot(long now) {
            Stage failureStage = firstFailureStage.get();
            long waitStarted = indexWaitStartedNanos;
            PollTimes times =
                    new PollTimes(
                            sinceStartMs(rpcStartedNanos),
                            sinceStartMs(rpcCompletedNanos),
                            sinceStartMs(conversionStartedNanos),
                            sinceStartMs(conversionEndedNanos),
                            sinceStartMs(callbackReadyNanos),
                            sinceStartMs(callbackStartedNanos),
                            sinceStartMs(workerLockAttemptNanos),
                            sinceStartMs(workerLockAcquiredNanos),
                            sinceStartMs(healthUpdatedNanos),
                            sinceStartMs(statusCommittedNanos),
                            sinceStartMs(workerLockReleasedNanos),
                            sinceStartMs(leaseReleasedNanos),
                            sinceStartMs(callbackEndedNanos),
                            sinceStartMs(runnerSubmittedNanos),
                            sinceStartMs(runnerStartedNanos),
                            sinceStartMs(cacheLockAttemptNanos),
                            sinceStartMs(cacheLockAcquiredNanos),
                            sinceStartMs(cacheLockReleasedNanos));
            IndexWait wait =
                    new IndexWait(
                            waitStarted == 0L ? 0L : elapsedUs(waitStarted, now),
                            indexTotalWaitUs,
                            indexContendedHoldUs,
                            indexCount);
            return new PollSnapshot(
                    id,
                    syncCycleId,
                    kind,
                    startedAtMs,
                    stage,
                    times.leaseEndMs() >= 0L,
                    rpcResult,
                    failureStage == null ? null : failureStage.name(),
                    failureStage == null ? null : error,
                    times,
                    wait,
                    skippedBusy.get(),
                    rpcTimeoutMs,
                    outcome);
        }

        private long sinceStartMs(long atNanos) {
            return atNanos == 0L
                    ? -1L
                    : TimeUnit.NANOSECONDS.toMillis(Math.max(0L, atNanos - startedNanos));
        }
    }

    /** Captures both poll kinds without locks; concurrent fields can describe adjacent instants. */
    public Snapshot snapshot() {
        Poll statusPoll = latestStatusPoll;
        Poll cachePoll = latestCachePoll;
        boolean isRetiring = retiring;
        long now = System.nanoTime();
        // Read totals before active starts, matching record's clear-before-accumulate order.
        return new Snapshot(
                System.currentTimeMillis(),
                isRetiring,
                isRetiring ? retirementPhase : phase,
                unavailableReason,
                decidedAtMs,
                lastSuccessAgeUs,
                consecutiveFailures,
                gateClosedAtMs.get(),
                ongoingTotalUs(cleanupWaitUs, cleanupAttemptNanos, now),
                ongoingTotalUs(cleanupHoldUs, cleanupAcquiredNanos, now),
                scannedKeys,
                removedMappings,
                error.get(),
                statusPoll == null ? null : statusPoll.snapshot(now),
                cachePoll == null ? null : cachePoll.snapshot(now));
    }

    private static long ongoingTotalUs(long completedUs, long startNanos, long now) {
        return startNanos == 0L
                ? completedUs
                : Math.max(0L, completedUs) + elapsedUs(startNanos, now);
    }

    private static long elapsedUs(long startNanos, long endNanos) {
        return startNanos == 0L
                ? -1L
                : TimeUnit.NANOSECONDS.toMicros(Math.max(0L, endNanos - startNanos));
    }

    private static String errorSummary(Throwable failure) {
        if (failure == null) {
            return null;
        }
        var text = new StringBuilder();
        try {
            Throwable unwrapped = unwrap(failure);
            // Preserve types before the bounded message: an outer description must not hide the
            // root cause.
            appendCauseTypes(text, unwrapped);
            appendSuppressedTypes(text, failure);
            String message = unwrapped.getMessage();
            if (message != null) {
                text.append(':').append(bounded(message, MAX_ERROR_MESSAGE_CHARS));
            }
        } catch (RuntimeException | LinkageError ignored) {
            if (text.isEmpty()) {
                text.append(bounded(failure.getClass().getName(), MAX_ERROR_TYPE_CHARS));
            }
        }
        return text.toString();
    }

    private static void appendCauseTypes(StringBuilder text, Throwable failure) {
        var seen = new IdentityHashMap<Throwable, Boolean>();
        int depth = 0;
        while (failure != null && depth < MAX_CAUSE_DEPTH) {
            if (depth != 0) {
                text.append("->");
            }
            if (seen.put(failure, Boolean.TRUE) != null) {
                text.append("[cycle]");
                return;
            }
            String type = failure.getClass().getSimpleName();
            text.append(
                    bounded(
                            type.isEmpty() ? failure.getClass().getName() : type,
                            MAX_ERROR_TYPE_CHARS));
            failure = failure.getCause();
            depth++;
        }
        if (failure != null) {
            text.append("->[truncated]");
        }
    }

    private static void appendSuppressedTypes(StringBuilder text, Throwable failure) {
        var seen = new IdentityHashMap<Throwable, Boolean>();
        int count = 0;
        for (int depth = 0;
                failure != null
                        && depth < MAX_CAUSE_DEPTH
                        && seen.put(failure, Boolean.TRUE) == null;
                depth++, failure = failure.getCause()) {
            for (Throwable suppressed : failure.getSuppressed()) {
                // Suppressed attempts are diagnostic history, not the final failure's causal chain.
                if (count == 0) {
                    text.append(" suppressed=[");
                } else {
                    text.append(';');
                }
                if (count == MAX_SUPPRESSED_ERRORS) {
                    text.append("[truncated]]");
                    return;
                }
                appendCauseTypes(text, suppressed);
                count++;
            }
        }
        if (count != 0) {
            text.append(']');
        }
    }

    public static Throwable unwrap(Throwable failure) {
        if (!(failure instanceof CompletionException || failure instanceof ExecutionException)) {
            return failure;
        }
        var seen = new IdentityHashMap<Throwable, Boolean>();
        for (int depth = 0;
                depth < MAX_CAUSE_DEPTH
                        && (failure instanceof CompletionException
                                || failure instanceof ExecutionException)
                        && seen.put(failure, Boolean.TRUE) == null;
                depth++) {
            Throwable cause = failure.getCause();
            if (cause == null) {
                break;
            }
            failure = cause;
        }
        return failure;
    }

    private static String bounded(String text, int limit) {
        return text == null
                ? "UNKNOWN"
                : text.substring(0, Math.min(limit, text.length()))
                        .replace('\n', ' ')
                        .replace('\r', ' ');
    }
}
