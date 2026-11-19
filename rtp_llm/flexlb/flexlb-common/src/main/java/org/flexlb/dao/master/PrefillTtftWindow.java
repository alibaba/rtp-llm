package org.flexlb.dao.master;

import java.util.HashMap;
import java.util.Map;

/**
 * Bounded, generation-local request TTFT observations from committed worker status.
 */
public final class PrefillTtftWindow {
    public static final long RETENTION_MS = 600_000;
    private static final int MAX_SAMPLES = 100_000;
    private final Map<String, Sample> samples = new HashMap<>();
    private long startedMs;
    private long latestStatusMs;
    private long invalidThroughMs;
    private long lastCleanupMs;
    private Long latestFinishedVersion;
    private boolean finishedCursorRegressed;

    private record Sample(long receivedMs, long firstTokenMs) {
    }

    public record Measurement(double totalMs, long count, long latestStatusMs) {
    }

    /**
     * Called only after a strictly newer worker status has committed.
     */
    public synchronized void collect(WorkerStatus.StatusObservation observation, long nowMs) {
        if (startedMs == 0) {
            startedMs = nowMs;
        }
        Long finishedVersion = observation.latestFinishedVersion();
        if (latestFinishedVersion != null && finishedVersion != null) {
            if (finishedVersion < latestFinishedVersion) {
                finishedCursorRegressed = true;
            } else if (finishedVersion - latestFinishedVersion > observation.finishedTasks().size()) {
                invalidThroughMs = nowMs;
            }
        }
        if (finishedVersion == null || nowMs < latestStatusMs) {
            invalidThroughMs = nowMs;
        }
        latestFinishedVersion = finishedVersion;
        latestStatusMs = nowMs;
        if (nowMs - lastCleanupMs >= 1000) {
            samples.values().removeIf(sample -> sample.firstTokenMs() <= nowMs - RETENTION_MS);
            lastCleanupMs = nowMs;
        }
        for (var task : observation.activeTasks().values()) {
            add(task, nowMs);
        }
        for (var task : observation.finishedTasks().values()) {
            add(task, nowMs);
        }
    }

    private void add(WorkerStatus.TaskObservation task, long nowMs) {
        var telemetry = task.telemetry();
        if (telemetry == null || telemetry.firstTokenTimeMs() <= 0) {
            return;
        }
        long first = telemetry.firstTokenTimeMs();
        long received = telemetry.requestReceivedTimeMs();
        if (received <= 0 || first < received || first > nowMs + 1000) {
            invalidThroughMs = nowMs;
            return;
        }
        if (first <= nowMs - RETENTION_MS || samples.containsKey(task.requestId())) {
            return;
        }
        if (samples.size() >= MAX_SAMPLES) {
            invalidThroughMs = nowMs;
            return;
        }
        samples.put(task.requestId(), new Sample(received, first));
    }

    public synchronized Measurement read(long startMs, long nowMs, long maxStatusAgeMs, long minRequestReceivedMs) {
        if (finishedCursorRegressed || startMs < startedMs || latestStatusMs == 0 || latestStatusMs > nowMs
                || nowMs - latestStatusMs > maxStatusAgeMs || invalidThroughMs >= startMs
                || nowMs - startMs > RETENTION_MS) {
            throw new IllegalStateException("Incomplete or stale worker TTFT window");
        }
        double total = 0;
        long count = 0;
        for (Sample sample : samples.values()) {
            if (sample.receivedMs() >= minRequestReceivedMs && sample.firstTokenMs() >= startMs
                    && sample.firstTokenMs() <= nowMs) {
                total += sample.firstTokenMs() - sample.receivedMs();
                count++;
            }
        }
        return new Measurement(total, count, latestStatusMs);
    }
}
