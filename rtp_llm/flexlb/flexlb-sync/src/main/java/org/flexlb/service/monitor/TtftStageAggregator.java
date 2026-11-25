package org.flexlb.service.monitor;

import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.metric.FlexMonitor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.LongSupplier;

/**
 * Matches one DashClient dispatch to its outer P receipt and finished engine task.
 * Retries have different dispatch IDs, even when their execution intervals overlap.
 */
public final class TtftStageAggregator {
    private final Map<String, Attempt> attempts = new HashMap<>();
    private final Map<String, Long> published = new HashMap<>();
    private final Map<String, Long> invalidated = new HashMap<>();
    private final int capacity;
    private final long retentionMs;
    private final long graceMs;
    private final boolean remoteKvBeforeScheduler;
    private final LongSupplier clock;
    private final TtftStageReporter reporter;
    private final TtftStageQuantiles quantiles;
    private long collectFromUs;

    public TtftStageAggregator(FlexMonitor monitor,
                               int capacity,
                               long retentionMs,
                               long graceMs,
                               boolean remoteKvBeforeScheduler) {
        this(monitor, capacity, retentionMs, graceMs, remoteKvBeforeScheduler, System::currentTimeMillis);
    }

    TtftStageAggregator(FlexMonitor monitor,
                        int capacity,
                        long retentionMs,
                        long graceMs,
                        boolean remoteKvBeforeScheduler,
                        LongSupplier clock) {
        if (capacity < 3 || retentionMs <= graceMs || graceMs < 1000) {
            throw new IllegalArgumentException("TTFT matching requires capacity >= 3 and retention > grace >= 1000ms");
        }
        this.capacity = capacity;
        this.retentionMs = retentionMs;
        this.graceMs = graceMs;
        this.remoteKvBeforeScheduler = remoteKvBeforeScheduler;
        this.clock = clock;
        reporter = new TtftStageReporter(monitor);
        quantiles = new TtftStageQuantiles(reporter, capacity, clock.getAsLong());
    }

    static boolean validTimingField(String value) {
        return value != null && !value.isBlank() && value.length() <= 2048;
    }

    // The collector owns publication. A worker producer can concurrently discard a rejected batch.
    synchronized void addTiming(String source,
                                String id,
                                String role,
                                String deployment,
                                long timestampUs,
                                Long failedAtUs) {
        boolean prefill = "prefill".equals(role);
        if (failedAtUs != null && (failedAtUs < timestampUs || failedAtUs / 1000 > clock.getAsLong() + graceMs)) {
            return;
        }
        if (prefill && !id.equals(TtftRequestIds.dispatchId(id))) {
            // Only the outer P receipt marks the end of dispatch.
            return;
        }
        var attempt = getAttempt(id, timestampUs);
        if (attempt == null) {
            return;
        }
        var receipt = new Timing(source, deployment, timestampUs);
        if (prefill) {
            if (attempt.prefill != null && !attempt.prefill.equals(receipt)) {
                close(id);
            } else if (attempt.prefill == null) {
                attempt.prefill = receipt;
                attempt.changedMs = clock.getAsLong();
            }
        } else if (attempt.dispatch == null) {
            attempt.dispatch = receipt;
            attempt.failedAtUs = failedAtUs;
            attempt.changedMs = clock.getAsLong();
        } else if (!attempt.dispatch.equals(receipt) || !Objects.equals(attempt.failedAtUs, failedAtUs)) {
            close(id);
        }
    }

    public synchronized void addWorkerStatus(String source, WorkerStatus.TaskObservation task) {
        if (task == null || !validTimingField(task.requestId()) || task.telemetry() == null) {
            return;
        }
        long receivedMs = task.telemetry().requestReceivedTimeMs();
        if (receivedMs <= 0 || receivedMs > Long.MAX_VALUE / 1000) {
            return;
        }
        String id = TtftRequestIds.dispatchId(task.requestId());
        var attempt = getAttempt(id, receivedMs * 1000);
        if (attempt == null) {
            return;
        }
        if (attempt.worker != null) {
            if (attempt.worker.requestId().equals(task.requestId())) {
                if (!attempt.worker.equals(task) || !attempt.workerSource.equals(source)) {
                    close(id);
                }
                return;
            }
            // A P proxy may retry internally. Attribute the four measured stages only to the
            // worker which generated the token. Two distinct token producers are ambiguous.
            if (task.telemetry().firstTokenTimeMs() == 0) {
                return;
            }
            if (attempt.worker.telemetry().firstTokenTimeMs() != 0) {
                close(id);
                return;
            }
        }
        attempt.worker = task;
        attempt.workerSource = source;
        attempt.changedMs = clock.getAsLong();
    }

    private Attempt getAttempt(String id, long timestampUs) {
        long now = clock.getAsLong();
        if (timestampUs < collectFromUs || timestampUs / 1000 <= now - retentionMs || timestampUs / 1000 > now + graceMs
                || published.containsKey(id) || invalidated.containsKey(id)) {
            return null;
        }
        var attempt = attempts.get(id);
        if (attempt == null) {
            if (attempts.size() >= capacity) {
                // A later failure event cannot rule out a token from a rejected worker record.
                invalidated.put(id, now);
                return null;
            }
            attempt = new Attempt(now);
            attempts.put(id, attempt);
        }
        return attempt;
    }

    private void close(String id) {
        attempts.remove(id);
        invalidated.put(id, clock.getAsLong());
    }

    public synchronized void clearSource(String source) {
        var affected = new ArrayList<String>();
        attempts.forEach((id, attempt) -> {
            if (source.equals(attempt.workerSource)
                    || attempt.prefill != null && source.equals(attempt.prefill.source())
                    || attempt.dispatch != null && source.equals(attempt.dispatch.source())) {
                affected.add(id);
            }
        });
        affected.forEach(this::close);
    }

    public synchronized void discardWorkerStatus(String source, Iterable<WorkerStatus.TaskObservation> tasks) {
        for (var task : tasks) {
            if (task == null || !validTimingField(task.requestId())) {
                continue;
            }
            String id = TtftRequestIds.dispatchId(task.requestId());
            long receivedMs = task.telemetry() == null ? 0 : task.telemetry().requestReceivedTimeMs();
            if (published.containsKey(id) || receivedMs > 0 && receivedMs < clock.getAsLong() - retentionMs) {
                continue;
            }
            var attempt = attempts.get(id);
            if (attempt != null && task.equals(attempt.worker) && source.equals(attempt.workerSource)) {
                continue;
            }
            close(id);
        }
    }

    public synchronized void resetAll() {
        attempts.clear();
        published.clear();
        invalidated.clear();
        long now = clock.getAsLong();
        collectFromUs = now * 1000;
        quantiles.reset(now);
    }

    public void publishReady() {
        long now = clock.getAsLong();
        List<String> ids;
        synchronized (this) {
            published.values().removeIf(time -> time < now - retentionMs);
            invalidated.values().removeIf(time -> time < now - retentionMs - graceMs);
            ids = new ArrayList<>(attempts.keySet());
        }
        for (String id : ids) {
            Result result = takeResult(id, now);
            if (result == null) {
                continue;
            }
            if (result.failedDurationUs() != null) {
                reporter.reportFailure(result.deployment(), result.failedDurationUs());
            } else {
                reporter.report(result.deployment(), result.stages(), result.pleUs());
                quantiles.add(result.deployment(), id, result.stages(), now);
            }
        }
        quantiles.advance(now);
    }

    private synchronized Result takeResult(String id, long now) {
        var attempt = attempts.get(id);
        if (attempt == null) {
            return null;
        }
        boolean expired = now - attempt.createdMs >= retentionMs;
        if (!expired && now - attempt.changedMs < graceMs) {
            return null;
        }
        Result result = calculate(id, attempt, expired);
        if (result != null && published.size() < capacity) {
            attempts.remove(id);
            published.put(id, now);
        } else {
            if (expired) {
                attempts.remove(id);
            }
            return null;
        }
        return result;
    }

    private Result calculate(String id, Attempt attempt, boolean expired) {
        if (attempt.dispatch == null) {
            return null;
        }
        String deployment = attempt.dispatch.deployment();
        long start = attempt.dispatch.timeUs();
        var timing = attempt.worker == null ? null : attempt.worker.telemetry();
        long tokenMs = timing == null ? 0 : timing.firstTokenTimeMs();
        if (tokenMs < 0 || tokenMs > Long.MAX_VALUE / 1000) {
            return null;
        }
        long tokenUs = tokenMs * 1000;
        if (attempt.failedAtUs != null && (tokenUs == 0 || tokenUs > attempt.failedAtUs)) {
            // A client timeout can race a finished WorkerStatus which already contains a token.
            // If the worker is missing, allow the whole retention window for that observation.
            boolean directWorker = timing != null && attempt.worker.requestId().equals(id);
            if (directWorker || expired) {
                return new Result(deployment, null, null, attempt.failedAtUs - start);
            }
            return null;
        }
        if (tokenUs == 0 || attempt.prefill == null) {
            return null;
        }
        long engineStart = timing.requestReceivedTimeMs() * 1000;
        long receipt = attempt.prefill.timeUs();
        if (tokenUs < engineStart || receipt - engineStart > 999 || start - tokenUs > 999) {
            return null;
        }
        long total = Math.max(0, tokenUs - start);
        long dispatch = Math.min(total, Math.max(0, receipt - start));
        long[] stages = stages(timing, total, dispatch, tokenUs - engineStart);
        Long ple = timing.schedulerWaitForPleReadyUs();
        if (ple != null && (ple < 0 || ple - 999 > tokenUs - engineStart)) {
            ple = null;
        }
        return stages == null && ple == null ? null : new Result(deployment, stages, ple, null);
    }

    private long[] stages(WorkerStatus.TaskTelemetry timing, long total, long dispatch, long engineDuration) {
        if (timing.hybridKvAllocWaitUs() == null && timing.schedulerWaitUs() == null
                && timing.prefillInflightWallUs() == null && timing.engineAddToFirstTokenUs() == null
                || timing.remoteKvWaitMs() < 0 || timing.remoteKvWaitMs() > Long.MAX_VALUE / 1000) {
            return null;
        }
        long[] values = {total, dispatch, orZero(timing.hybridKvAllocWaitUs()),
                remoteKvBeforeScheduler ? timing.remoteKvWaitMs() * 1000 : 0, orZero(timing.schedulerWaitUs()),
                orZero(timing.prefillInflightWallUs()), total - dispatch};
        for (int i = 2; i <= 5; i++) {
            if (values[i] < 0 || values[i] > values[6]) {
                return null;
            }
            values[6] -= values[i];
        }
        return total - dispatch - values[6] - 999 > engineDuration ? null : values;
    }

    private static long orZero(Long value) {
        return value == null ? 0 : value;
    }

    private static final class Attempt {
        private final long createdMs;
        private long changedMs;
        private Timing dispatch;
        private Timing prefill;
        private Long failedAtUs;
        private String workerSource;
        private WorkerStatus.TaskObservation worker;

        private Attempt(long now) {
            createdMs = now;
            changedMs = now;
        }
    }

    private record Timing(String source, String deployment, long timeUs) {
    }

    private record Result(String deployment, long[] stages, Long pleUs, Long failedDurationUs) {
    }
}
