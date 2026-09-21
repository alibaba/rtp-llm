package org.flexlb.mockengine;

import java.util.LinkedHashMap;
import java.util.Map;

/** Batch-atomic equivalent of RtpLLMTokenPSMetricsCollector (microseconds). */
final class PrefillTpsMetrics {
    record Batch(long generation, long startedNanos, long computeTokens, long inputTokens) {}
    record Snapshot(long generation, long startedNanos, int active,
                    long computeTokens, long inputTokens, long computeUs, long inputUs) {}

    private long generation;
    private long startedNanos;
    private int active;
    private long computeTokens, inputTokens, computeUs, inputUs;

    PrefillTpsMetrics() { this(System.nanoTime()); }
    PrefillTpsMetrics(long now) { startedNanos = now; }

    synchronized Batch begin(long compute, long input, long now) {
        active++;
        return new Batch(generation, now, compute, input);
    }

    synchronized void finish(Batch batch, long now) {
        if (batch == null || batch.generation() != generation) return;
        active--;
        long durationUs = (now - batch.startedNanos()) / 1000;
        // Each nonempty numerator owns ONE duration per executed batch. Real
        // addTokenSize excludes zero-token batches from that numerator's clock.
        if (durationUs <= 0) return;
        if (batch.computeTokens() > 0) {
            computeTokens += batch.computeTokens();
            computeUs += durationUs;
        }
        if (batch.inputTokens() > 0) {
            inputTokens += batch.inputTokens();
            inputUs += durationUs;
        }
    }

    synchronized Snapshot snapshot() {
        return new Snapshot(generation, startedNanos, active, computeTokens, inputTokens, computeUs, inputUs);
    }

    synchronized void reset(long now) {
        generation++;
        startedNanos = now;
        active = 0;
        computeTokens = inputTokens = computeUs = inputUs = 0;
    }

    /** Independent HTTP/KMonitor cursors: neither observer drains the other. */
    static final class Reader {
        private Snapshot previous;
        private long reportedAt;
        private Map<String, Double> last = Map.of();

        synchronized Map<String, Double> last() { return last; }

        synchronized Map<String, Double> sample(Snapshot current, long now) {
            if (previous == null || previous.generation() != current.generation()) {
                previous = new Snapshot(current.generation(), current.startedNanos(), 0, 0, 0, 0, 0);
                reportedAt = current.startedNanos();
            }
            long compute = current.computeTokens() - previous.computeTokens();
            long input = current.inputTokens() - previous.inputTokens();
            long computeTime = current.computeUs() - previous.computeUs();
            long inputTime = current.inputUs() - previous.inputUs();
            // Do not invent zero samples while a long step is in flight. Keep
            // both the last completed gauge and wall-clock origin until it completes.
            if (computeTime == 0 && inputTime == 0 && current.active() > 0) {
                return last;
            }
            long wallUs = Math.max(1, (now - reportedAt) / 1000);
            Map<String, Double> values = new LinkedHashMap<>();
            boolean idle = computeTime == 0 && inputTime == 0;
            if (computeTime > 0 || idle)
                values.put("context_tps", rate(compute, computeTime));
            if (inputTime > 0 || idle)
                values.put("context_tps_with_cache", rate(input, inputTime));
            values.put("context_wall_tps", rate(compute, wallUs));
            values.put("context_wall_tps_with_cache", rate(input, wallUs));
            values.put("wall_tps_report_interval_us", (double) wallUs);
            previous = current;
            reportedAt = now;
            last = Map.copyOf(values);
            return last;
        }

        private static double rate(long tokens, long us) {
            return us > 0 ? tokens * 1_000_000.0 / us : 0.0;
        }
    }
}
