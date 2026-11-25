package org.flexlb.balance.delivery;

import org.flexlb.service.monitor.BatchSchedulerReporter;

/** Timing of a SINGLE route in its worker queue. Capacity notifications and waits share the queue lock. */
public final class RouteDispatchTiming {
    public enum Stage {
        QUEUE_OR_RESELECTION, DISPATCH_ATTEMPT, PREFILL_RUNNING, PREFILL_INFLIGHT,
        PREFILL_ENGINE_WAITING, PREFILL_OTHER, PREFILL_KV, DECODE_CAPACITY, REOPEN_TO_RESUME
    }

    private static final Stage[] STAGES = Stage.values();
    private final long[] elapsed = new long[STAGES.length];
    private Stage current = Stage.QUEUE_OR_RESELECTION;
    private long since;

    public void enter(Stage next, long nowNanos) {
        if (current == next) { return; }
        if (current != Stage.QUEUE_OR_RESELECTION) {
            elapsed[current.ordinal()] += nowNanos - since;
        }
        current = next;
        since = nowNanos;
    }

    /** Same request population and wall-clock total as routing.queue.wait.time.ms, including zero samples. */
    public void report(BatchSchedulerReporter reporter, String ip, int priority, long totalMs, long nowNanos) {
        enter(Stage.QUEUE_OR_RESELECTION, nowNanos);
        long remaining = totalMs;
        long nanos = 0;
        // Carry fractions across stages because timers record whole milliseconds.
        for (Stage stage : STAGES) {
            if (stage == Stage.QUEUE_OR_RESELECTION) { continue; }
            nanos += elapsed[stage.ordinal()];
            long ms = Math.min(remaining, nanos / 1_000_000L);
            nanos %= 1_000_000L;
            reporter.reportRouteDispatchStage(ip, priority, stage.name(), ms);
            remaining -= ms;
        }
        reporter.reportRouteDispatchStage(ip, priority, Stage.QUEUE_OR_RESELECTION.name(), remaining);
    }
}
