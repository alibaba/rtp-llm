package org.flexlb.config.scheduler;

import lombok.Getter;
import lombok.Setter;
import org.flexlb.config.monitoring.NettyMemoryReporterConfig;

/**
 * Scheduler mode and QUEUE-owned settings.
 */
@Getter
@Setter
public final class SchedulerConfig {
    private Type type = Type.QUEUE;

    /**
     * Opt-in diagnostics; available in both scheduler modes.
     */
    private NettyMemoryReporterConfig nettyMemoryReporter = new NettyMemoryReporterConfig();

    /**
     * Maximum queue entries examined per candidate requested by the planner.
     */
    private double scanBudgetMultiplier = 2.0;

    /**
     * Maximum delay before checking capacity when no notification arrives.
     */
    private long capacityCheckIntervalMs = 100L;

    private QueueOrderingConfig ordering = new QueueOrderingConfig();

    public enum Type {
        DIRECT,
        QUEUE
    }
}
