package org.flexlb.util.metrics;

/**
 * Executor counters without permission to submit or stop work.
 */
public interface ExecutorStats {
    int activeThreads();

    int queuedTasks();

    int poolSize();

    long completedTasks();
}
