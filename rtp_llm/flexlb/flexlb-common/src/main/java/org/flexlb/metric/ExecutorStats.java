package org.flexlb.metric;

/** Executor counters without permission to submit or stop work. */
public interface ExecutorStats {
    int activeThreads();

    int queuedTasks();

    int poolSize();

    long completedTasks();
}
