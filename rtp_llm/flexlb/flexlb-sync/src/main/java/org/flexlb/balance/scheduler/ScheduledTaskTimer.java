package org.flexlb.balance.scheduler;

import java.util.Objects;
import java.util.concurrent.Future;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/** Owns a timer pool; callers own the meaning and cancellation of its scheduled tasks. */
public final class ScheduledTaskTimer implements TaskTimer, AutoCloseable {
    private final ScheduledThreadPoolExecutor executor;

    ScheduledTaskTimer() {
        this(2, "flexlb-request-cleanup-");
    }

    ScheduledTaskTimer(int threads, String threadName) {
        executor =
                new ScheduledThreadPoolExecutor(
                        threads, Thread.ofPlatform().daemon().name(threadName, 1).factory());
        executor.setRemoveOnCancelPolicy(true);
        executor.setExecuteExistingDelayedTasksAfterShutdownPolicy(false);
        executor.setContinueExistingPeriodicTasksAfterShutdownPolicy(false);
    }

    @Override
    public Future<?> schedule(Runnable task, long delayMs) {
        return executor.schedule(
                Objects.requireNonNull(task, "task"), delayMs, TimeUnit.MILLISECONDS);
    }

    @Override
    public void close() {
        executor.shutdown();
        boolean interrupted = false;
        while (!executor.isTerminated()) {
            try {
                executor.awaitTermination(1, TimeUnit.DAYS);
            } catch (InterruptedException ignored) {
                interrupted = true;
            }
        }
        if (interrupted) Thread.currentThread().interrupt();
    }
}
