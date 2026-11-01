package org.flexlb.balance.scheduler;

import java.util.Objects;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/** Owns the existing timer pool for bounded waits in delivery and preemption. */
public final class ScheduledTaskTimer implements TaskTimer {
    private final ScheduledThreadPoolExecutor cleanupExecutor =
            new ScheduledThreadPoolExecutor(
                    2, Thread.ofPlatform().daemon().name("flexlb-request-cleanup-", 1).factory());

    ScheduledTaskTimer() {
        cleanupExecutor.setRemoveOnCancelPolicy(true);
        cleanupExecutor.setExecuteExistingDelayedTasksAfterShutdownPolicy(false);
        cleanupExecutor.setContinueExistingPeriodicTasksAfterShutdownPolicy(false);
    }

    @Override
    public ScheduledTask schedule(Runnable task, long delayMs) {
        var future =
                cleanupExecutor.schedule(
                        Objects.requireNonNull(task, "task"), delayMs, TimeUnit.MILLISECONDS);
        return () -> future.cancel(false);
    }

    public void close() {
        cleanupExecutor.shutdown();
        boolean interrupted = false;
        while (!cleanupExecutor.isTerminated()) {
            try {
                cleanupExecutor.awaitTermination(1, TimeUnit.DAYS);
            } catch (InterruptedException ignored) {
                interrupted = true;
            }
        }
        if (interrupted) Thread.currentThread().interrupt();
    }
}
