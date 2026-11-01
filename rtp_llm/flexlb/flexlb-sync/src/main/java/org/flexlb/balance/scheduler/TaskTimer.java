package org.flexlb.balance.scheduler;

/** Schedules a one-shot task; cancellation does not settle a request or release resources. */
@FunctionalInterface
public interface TaskTimer {
    @FunctionalInterface
    interface ScheduledTask {
        /** Cancels a pending callback; a callback already running may finish. */
        void cancel();
    }

    /** The callback may start before this method returns. */
    ScheduledTask schedule(Runnable task, long delayMs);
}
