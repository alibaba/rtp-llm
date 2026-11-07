package org.flexlb.balance.scheduler;

import java.util.concurrent.Future;

/** Schedules a one-shot task; cancellation does not settle a request or release resources. */
@FunctionalInterface
public interface TaskTimer {
    /** Returns a non-null handle. The callback may start before this method returns; cancel(false) leaves running work intact. */
    Future<?> schedule(Runnable task, long delayMs);
}
