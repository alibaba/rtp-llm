package org.flexlb.mockengine;

import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

/** Absolute monotonic request budget, independent of Fetch attachment and frame gaps.
 * Real references: GenerateStream::checkTimeoutWithoutLock, DecodeGenerateContext.
 */
final class MockRequestDeadline implements AutoCloseable {
    private final long started = System.nanoTime();
    private final long timeoutMs;
    private ScheduledFuture<?> timer;
    private boolean finished;
    private volatile boolean expired;

    MockRequestDeadline(long timeoutMs) { this.timeoutMs = timeoutMs; }
    long remainingMs() {
        if (timeoutMs <= 0) return Long.MAX_VALUE;
        return Math.max(0, timeoutMs - TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - started));
    }
    boolean expired() { return expired; }
    synchronized void arm(ScheduledExecutorService scheduler, Runnable timeout) {
        if (timeoutMs <= 0 || finished) return;
        timer = scheduler.schedule(() -> {
            synchronized (this) {
                if (finished) return;
                expired = true;
                finished = true;
            }
            timeout.run();
        }, remainingMs(), TimeUnit.MILLISECONDS);
    }
    @Override public synchronized void close() {
        finished = true;
        if (timer != null) timer.cancel(false);
    }
}
