package org.flexlb.balance.scheduler;

import java.util.Objects;
import java.util.concurrent.locks.LockSupport;

/** Drives decisions and deadlines on a supplied thread; never chooses requests or retries. */
public final class SchedulerLoop {
    private final Thread thread;
    private volatile boolean stopped;
    private final java.util.concurrent.atomic.AtomicLong notifications =
            new java.util.concurrent.atomic.AtomicLong();

    public SchedulerLoop(Thread.Builder builder, RequestScheduler scheduler, Runnable onStopped) {
        Objects.requireNonNull(scheduler);
        Objects.requireNonNull(onStopped);
        thread = builder.unstarted(() -> {
            try {
                while (!stopped) {
                    long observed = notifications.get();
                    long next = scheduler.schedule(System.nanoTime());
                    if (stopped) break;
                    if (notifications.get() != observed) continue;
                    if (next == Long.MAX_VALUE) {
                        LockSupport.park(this);
                    } else {
                        long remaining = next - System.nanoTime();
                        if (remaining > 0) LockSupport.parkNanos(this, remaining);
                    }
                    Thread.interrupted();
                }
            } finally {
                stopped = true;
                onStopped.run();
            }
        });
    }

    public void start() {
        thread.start();
    }

    /** The revision also covers permits consumed by blocking operations inside schedule(). */
    public void wake() {
        notifications.incrementAndGet();
        LockSupport.unpark(thread);
    }

    public void stop() {
        stopped = true;
        wake();
    }

    public void awaitStopped() {
        if (Thread.currentThread() == thread) return;
        boolean interrupted = false;
        while (thread.isAlive()) {
            try {
                thread.join();
            } catch (InterruptedException interruption) {
                interrupted = true;
            }
        }
        if (interrupted) Thread.currentThread().interrupt();
    }
}
