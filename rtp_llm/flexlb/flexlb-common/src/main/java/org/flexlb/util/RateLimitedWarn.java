package org.flexlb.util;

import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Emits at most one WARN per interval; calls landing inside the window are counted and the
 * count rides on the next emitted line as a trailing {@code , suppressed=N}, so outage
 * magnitude stays visible without the volume. During an FE/master outage at production QPS an
 * uncapped WARN stream is tens of thousands of log lines per second — enough to cost real
 * throughput.
 */
public final class RateLimitedWarn {

    private final long intervalNanos;
    private final AtomicLong lastWarnNanos = new AtomicLong();
    private final AtomicLong suppressed = new AtomicLong();

    public RateLimitedWarn(long interval, TimeUnit unit) {
        this.intervalNanos = unit.toNanos(interval);
    }

    /**
     * Logs {@code format} at most once per interval, appending {@code , suppressed=N} with the
     * number of calls swallowed since the last emitted line.
     */
    public void warn(String format, Object... args) {
        long now = System.nanoTime();
        long last = lastWarnNanos.get();
        if (now - last >= intervalNanos && lastWarnNanos.compareAndSet(last, now)) {
            Object[] withSuppressed = new Object[args.length + 1];
            System.arraycopy(args, 0, withSuppressed, 0, args.length);
            withSuppressed[args.length] = suppressed.getAndSet(0);
            Logger.warn(format + ", suppressed={}", withSuppressed);
        } else {
            suppressed.incrementAndGet();
        }
    }
}
