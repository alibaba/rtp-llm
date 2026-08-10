package org.flexlb.util;

import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.LongSupplier;

/**
 * Emits at most one WARN per interval; calls landing inside the window are counted and the
 * count rides on the next emitted line as a trailing {@code , suppressed=N}, so outage
 * magnitude stays visible without the volume. During an FE/master outage at production QPS an
 * uncapped WARN stream is tens of thousands of log lines per second — enough to cost real
 * throughput.
 */
public final class RateLimitedWarn {

    private final long intervalNanos;
    private final LongSupplier nanoClock;
    private final AtomicLong lastWarnNanos;
    private final AtomicLong suppressed = new AtomicLong();

    public RateLimitedWarn(long interval, TimeUnit unit) {
        this(interval, unit, System::nanoTime);
    }

    /**
     * Test seam for the clock. {@code lastWarnNanos} starts one interval in the past so the first
     * call always emits: {@link System#nanoTime()} has an arbitrary origin (a negative value is
     * JVM-legal), so any fixed initial value would risk suppressing every warn forever.
     */
    RateLimitedWarn(long interval, TimeUnit unit, LongSupplier nanoClock) {
        this.intervalNanos = unit.toNanos(interval);
        this.nanoClock = nanoClock;
        this.lastWarnNanos = new AtomicLong(nanoClock.getAsLong() - intervalNanos);
    }

    /**
     * Logs {@code format} at most once per interval, appending {@code , suppressed=N} with the
     * number of calls swallowed since the last emitted line.
     */
    public void warn(String format, Object... args) {
        long now = nanoClock.getAsLong();
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
