package org.flexlb.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/** Opt-in timings for shrink investigations. Logging always happens outside cache locks. */
public final class SyncDiagnostics {
    public static final boolean ENABLED = "1".equals(System.getenv("FLEXLB_SYNC_DIAGNOSTICS"));
    private static final Logger LOG = LoggerFactory.getLogger("syncLogger");
    private static final ConcurrentHashMap<String, AtomicLong> LAST = new ConcurrentHashMap<>();
    private static final ThreadLocal<long[]> LOCK_TIME = ThreadLocal.withInitial(() -> new long[2]);
    private SyncDiagnostics() { }
    public static boolean sample(String key) {
        if (!ENABLED) return false;
        long now = System.nanoTime();
        AtomicLong previous = LAST.computeIfAbsent(key, ignored -> new AtomicLong());
        long old = previous.get();
        return now - old >= 1_000_000_000L && previous.compareAndSet(old, now);
    }
    public static double ms(long nanos) { return nanos / 1_000_000.0; }
    public static void event(String format, Object... args) {
        if (ENABLED) LOG.info("[SyncDiag] " + format, args);
    }
    public static long[] lockSnapshot() {
        return ENABLED ? LOCK_TIME.get().clone() : new long[2];
    }
    public static void cacheLock(String op, String worker, long start, long acquired, long released, int size) {
        if (!ENABLED) return;
        long wait = acquired - start;
        long hold = released - acquired;
        long[] totals = LOCK_TIME.get();
        totals[0] += wait;
        totals[1] += hold;
        if (op.equals("remove_all") || (wait >= 10_000_000L || hold >= 10_000_000L)
                && sample("lock:" + op + ":" + worker)) {
            event("event=cache_lock op={} worker={} wait_ms={} hold_ms={} size={} start_epoch_ms={}",
                    op, worker, ms(wait), ms(hold), size,
                    System.currentTimeMillis() - (released - start) / 1_000_000L);
        }
    }
    public static void cacheRemoval(String worker, long start, long wait, long hold,
                                    long maxHold, int acquisitions, int keys, int size) {
        if (!ENABLED) return;
        long[] totals = LOCK_TIME.get();
        totals[0] += wait;
        totals[1] += hold;
        event("event=cache_lock op=remove_all worker={} wait_ms={} hold_ms={} max_hold_ms={} lock_acquisitions={} keys={} size={} start_epoch_ms={}",
                worker, ms(wait), ms(hold), ms(maxHold), acquisitions, keys, size,
                System.currentTimeMillis() - (System.nanoTime() - start) / 1_000_000L);
    }
}
