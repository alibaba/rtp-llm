package org.flexlb.balance.strategy;

import it.unimi.dsi.fastutil.longs.LongArrayFIFOQueue;
import org.flexlb.dao.BalanceContext;

import java.util.function.LongSupplier;

/** The trailing second of baseline and additional Prefill compute on this Master. */
final class HotspotRedirectLimiter {
    private static final long WINDOW_NANOS = 1_000_000_000L;

    private final LongSupplier nanoTime;
    private final TokenWindow baseline = new TokenWindow();
    private final TokenWindow extra = new TokenWindow();
    private long lastPaidRedirect;
    private long paidRedirectGap;

    HotspotRedirectLimiter() {
        this(System::nanoTime);
    }

    HotspotRedirectLimiter(LongSupplier nanoTime) {
        this.nanoTime = nanoTime;
    }

    /** Count the first available best-hit estimate for this request generation, across placement retries. */
    synchronized void recordPlacement(BalanceContext request, long bestHitTokens) {
        if (!request.claimPrefillComputeSample()) {
            return;
        }
        long now = nanoTime.getAsLong();
        baseline.expire(now);
        baseline.add(now, Math.max(0L, request.getRequest().getSeqLen() - bestHitTokens));
    }

    synchronized boolean tryAcquire(long extraTokens, int permille) {
        if (extraTokens == 0) {
            return true;
        }
        long now = nanoTime.getAsLong();
        baseline.expire(now);
        extra.expire(now);
        // Divide before multiplying to preserve the exact floor even at large token totals.
        long quota = baseline.total / 1000 * permille + baseline.total % 1000 * permille / 1000;
        if (extraTokens > quota - extra.total) {
            return false;
        }
        if (extra.total > 0 && now - lastPaidRedirect < paidRedirectGap) {
            return false;
        }
        extra.add(now, extraTokens);
        lastPaidRedirect = now;
        paidRedirectGap = (long) Math.ceil((double) extraTokens * WINDOW_NANOS / quota);
        return true;
    }

    private static final class TokenWindow {
        private final LongArrayFIFOQueue timestamps = new LongArrayFIFOQueue();
        private final LongArrayFIFOQueue tokens = new LongArrayFIFOQueue();
        private long total;

        void expire(long now) {
            while (!timestamps.isEmpty() && now - timestamps.firstLong() >= WINDOW_NANOS) {
                timestamps.dequeueLong();
                total -= tokens.dequeueLong();
            }
        }

        void add(long now, long amount) {
            // Discard only unrepresentable baseline credit. A granted cost always fits.
            long credited = Math.min(amount, Long.MAX_VALUE - total);
            if (credited > 0) {
                timestamps.enqueue(now);
                tokens.enqueue(credited);
                total += credited;
            }
        }
    }
}
