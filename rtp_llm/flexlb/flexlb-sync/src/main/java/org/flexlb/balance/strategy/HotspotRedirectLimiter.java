package org.flexlb.balance.strategy;

import it.unimi.dsi.fastutil.longs.LongArrayFIFOQueue;
import org.flexlb.dao.BalanceContext;

import java.util.LinkedHashSet;
import java.util.function.LongSupplier;

/** The trailing second of baseline and additional Prefill compute on this Master. */
public final class HotspotRedirectLimiter {
    private static final long WINDOW_NANOS = 1_000_000_000L;

    private final LongSupplier nanoTime;
    private final TokenWindow baseline = new TokenWindow();
    private final LinkedHashSet<Ticket> charges = new LinkedHashSet<>();
    private long chargedTokens;

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

    sealed interface Acquisition permits Ticket, Rejection {}

    enum Rejection implements Acquisition { BUDGET_EXHAUSTED, PACING }

    synchronized Acquisition tryAcquire(long extraTokens, int permille) {
        if (extraTokens == 0) {
            return Ticket.FREE;
        }
        long now = nanoTime.getAsLong();
        baseline.expire(now);
        while (!charges.isEmpty() && now - charges.getFirst().at >= WINDOW_NANOS) {
            chargedTokens -= charges.removeFirst().tokens;
        }
        // Divide before multiplying to preserve the exact floor even at large token totals.
        long quota = baseline.total / 1000 * permille + baseline.total % 1000 * permille / 1000;
        if (extraTokens > quota - chargedTokens) {
            return Rejection.BUDGET_EXHAUSTED;
        }
        if (!charges.isEmpty()) {
            Ticket latest = charges.getLast();
            if (now - latest.at < latest.gap) { return Rejection.PACING; }
        }
        long gap = (long) Math.ceil((double) extraTokens * WINDOW_NANOS / quota);
        Ticket ticket = new Ticket(this, now, extraTokens, gap);
        charges.add(ticket);
        chargedTokens += extraTokens;
        return ticket;
    }

    /** A charge counts from acquisition. Abandoning an uncommitted route restores its quota and pacing. */
    public static final class Ticket implements Acquisition, AutoCloseable {
        private static final Ticket FREE = new Ticket(null, 0L, 0L, 0L);
        private final HotspotRedirectLimiter limiter;
        private final long at;
        private final long tokens;
        private final long gap;
        private boolean committed;

        private Ticket(HotspotRedirectLimiter limiter, long at, long tokens, long gap) {
            this.limiter = limiter;
            this.at = at;
            this.tokens = tokens;
            this.gap = gap;
        }

        public void commit() {
            if (limiter != null) {
                synchronized (limiter) { committed = true; }
            }
        }

        @Override
        public void close() {
            if (limiter != null) {
                synchronized (limiter) {
                    if (!committed && limiter.charges.remove(this)) {
                        limiter.chargedTokens -= tokens;
                    }
                }
            }
        }
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
