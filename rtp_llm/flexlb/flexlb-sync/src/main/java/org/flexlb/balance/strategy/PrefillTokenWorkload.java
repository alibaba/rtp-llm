package org.flexlb.balance.strategy;

import org.flexlb.balance.AdmissionVersion;
import java.math.BigInteger;

/** Accepted and provisional Prefill work on one endpoint generation. */
public final class PrefillTokenWorkload {
    public record Snapshot(long pendingTokens, long plannedTokens, long plannedRequests,
                           long plannedDurationMs, long observedRequests, long admissionVersion) {
        public Snapshot(long pendingTokens, long plannedTokens, long plannedRequests,
                        long plannedDurationMs, long observedRequests) {
            this(pendingTokens, plannedTokens, plannedRequests, plannedDurationMs, observedRequests, 0L);
        }
        public Snapshot(long pendingTokens, long plannedTokens) {
            this(pendingTokens, plannedTokens, 0L, 0L, 0L);
        }

        public Snapshot(long pendingTokens) {
            this(pendingTokens, 0L);
        }

        public double score() {
            return (double) pendingTokens + plannedTokens;
        }
    }

    private long observedRequests;
    private long pendingTokens;
    private long plannedTokens;
    private long plannedRequests;
    private long plannedDurationMs;
    // Predictors may return Long.MAX_VALUE; keep exact excess so a later release restores the score.
    private BigInteger overflowedDuration;
    private Snapshot lastSnapshot;
    private final AdmissionVersion admissionVersion = new AdmissionVersion();

    /** A provisional placement owns workload until admission accepts it or the attempt closes. */
    public final class Reservation implements AutoCloseable {
        private final long tokens;
        private final long durationMs;
        private boolean open = true;

        private Reservation(long tokens, long durationMs) {
            this.tokens = tokens;
            this.durationMs = durationMs;
        }

        private PrefillTokenWorkload owner() {
            return PrefillTokenWorkload.this;
        }

        private void removePlan() {
            if (open) {
                plannedTokens -= tokens;
                changePlannedDuration(-durationMs);
                plannedRequests--;
                changed();
                open = false;
            }
        }

        @Override
        public void close() {
            synchronized (PrefillTokenWorkload.this) {
                removePlan();
            }
        }
    }

    public Reservation plan(long tokens) {
        return tryPlan(AdmissionVersion.ANY, tokens, 0L);
    }

    public synchronized Reservation tryPlan(long expectedVersion, long tokens, long durationMs) {
        if (tokens < 0 || durationMs < 0) {
            throw new IllegalArgumentException("negative planned work");
        }
        if (!admissionVersion.accepts(expectedVersion)) {
            return null;
        }
        long nextTokens = Math.addExact(plannedTokens, tokens);
        Reservation reservation = new Reservation(tokens, durationMs);
        changePlannedDuration(durationMs);
        plannedRequests++;
        plannedTokens = nextTokens;
        changed();
        return reservation;
    }

    private void changePlannedDuration(long delta) {
        if (overflowedDuration == null && delta <= Long.MAX_VALUE - plannedDurationMs) {
            plannedDurationMs += delta;
            return;
        }
        BigInteger total = overflowedDuration == null ? BigInteger.valueOf(plannedDurationMs) : overflowedDuration;
        total = total.add(BigInteger.valueOf(delta));
        if (total.bitLength() <= Long.SIZE - 1) {
            plannedDurationMs = total.longValueExact();
            overflowedDuration = null;
        } else {
            plannedDurationMs = Long.MAX_VALUE;
            overflowedDuration = total;
        }
    }

    public void accept(long tokens) {
        accept(tokens, null);
    }

    /** Publish the canonical item's work and remove its exact plan in one ledger update. */
    public synchronized void accept(long tokens, Reservation plan) {
        if (tokens < 0) {
            throw new IllegalArgumentException("negative accepted work");
        }
        if (plan != null && plan.owner() != this) {
            throw new IllegalArgumentException("Prefill plan belongs to another worker");
        }
        pendingTokens = Math.addExact(pendingTokens, tokens);
        changed();
        if (plan != null) {
            plan.removePlan();
        }
    }

    /** Completion, cancellation or failed admission releases the original accepted work. */
    public synchronized void release(long tokens) {
        if (tokens < 0 || tokens > pendingTokens) {
            throw new IllegalArgumentException("unowned work");
        }
        pendingTokens -= tokens;
        changed();
    }

    public synchronized long pendingTokens() {
        return pendingTokens;
    }

    public synchronized void updateObservedRequests(long requests) {
        observedRequests = requests;
        changed();
    }

    private void changed() {
        admissionVersion.advance();
        lastSnapshot = null;
    }

    public synchronized Snapshot snapshot() {
        if (lastSnapshot == null) {
            lastSnapshot = new Snapshot(pendingTokens, plannedTokens, plannedRequests, plannedDurationMs, observedRequests, admissionVersion.current());
        }
        return lastSnapshot;
    }
}
