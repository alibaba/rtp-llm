package org.flexlb.balance.scheduler;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** Deltas for mutations of one Prefill generation's Master waiting queue. */
public final class PrefillQueueFlow {
    public enum Event {
        ENQUEUE, MIGRATION_IN, MIGRATION_RESTORED,
        MIGRATION_OUT, DISPATCH_COMMITTED, EXPIRED, CANCELLED, WITHDRAWN, FAILED, STOPPED, REMOVED
    }

    public record Entry(Event event, long requests, long computeTokens) { }

    public record Snapshot(List<Entry> entries) {
        public Snapshot { entries = List.copyOf(entries); }
    }

    private static final Event[] EVENTS = Event.values();
    private static final Snapshot EMPTY = new Snapshot(List.of());
    private final long[] requests = new long[EVENTS.length];
    private final long[] computeTokens = new long[EVENTS.length];

    /** Caller owns the same queue lock used for the mutation. */
    void record(Event event, ScheduledRequest item) {
        record(event, 1L, addComputeTokens(0L, item));
    }

    static long addComputeTokens(long total, ScheduledRequest item) {
        long tokens = Math.max(0L, item.seqLen() - item.hitCache());
        return total + Math.min(tokens, Long.MAX_VALUE - total);
    }

    /** Caller owns the queue lock. */
    void record(Event event, long count, long tokens) {
        int index = event.ordinal();
        requests[index] += Math.min(count, Long.MAX_VALUE - requests[index]);
        computeTokens[index] += Math.min(tokens, Long.MAX_VALUE - computeTokens[index]);
    }

    /** Caller owns the queue lock. Only non-empty event buckets are emitted. */
    Snapshot drain() {
        List<Entry> entries = null;
        for (Event event : EVENTS) {
            int index = event.ordinal();
            if (requests[index] != 0L) {
                if (entries == null) { entries = new ArrayList<>(); }
                entries.add(new Entry(event, requests[index], computeTokens[index]));
            }
        }
        if (entries == null) { return EMPTY; }
        Snapshot snapshot = new Snapshot(entries);
        Arrays.fill(requests, 0L);
        Arrays.fill(computeTokens, 0L);
        return snapshot;
    }
}
