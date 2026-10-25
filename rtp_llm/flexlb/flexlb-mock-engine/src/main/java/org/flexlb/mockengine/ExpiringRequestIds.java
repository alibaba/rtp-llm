package org.flexlb.mockengine;

import java.util.HashMap;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.concurrent.TimeUnit;
import java.util.function.LongSupplier;

/** Semantic cancel history: time expiry, independent of diagnostic row caps. */
final class ExpiringRequestIds {
    private record Expiry(long at, long id) implements Comparable<Expiry> {
        public int compareTo(Expiry other) { return Long.compare(at, other.at); }
    }
    private final Map<Long, Long> ids = new HashMap<>();
    private final PriorityQueue<Expiry> expiries = new PriorityQueue<>();
    private final LongSupplier clock;
    private final long ttl;

    ExpiringRequestIds() { this(System::nanoTime, TimeUnit.MINUTES.toNanos(10)); }
    ExpiringRequestIds(LongSupplier clock, long ttl) {
        if (ttl <= 0) throw new IllegalArgumentException("positive history TTL required");
        this.clock = clock;
        this.ttl = ttl;
    }
    synchronized void add(long id) {
        sweep();
        long at = clock.getAsLong() + ttl;
        ids.put(id, at);
        expiries.add(new Expiry(at, id));
    }
    synchronized boolean contains(long id) { sweep(); return ids.containsKey(id); }
    synchronized void clear() { ids.clear(); expiries.clear(); }
    private void sweep() {
        long now = clock.getAsLong();
        while (!expiries.isEmpty() && expiries.peek().at() <= now) {
            Expiry old = expiries.remove();
            ids.remove(old.id(), old.at());
        }
    }
}
