package org.flexlb.dispatcher;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;

/**
 * Round-robin pool of FE base URLs. Addresses come through a {@link Supplier} so the upstream
 * (service discovery) owns the freshness story — every {@link #next()} reads a fresh snapshot,
 * no internal cache.
 */
public class FePool {

    private final Supplier<List<String>> source;
    private final AtomicInteger cursor = new AtomicInteger(0);

    public FePool(Supplier<List<String>> source) {
        if (source == null) {
            throw new IllegalArgumentException("FE pool source must not be null");
        }
        this.source = source;
    }

    public FePool(List<String> addresses) {
        if (addresses == null || addresses.isEmpty()) {
            throw new IllegalArgumentException("FE pool must not be empty");
        }
        List<String> snapshot = List.copyOf(addresses);
        this.source = () -> snapshot;
    }

    public String next() {
        List<String> snapshot = source.get();
        if (snapshot == null || snapshot.isEmpty()) {
            throw new IllegalStateException("no FE endpoints available");
        }
        return snapshot.get(Math.floorMod(cursor.getAndIncrement(), snapshot.size()));
    }

    public int size() {
        List<String> snapshot = source.get();
        return snapshot == null ? 0 : snapshot.size();
    }
}
