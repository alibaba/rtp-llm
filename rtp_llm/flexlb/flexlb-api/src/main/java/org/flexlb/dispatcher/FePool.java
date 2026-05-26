package org.flexlb.dispatcher;

import org.flexlb.util.Logger;

import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Predicate;
import java.util.function.Supplier;

/**
 * Round-robin pool of FE base URLs. Addresses come through a {@link Supplier} so the upstream
 * (service discovery) owns the freshness story — every {@link #next()} reads a fresh snapshot,
 * no internal cache.
 *
 * <p>{@code isAlive} is the liveness predicate consulted on every pick. Dead hosts are skipped;
 * if every host in the snapshot is dead the pool falls back to plain round-robin instead of
 * refusing service — stale probe data is a worse failure mode than gambling on a possibly-
 * recovered host (and a real outage will be obvious from request errors).
 *
 * <p>The predicate is required: production wires it to {@link FeHealthChecker#isAlive(String)},
 * and tests that don't exercise health filtering pass {@code url -> true} to explicitly declare
 * "all hosts are alive in this test". Leaving the door open to "no health check" would let a
 * call site accidentally regress to the pre-health-check behavior where ~1/N requests land on a
 * dead host until ops intervenes.
 */
public class FePool {

    private final Supplier<List<String>> source;
    private final Predicate<String> isAlive;
    private final AtomicInteger cursor = new AtomicInteger(0);
    /**
     * Latch so the "all FE dead, falling back to RR" diagnostic fires once per outage event,
     * not on every {@link #next()} call during a sustained outage (which would be N×QPS).
     * Resets the instant any subsequent pick finds an alive host.
     */
    private final AtomicBoolean allDeadReported = new AtomicBoolean(false);

    public FePool(Supplier<List<String>> source, Predicate<String> isAlive) {
        if (source == null) {
            throw new IllegalArgumentException("FE pool source must not be null");
        }
        if (isAlive == null) {
            throw new IllegalArgumentException(
                    "isAlive predicate must not be null — pass url -> true for tests");
        }
        this.source = source;
        this.isAlive = isAlive;
    }

    /**
     * Returns the next FE base URL in round-robin order, skipping hosts the predicate marks dead.
     * When every host is dead, falls back to plain round-robin rather than throwing — see class
     * javadoc.
     *
     * @throws IllegalStateException if the current snapshot has no endpoints at all.
     */
    public String next() {
        List<String> snapshot = source.get();
        if (snapshot == null || snapshot.isEmpty()) {
            throw new IllegalStateException("no FE endpoints available");
        }
        int size = snapshot.size();
        int start = cursor.getAndIncrement();
        for (int i = 0; i < size; i++) {
            String candidate = snapshot.get(Math.floorMod(start + i, size));
            if (isAlive.test(candidate)) {
                allDeadReported.set(false);
                return candidate;
            }
        }
        // All dead — fall through to plain round-robin at the original cursor position. Log
        // once per outage so the operator knows the dispatcher is gambling rather than refusing,
        // without flooding the log at request rate.
        if (allDeadReported.compareAndSet(false, true)) {
            Logger.warn("FE pool all-dead fallback: pool size={}, returning RR pick anyway "
                    + "(stale probe data is preferred over refusing service)", size);
        }
        return snapshot.get(Math.floorMod(start, size));
    }
}
