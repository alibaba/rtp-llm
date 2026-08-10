package org.flexlb.dispatcher;

import org.flexlb.util.Logger;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
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
@Component
@ConditionalOnProperty(prefix = "dispatch", name = "fe-pool-service-id")
public class FePool {

    private final Supplier<List<String>> source;
    private final Predicate<String> isAlive;
    /**
     * Rotation cursor. A {@code long} (not {@code int}) so it never wraps at any realistic QPS
     * (2^63 picks); the earlier {@code int} cursor wrapped every ~2^32 picks, and because 2^32 is
     * not a multiple of the pool size that produced a one-off RR discontinuity at the wrap point.
     */
    private final AtomicLong cursor = new AtomicLong(0);
    /**
     * Latch so the "all FE dead, falling back to RR" diagnostic fires once per outage event,
     * not on every {@link #next()} call during a sustained outage (which would be N×QPS).
     * Resets the instant any subsequent pick finds an alive host.
     */
    private final AtomicBoolean allDeadReported = new AtomicBoolean(false);

    public FePool(DispatcherFePoolRefresher refresher, FeHealthChecker healthChecker) {
        this.source = refresher.source();
        this.isAlive = healthChecker::isAlive;
    }

    /**
     * Returns the next FE base URL in round-robin order, skipping hosts the predicate marks dead.
     * When every host is dead, falls back to plain round-robin rather than throwing — see class
     * javadoc.
     *
     * @throws IllegalStateException if the current snapshot has no endpoints at all.
     */
    public String next() {
        List<String> pool = livePool();
        return pool.get(Math.floorMod(cursor.getAndIncrement(), pool.size()));
    }

    /**
     * Returns {@code count} FE base URLs for a single batch, advancing the shared cursor by exactly
     * {@code count}. Resolves the liveness-filtered pool <em>once</em> for the whole batch instead
     * of once per pick as repeated {@link #next()} would: {@link MasterFeAssigner} calls this once
     * per batch-schedule request with {@code count == targets.size()}, so a 500-target request that
     * would otherwise rebuild and re-filter the FE snapshot 500 times (once per {@code next()}) now
     * does it once. Per-pick semantics are identical to {@link #next()}: round-robin over the alive
     * subset, or over the full snapshot when all dead. The returned list has exactly {@code count}
     * elements (empty when {@code count <= 0}), so the caller can zip it 1:1 with its targets.
     *
     * <p>The cursor block is reserved atomically, so two concurrent batches never overlap picks.
     * But each call resolves its own {@link #livePool()} snapshot, so when the alive set changes
     * between two concurrent calls the global "cursor sequence -> host" round-robin is only
     * approximate across that change point; each individual batch is still internally consistent.
     *
     * @throws IllegalStateException if the current snapshot has no endpoints at all — thrown before
     *     any pick, so an empty snapshot yields no partial assignment (all-or-nothing per batch).
     */
    public List<String> nextBatch(int count) {
        if (count <= 0) {
            return new ArrayList<>();
        }
        List<String> pool = livePool();
        // Reserve a contiguous cursor block so concurrent batches interleave into disjoint ranges
        // rather than contending pick-by-pick. The cursor is a long, so at any realistic QPS it
        // never wraps (2^63 picks); floorMod still keeps every index valid if it ever did.
        long base = cursor.getAndAdd(count);
        int n = pool.size();
        List<String> picks = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            picks.add(pool.get(Math.floorMod(base + i, n)));
        }
        return picks;
    }

    /**
     * The non-empty pool to round-robin over for this call: the alive subset, or — when every host
     * is dead — the full snapshot (see class javadoc). Resolved from a fresh supplier snapshot on
     * every call so upstream discovery owns freshness.
     *
     * @throws IllegalStateException if the snapshot has no endpoints at all.
     */
    private List<String> livePool() {
        List<String> snapshot = source.get();
        if (snapshot == null || snapshot.isEmpty()) {
            throw new IllegalStateException("no FE endpoints available");
        }
        // Round-robin over the alive subset so a dead host's share spreads across the whole pool
        // instead of funneling onto its successor. The steady state is "no dead host", so scan
        // without allocating and only materialize the filtered subset once a dead host actually
        // has to be dropped — an all-alive snapshot is then round-robined in place, no per-call
        // copy of the whole pool.
        List<String> alive = null;
        for (int i = 0; i < snapshot.size(); i++) {
            String candidate = snapshot.get(i);
            if (isAlive.test(candidate)) {
                if (alive != null) {
                    alive.add(candidate);
                }
            } else if (alive == null) {
                // First dead host: seed the filtered list with the all-alive prefix scanned so far.
                alive = new ArrayList<>(snapshot.size());
                for (int j = 0; j < i; j++) {
                    alive.add(snapshot.get(j));
                }
            }
        }
        if (alive == null) {
            // No dead host in this snapshot — round-robin over it directly, skipping the copy.
            // Safe to hand back by reference: the supplier publishes an unmodifiable list and swaps
            // it wholesale on refresh (never mutates in place), and both pick paths only read it.
            allDeadReported.set(false);
            return snapshot;
        }
        if (!alive.isEmpty()) {
            allDeadReported.set(false);
            return alive;
        }
        // All dead — fall through to plain round-robin over the full snapshot. Log once per
        // outage so the operator knows the dispatcher is gambling rather than refusing,
        // without flooding the log at request rate.
        if (allDeadReported.compareAndSet(false, true)) {
            Logger.warn("FE pool all-dead fallback: pool size={}, returning RR pick anyway "
                    + "(stale probe data is preferred over refusing service)", snapshot.size());
        }
        return snapshot;
    }
}
