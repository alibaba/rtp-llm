package org.flexlb.sync.synchronizer;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Keeps at most one task per key in flight: a submission that arrives while the key's previous task
 * is still running is dropped rather than queued.
 *
 * <p>Exists because a fixed-rate scheduler that only <em>hands off</em> work does not bound the work
 * itself — ticks keep firing while a slow task runs, so tasks overlap. That is only safe when tasks
 * are commutative, and a sync round is not: each applies its own snapshot as the truth, so an older
 * round finishing last silently reverts a newer one.
 */
final class SingleFlightGate {

    private final Map<String, AtomicBoolean> inFlight = new ConcurrentHashMap<>();

    /**
     * Submit {@code task} for {@code key} unless a task for that key is still running.
     *
     * @return {@code true} if submitted, {@code false} if skipped because the key was busy
     */
    boolean submit(String key, ExecutorService executor, Runnable task) {
        AtomicBoolean gate = inFlight.computeIfAbsent(key, k -> new AtomicBoolean());
        if (!gate.compareAndSet(false, true)) {
            return false;
        }
        try {
            executor.submit(() -> {
                try {
                    task.run();
                } finally {
                    gate.set(false);
                }
            });
            return true;
        } catch (RuntimeException e) {
            // Rejected by the executor — release the key or it would never run again.
            gate.set(false);
            throw e;
        }
    }
}
