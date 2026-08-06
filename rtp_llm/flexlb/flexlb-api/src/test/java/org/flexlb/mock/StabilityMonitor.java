package org.flexlb.mock;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.InflightStore;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;

import static org.junit.jupiter.api.Assertions.fail;

/**
 * Leak canary for mock-worker tests: verifies that after a test scenario
 * finishes, every layer of inflight accounting has drained back to zero and
 * no tracked future is left hanging.
 *
 * <p>Checked gauges:
 * <ol>
 *   <li>{@link InflightStore#activeCount()} — global non-terminal items</li>
 *   <li>{@link PrefillEndpoint#prefillInflightCount()} +
 *       {@link PrefillEndpoint#prefillEngineTaskCount()} — per-worker batch tracking</li>
 *   <li>{@link DecodeEndpoint#decodeInflightCount()} /
 *       {@link DecodeEndpoint#decodeInflightHardKvReserved()} /
 *       {@link DecodeEndpoint#decodeInflightExpectedKvReserved()} — decode reservation</li>
 *   <li>every future registered via {@link #track} is done</li>
 * </ol>
 *
 * <p>Usage:
 * <pre>{@code
 * StabilityMonitor monitor = new StabilityMonitor(inflightStore)
 *         .watchPrefill(getPrefillEndpoint())
 *         .watchDecode(getDecodeEndpoint());
 * monitor.track(submitRequest(1));
 * // ... scenario ...
 * monitor.assertQuiescent(5_000);
 * }</pre>
 */
public final class StabilityMonitor {

    private final InflightStore inflightStore;
    private final List<PrefillEndpoint> prefillEndpoints = new ArrayList<>();
    private final List<DecodeEndpoint> decodeEndpoints = new ArrayList<>();
    private final CopyOnWriteArrayList<CompletableFuture<?>> trackedFutures = new CopyOnWriteArrayList<>();
    private Runnable pump;

    public StabilityMonitor(InflightStore inflightStore) {
        this.inflightStore = inflightStore;
    }

    /**
     * Register an action executed on every poll round of
     * {@link #assertQuiescent}, standing in for a periodic production runner
     * the mock harness does not start (e.g. the status-sync flow that feeds
     * engine finished reports back into the endpoints).
     */
    public StabilityMonitor pump(Runnable pump) {
        this.pump = pump;
        return this;
    }

    public StabilityMonitor watchPrefill(PrefillEndpoint endpoint) {
        prefillEndpoints.add(endpoint);
        return this;
    }

    public StabilityMonitor watchDecode(DecodeEndpoint endpoint) {
        decodeEndpoints.add(endpoint);
        return this;
    }

    /** Register a future whose completion is required for quiescence. */
    public <T> CompletableFuture<T> track(CompletableFuture<T> future) {
        trackedFutures.add(future);
        return future;
    }

    /**
     * Poll until all gauges are zero and all tracked futures are done, or
     * fail with a per-gauge diagnostic after the timeout.
     */
    public void assertQuiescent(long timeoutMs) {
        long deadline = System.currentTimeMillis() + timeoutMs;
        while (System.currentTimeMillis() < deadline) {
            if (pump != null) {
                pump.run();
            }
            if (leaks().isEmpty()) {
                return;
            }
            try {
                Thread.sleep(50);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
        List<String> leaks = leaks();
        if (!leaks.isEmpty()) {
            fail("Not quiescent within " + timeoutMs + "ms — leaked accounting: " + leaks);
        }
    }

    /** Snapshot of all non-zero gauges and undone futures (empty = quiescent). */
    private List<String> leaks() {
        List<String> leaks = new ArrayList<>();
        int active = inflightStore.activeCount();
        if (active != 0) {
            leaks.add("inflightStore.activeCount=" + active);
        }
        for (PrefillEndpoint ep : prefillEndpoints) {
            int inflight = ep.prefillInflightCount();
            int engineTasks = ep.prefillEngineTaskCount();
            if (inflight != 0 || engineTasks != 0) {
                leaks.add("prefill[" + ep.getIp() + "].inflight=" + inflight
                        + ",engineTasks=" + engineTasks);
            }
        }
        for (DecodeEndpoint ep : decodeEndpoints) {
            int inflight = ep.decodeInflightCount();
            long hardKv = ep.decodeInflightHardKvReserved();
            long expectedKv = ep.decodeInflightExpectedKvReserved();
            if (inflight != 0 || hardKv != 0 || expectedKv != 0) {
                leaks.add("decode[" + ep.getIp() + "].inflight=" + inflight
                        + ",hardKv=" + hardKv + ",expectedKv=" + expectedKv);
            }
        }
        long undone = trackedFutures.stream().filter(f -> !f.isDone()).count();
        if (undone != 0) {
            leaks.add("undoneFutures=" + undone + "/" + trackedFutures.size());
        }
        return leaks;
    }
}
