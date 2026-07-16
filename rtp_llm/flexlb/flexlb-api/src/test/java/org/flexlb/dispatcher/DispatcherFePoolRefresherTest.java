package org.flexlb.dispatcher;

import org.flexlb.dao.master.WorkerHost;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.flexlb.dispatcher.DispatcherTestSupport.StubServiceDiscovery;
import static org.flexlb.dispatcher.DispatcherTestSupport.refresher;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

/**
 * Coverage for the two FE-pool refresh paths that production has historically relied on.
 * Specifically guards the cold-seed regression — an empty discovery snapshot at boot must
 * not throw, and a subsequent poll/listener push must lift the pool to its real size.
 * Without this guard the dispatcher would have shipped with FePool permanently empty under
 * a slow VipServer cache fill (the exact failure mode that motivated commit 5ae94fd99).
 */
class DispatcherFePoolRefresherTest {

    @Test
    void coldSeedAcceptsEmptyDiscoverySnapshot() {
        StubServiceDiscovery discovery = new StubServiceDiscovery("svc.fe");
        DispatcherFePoolRefresher r = refresher(discovery, "svc.fe");
        assertEquals(0, r.currentSize(),
                "cold seed must accept an empty initial pool without throwing");
        assertEquals(1, discovery.getHostsCalls,
                "constructor must seed via getHosts exactly once");
        assertNotNull(discovery.registeredListener,
                "constructor must register a listener for push-based fast-path updates");
    }

    @Test
    void bootSeedDiscoveryExceptionDegradesToEmptyPoolAndPollRecovers() {
        // A transient discovery hiccup at boot must not kill the Spring context: freshness
        // is owned by the listener + poll paths, so boot degrades to an empty snapshot and
        // the next poll repairs it.
        java.util.concurrent.atomic.AtomicBoolean failing = new java.util.concurrent.atomic.AtomicBoolean(true);
        StubServiceDiscovery inner = new StubServiceDiscovery(
                "svc.fe", WorkerHost.of("10.0.0.1", 8088));
        org.flexlb.discovery.ServiceDiscovery flaky = new org.flexlb.discovery.ServiceDiscovery() {
            @Override
            public java.util.List<WorkerHost> getHosts(String address) {
                if (failing.get()) {
                    throw new IllegalStateException("vipserver cache not ready");
                }
                return inner.getHosts(address);
            }

            @Override
            public void listen(String address, org.flexlb.discovery.ServiceHostListener listener) {
                inner.listen(address, listener);
            }

            @Override
            public void shutdown() {}
        };

        DispatcherFePoolRefresher r = refresher(flaky, "svc.fe");
        assertEquals(0, r.currentSize(),
                "boot-time discovery failure must degrade to an empty pool, not fail startup");
        assertNotNull(inner.registeredListener,
                "listener must still be registered after a failed boot seed");

        failing.set(false);
        r.refresh();
        assertEquals(1, r.currentSize(), "poll must repair the pool once discovery recovers");
    }

    @Test
    void pollPathFillsPoolAfterColdSeed() {
        StubServiceDiscovery discovery = new StubServiceDiscovery("svc.fe");
        DispatcherFePoolRefresher r = refresher(discovery, "svc.fe");
        assertEquals(0, r.currentSize());

        discovery.setHosts(List.of(
                WorkerHost.of("10.0.0.1", 8088),
                WorkerHost.of("10.0.0.2", 8088)));
        r.refresh();

        assertEquals(2, r.currentSize(),
                "next poll must surface the freshly populated discovery snapshot");
    }

    @Test
    void listenerPushUpdatesPool() {
        StubServiceDiscovery discovery = new StubServiceDiscovery(
                "svc.fe", WorkerHost.of("10.0.0.1", 8088));
        DispatcherFePoolRefresher r = refresher(discovery, "svc.fe");
        assertEquals(1, r.currentSize());

        discovery.pushHosts(List.of(
                WorkerHost.of("10.0.0.2", 8088),
                WorkerHost.of("10.0.0.3", 8088)));

        assertEquals(2, r.currentSize(),
                "listener push must replace the snapshot without waiting for the next poll");
    }

    @Test
    void emptyPollDoesNotDrainAPopulatedPool() {
        StubServiceDiscovery discovery = new StubServiceDiscovery(
                "svc.fe", WorkerHost.of("10.0.0.1", 8088), WorkerHost.of("10.0.0.2", 8088));
        DispatcherFePoolRefresher r = refresher(discovery, "svc.fe");
        assertEquals(2, r.currentSize());

        discovery.setHosts(List.of());
        r.refresh();

        assertEquals(2, r.currentSize(),
                "an empty discovery result is indistinguishable from a swallowed lookup failure, so it "
                        + "must not drop every FE and fail 100% of batch traffic — FeHealthChecker probes "
                        + "the known FEs and takes out any that are genuinely gone");
    }

    @Test
    void emptyListenerPushDoesNotDrainAPopulatedPool() {
        StubServiceDiscovery discovery = new StubServiceDiscovery(
                "svc.fe", WorkerHost.of("10.0.0.1", 8088));
        DispatcherFePoolRefresher r = refresher(discovery, "svc.fe");
        assertEquals(1, r.currentSize());

        discovery.pushHosts(List.of());

        assertEquals(1, r.currentSize(),
                "the listener path must hold the same line as the poll path");
    }

    @Test
    void poolStillRecoversToANonEmptySnapshotAfterAnEmptyOne() {
        StubServiceDiscovery discovery = new StubServiceDiscovery(
                "svc.fe", WorkerHost.of("10.0.0.1", 8088));
        DispatcherFePoolRefresher r = refresher(discovery, "svc.fe");

        discovery.setHosts(List.of());
        r.refresh();
        discovery.setHosts(List.of(WorkerHost.of("10.0.0.7", 8088), WorkerHost.of("10.0.0.8", 8088)));
        r.refresh();

        assertEquals(2, r.currentSize(),
                "holding the last-known-good snapshot must not wedge the pool — a real answer replaces it");
    }
}
