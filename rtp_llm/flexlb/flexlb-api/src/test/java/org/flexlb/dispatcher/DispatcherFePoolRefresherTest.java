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
}
