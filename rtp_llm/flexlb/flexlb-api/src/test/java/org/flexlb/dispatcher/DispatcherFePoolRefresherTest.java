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
    void emptyPollWithinGraceDoesNotDrainAPopulatedPool() {
        StubServiceDiscovery discovery = new StubServiceDiscovery(
                "svc.fe", WorkerHost.of("10.0.0.1", 8088), WorkerHost.of("10.0.0.2", 8088));
        DispatcherFePoolRefresher r = refresher(discovery, "svc.fe");
        assertEquals(2, r.currentSize());

        discovery.setHosts(List.of());
        r.refresh();

        assertEquals(2, r.currentSize(),
                "a fresh empty discovery result may be a swallowed lookup failure, so within the grace "
                        + "window it must not drop every FE and fail 100% of batch traffic — "
                        + "FeHealthChecker probes the known FEs and takes out any that are genuinely gone");
    }

    @Test
    void emptyListenerPushWithinGraceDoesNotDrainAPopulatedPool() {
        StubServiceDiscovery discovery = new StubServiceDiscovery(
                "svc.fe", WorkerHost.of("10.0.0.1", 8088));
        DispatcherFePoolRefresher r = refresher(discovery, "svc.fe");
        assertEquals(1, r.currentSize());

        discovery.pushHosts(List.of());

        assertEquals(1, r.currentSize(),
                "the listener path must hold the same line as the poll path");
    }

    @Test
    void emptyDiscoveryBeyondGraceDrainsThePoolSoFePoolFailsFast() {
        // Scale-to-zero must converge. Without the grace bound the pool would hold the removed
        // FEs forever, FeHealthChecker would mark them all dead, and FePool.next()'s all-dead
        // fallback would keep shoveling every request at hosts that no longer exist.
        // Unconfigured grace here — this doubles as the guard that the default stays 5 minutes:
        // 1 minute of emptiness keeps the pool, 6 minutes drains it.
        java.util.concurrent.atomic.AtomicLong clock = new java.util.concurrent.atomic.AtomicLong(0);
        StubServiceDiscovery discovery = new StubServiceDiscovery(
                "svc.fe", WorkerHost.of("10.0.0.1", 8088), WorkerHost.of("10.0.0.2", 8088));
        DispatcherFePoolRefresher r = refresher(discovery, "svc.fe", clock::get);
        assertEquals(2, r.currentSize());

        discovery.setHosts(List.of());
        clock.addAndGet(java.util.concurrent.TimeUnit.MINUTES.toNanos(1));
        r.refresh();
        assertEquals(2, r.currentSize(), "within grace the known FEs are kept");

        clock.addAndGet(java.util.concurrent.TimeUnit.MINUTES.toNanos(5));
        r.refresh();
        assertEquals(0, r.currentSize(),
                "an empty result persisting beyond grace is the truth (fleet scaled to zero) "
                        + "and must drain the pool");

        FePool pool = DispatcherTestSupport.fePool(r.source(), url -> true);
        org.junit.jupiter.api.Assertions.assertThrows(IllegalStateException.class, pool::next,
                "with the pool drained, FePool must fail fast instead of timing out against "
                        + "removed hosts");
    }

    @Test
    void configuredGraceDrivesTheDrainPointInsteadOfTheHardcodedFiveMinutes() {
        // The grace window follows FlexlbConfig.discoveryFailureGraceMs, the same knob the sync
        // side reads — an operator who shortened it to 60s must see the FE pool drain on the
        // same schedule as the worker side, not 4 minutes later.
        java.util.concurrent.atomic.AtomicLong clock = new java.util.concurrent.atomic.AtomicLong(0);
        StubServiceDiscovery discovery = new StubServiceDiscovery(
                "svc.fe", WorkerHost.of("10.0.0.1", 8088));
        DispatcherFePoolRefresher r = refresher(discovery, "svc.fe", 60_000, clock::get);
        assertEquals(1, r.currentSize());

        discovery.setHosts(List.of());
        clock.addAndGet(java.util.concurrent.TimeUnit.SECONDS.toNanos(30));
        r.refresh();
        assertEquals(1, r.currentSize(), "within the configured 60s grace the known FEs are kept");

        clock.addAndGet(java.util.concurrent.TimeUnit.SECONDS.toNanos(90));
        r.refresh();
        assertEquals(0, r.currentSize(),
                "past the configured grace the empty result is accepted and the pool drains");
    }

    @Test
    void nonPositiveConfiguredGraceFallsBackToFiveMinuteDefault() {
        // A zero/negative discoveryFailureGraceMs is a misconfiguration, not a request to drain
        // the FE pool on the first discovery hiccup.
        assertEquals(java.util.concurrent.TimeUnit.MINUTES.toNanos(5),
                DispatcherFePoolRefresher.resolveGraceNanos(0));
        assertEquals(java.util.concurrent.TimeUnit.MINUTES.toNanos(5),
                DispatcherFePoolRefresher.resolveGraceNanos(-1));
        assertEquals(java.util.concurrent.TimeUnit.MILLISECONDS.toNanos(60_000),
                DispatcherFePoolRefresher.resolveGraceNanos(60_000));
    }

    @Test
    void springConstructorSourcesGraceFromTheSharedFlexlbConfig() {
        // Plumbing guard: the @Autowired constructor must read discoveryFailureGraceMs off the
        // same ConfigService/FlexlbConfig the sync side uses, so the two discovery consumers
        // cannot drift apart on which knob governs the window.
        StubServiceDiscovery discovery = new StubServiceDiscovery(
                "svc.fe", WorkerHost.of("10.0.0.1", 8088));
        DispatchConfig cfg = new DispatchConfig();
        cfg.setFePoolServiceId("svc.fe");
        org.flexlb.config.FlexlbConfig flexlbConfig = new org.flexlb.config.FlexlbConfig();
        flexlbConfig.setDiscoveryFailureGraceMs(60_000);
        org.flexlb.config.ConfigService configService =
                org.mockito.Mockito.mock(org.flexlb.config.ConfigService.class);
        org.mockito.Mockito.when(configService.loadBalanceConfig()).thenReturn(flexlbConfig);

        DispatcherFePoolRefresher r = new DispatcherFePoolRefresher(discovery, cfg, configService);

        assertEquals(1, r.currentSize());
        // The resolved window, not the interaction: reading the config and then dropping the value
        // (or reading a different knob) would silently leave the 5-minute default in force.
        assertEquals(java.util.concurrent.TimeUnit.MILLISECONDS.toNanos(60_000), r.graceNanos(),
                "the configured discoveryFailureGraceMs must be the window this refresher applies");
    }

    @Test
    void nonEmptyResultResetsTheGraceClock() {
        java.util.concurrent.atomic.AtomicLong clock = new java.util.concurrent.atomic.AtomicLong(0);
        StubServiceDiscovery discovery = new StubServiceDiscovery(
                "svc.fe", WorkerHost.of("10.0.0.1", 8088));
        DispatcherFePoolRefresher r = refresher(discovery, "svc.fe", clock::get);

        // 4 minutes of emptiness, then a real answer, then emptiness again: the second empty
        // run measures from the last non-empty result, not from the first empty one.
        discovery.setHosts(List.of());
        clock.addAndGet(java.util.concurrent.TimeUnit.MINUTES.toNanos(4));
        r.refresh();
        discovery.setHosts(List.of(WorkerHost.of("10.0.0.2", 8088)));
        r.refresh();
        discovery.setHosts(List.of());
        clock.addAndGet(java.util.concurrent.TimeUnit.MINUTES.toNanos(4));
        r.refresh();

        assertEquals(1, r.currentSize(),
                "an intermittent non-empty result must restart the grace window");
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
