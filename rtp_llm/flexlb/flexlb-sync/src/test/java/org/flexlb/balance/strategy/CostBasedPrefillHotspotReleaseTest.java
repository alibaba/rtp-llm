package org.flexlb.balance.strategy;

import org.flexlb.balance.delivery.DeliveryMetrics;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.QueueSnapshot;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.scheduler.RequestRegistry;
import org.flexlb.balance.scheduler.RouteDeliveryStrategy;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.config.SchedulerConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

import static java.util.concurrent.TimeUnit.MILLISECONDS;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** Shared redirect budget and hotspot selection for both Prefill ranking policies. */
class CostBasedPrefillHotspotReleaseTest {

    private final AtomicLong nowNanos = new AtomicLong();
    private final HotspotRedirectLimiter budget = new HotspotRedirectLimiter(nowNanos::get);

    private void recordColdArrival(long inputTokens) {
        var context = tokenContext(new FlexlbConfig());
        context.getRequest().setSeqLen(inputTokens);
        budget.recordPlacement(context, 0L);
    }

    @Test
    void bestHitBaselineIsPublishedOncePerRequestEvenAfterItsSampleExpires() {
        var request = tokenContext(new FlexlbConfig());
        java.util.stream.IntStream.range(0, 32).parallel()
                .forEach(i -> budget.recordPlacement(request, 900));
        assertFalse(budget.tryAcquire(101, 1000));
        assertTrue(budget.tryAcquire(100, 1000));

        nowNanos.set(MILLISECONDS.toNanos(1000));
        budget.recordPlacement(request, 0L);
        assertFalse(budget.tryAcquire(1, 1000), "a new cache estimate on retry cannot refill the window");
    }

    @Test
    void chargeOnlyAdditionalComputeTokensEvenWhenTheBudgetIsZero() {
        assertTrue(budget.tryAcquire(0, 0));
        assertFalse(budget.tryAcquire(1, 10));
        recordColdArrival(100_099);
        assertFalse(budget.tryAcquire(1001, 10));
        assertTrue(budget.tryAcquire(600, 10));
        assertTrue(budget.tryAcquire(0, 0));
        assertFalse(budget.tryAcquire(400, 10), "600 of 1000 tokens consumed 600 ms of pacing");
        nowNanos.set(MILLISECONDS.toNanos(600));
        assertFalse(budget.tryAcquire(401, 10));
        assertTrue(budget.tryAcquire(400, 10));
        assertTrue(budget.tryAcquire(0, 10));
    }

    @Test
    void baselineAndChargedTokensExpireOnTheirOwnSlidingTimestamps() {
        nowNanos.set(MILLISECONDS.toNanos(900));
        recordColdArrival(200_000);
        nowNanos.set(MILLISECONDS.toNanos(1050));
        assertTrue(budget.tryAcquire(1000, 10));
        nowNanos.set(MILLISECONDS.toNanos(1750));
        recordColdArrival(100_000);
        nowNanos.set(MILLISECONDS.toNanos(1900));
        assertFalse(budget.tryAcquire(1, 10), "remaining 1000-token allowance was already spent");
        nowNanos.set(MILLISECONDS.toNanos(2050));
        assertTrue(budget.tryAcquire(1000, 10));
        nowNanos.set(MILLISECONDS.toNanos(3000));
        assertFalse(budget.tryAcquire(1, 10));
    }

    @Test
    void releasedTokenBalanceLeaderDriftsToLowerUncachedWork() {
        FlexlbConfig config = configWithHotspotRelease();
        config.getRouter().getRoles().getPrefill().getHotspotRelease().setMaxExtraComputePermille(1000);
        recordColdArrival(1000);
        PrefillEndpoint leader = tokenEndpoint("leader");
        PrefillEndpoint other = tokenEndpoint("other");
        when(leader.observedRequestCount()).thenReturn(30L);
        when(leader.tokenWorkloadScore()).thenReturn(800.0);
        CostBasedPrefillStrategy strategy = tokenStrategy(
                config, List.of(
                        new EndpointRegistry.PrefillRoutingEntry("leader:8080", leader),
                        new EndpointRegistry.PrefillRoutingEntry("other:8080", other)));

        try (SelectedRole selected = strategy.select(
                tokenContext(config), RoleType.PREFILL, "g").value()) {
            assertEquals("other", selected.serverStatus().getServerIp(),
                    "a released leader must lose to lower uncached work");
        }

        config.getRouter().getRoles().getPrefill().getHotspotRelease().setMaxExtraComputePermille(0);
        try (SelectedRole selected = strategy.select(tokenContext(config), RoleType.PREFILL, "g").value()) {
            assertEquals("leader", selected.serverStatus().getServerIp(), "exhausted redirect quota keeps the pin");
        }
    }

    @Test
    void placementRetriesDoNotIncreaseTheBestHitComputeBudget() {
        var config = configWithHotspotRelease();
        config.getRouter().getRoles().getPrefill().getHotspotRelease().setMaxExtraComputePermille(1000);
        PrefillEndpoint leader = tokenEndpoint("leader");
        PrefillEndpoint other = tokenEndpoint("other");
        when(leader.observedRequestCount()).thenReturn(30L);
        when(leader.tokenWorkloadScore()).thenReturn(800.0);
        var strategy = tokenStrategy(config, List.of(
                new EndpointRegistry.PrefillRoutingEntry("leader:8080", leader),
                new EndpointRegistry.PrefillRoutingEntry("other:8080", other)));
        var request = tokenContext(config);
        // Each 1000-token request has 900 effective hits at the leader and 500 at the peer.
        for (int attempt = 0; attempt < 5; attempt++) {
            try (var selected = strategy.select(request, RoleType.PREFILL, "g").value()) {
                assertEquals("leader", selected.serverStatus().getServerIp());
            }
        }
        for (int id = 2; id <= 4; id++) {
            var next = tokenContext(config);
            next.getRequest().setRequestId(id);
            try (var selected = strategy.select(next, RoleType.PREFILL, "g").value()) {
                assertEquals(id == 4 ? "other" : "leader", selected.serverStatus().getServerIp());
            }
        }
    }

    @Test
    void unreleasedTokenBalanceLeaderKeepsThePin() {
        FlexlbConfig config = configWithHotspotRelease();
        PrefillEndpoint leader = tokenEndpoint("leader");
        PrefillEndpoint other = tokenEndpoint("other");
        when(leader.tokenWorkloadScore()).thenReturn(800.0);
        CostBasedPrefillStrategy strategy = tokenStrategy(
                config, List.of(
                        new EndpointRegistry.PrefillRoutingEntry("leader:8080", leader),
                        new EndpointRegistry.PrefillRoutingEntry("other:8080", other)));

        try (SelectedRole selected = strategy.select(
                tokenContext(config), RoleType.PREFILL, "g").value()) {
            assertEquals("leader", selected.serverStatus().getServerIp(),
                    "a non-hotspot leader keeps its cache affinity");
        }
    }

    @Test
    void disabledSwitchKeepsTokenBalancePinningUnchanged() {
        PrefillEndpoint leader = tokenEndpoint("leader");
        PrefillEndpoint other = tokenEndpoint("other");
        when(leader.tokenWorkloadScore()).thenReturn(800.0);
        FlexlbConfig config = tokenBalanceConfig(false);
        CostBasedPrefillStrategy strategy = tokenStrategy(
                config, List.of(
                        new EndpointRegistry.PrefillRoutingEntry("leader:8080", leader),
                        new EndpointRegistry.PrefillRoutingEntry("other:8080", other)));

        try (SelectedRole selected = strategy.select(
                tokenContext(config), RoleType.PREFILL, "g").value()) {
            assertEquals("leader", selected.serverStatus().getServerIp());
        }
        verify(leader, never()).observedRequestCount();
    }

    @Test
    void enabledSwitchReplacesTheAffinityTtftCap() {
        FlexlbConfig config = affinityConfig();
        EngineHealthReporter reporter = mock(EngineHealthReporter.class);
        var ranking = new PrefillCandidateRanking(reporter, budget);
        var request = tokenContext(config);

        config.getRouter().getRoles().getPrefill().getHotspotRelease().setEnabled(false);
        assertEquals(0, ranking.select(
                candidates(100L, 1000L, 500L, 600L), RoleType.PREFILL, null, request),
                "gate off: the TTFT cap must keep rejecting the leader");
        assertFalse(budget.tryAcquire(1, 1000), "disabled placements must not fund redirects");
        config.getRouter().getRoles().getPrefill().getHotspotRelease().setEnabled(true);
        assertEquals(1, ranking.select(
                candidates(100L, 1000L, 500L, 600L), RoleType.PREFILL, null, request),
                "gate on: the healthy leader retains affinity");
        assertTrue(budget.tryAcquire(1, 1000), "the same request can contribute once enabled");
        verify(reporter).reportCacheAffinityDecision(
                RoleType.PREFILL, "10.1.0.2", "CACHE_LEADER");
    }

    @Test
    void releasedAffinityLeaderFallsBackToBaseline() {
        FlexlbConfig config = affinityConfig();
        config.getRouter().getRoles().getPrefill().getHotspotRelease().setMaxExtraComputePermille(1000);
        recordColdArrival(1000);
        EngineHealthReporter reporter = mock(EngineHealthReporter.class);
        var ranking = new PrefillCandidateRanking(reporter, budget);
        var request = tokenContext(config);
        PrefillCandidateSet survivors = candidates(100L, 1000L, 500L, 600L);
        sampleRequestLoads(survivors, 4, 30);

        assertEquals(0, ranking.select(
                survivors, RoleType.PREFILL, null, request),
                "a released leader must lose to the minimum-TTFT baseline");
        verify(reporter).reportCacheAffinityDecision(
                RoleType.PREFILL, "10.1.0.1", "HOTSPOT_RELEASE");
        config.getRouter().getRoles().getPrefill().getHotspotRelease().setMaxExtraComputePermille(0);
        assertEquals(1, ranking.select(
                survivors, RoleType.PREFILL, null, request));
        verify(reporter).reportCacheAffinityDecision(RoleType.PREFILL, "10.1.0.2", "HOTSPOT_RATE_LIMIT");
    }

    @Test
    void redirectsHotMinimumTtftBaselineAtZeroCostWhenCacheHitsTie() {
        FlexlbConfig config = affinityConfig();
        config.getRouter().getRoles().getPrefill().getHotspotRelease().setMaxExtraComputePermille(0);
        var reporter = mock(EngineHealthReporter.class);
        var ranking = new PrefillCandidateRanking(reporter, budget);
        var survivors = candidates(100, 200, 500, 500);
        sampleRequestLoads(survivors, 30, 4);

        assertEquals(1, ranking.select(survivors, RoleType.PREFILL, "g", tokenContext(config)));
        verify(reporter).reportCacheAffinityDecision(RoleType.PREFILL, "10.1.0.2", "HOTSPOT_RELEASE");
    }

    @Test
    void redirectFromFastestBestHitBaselineRequiresEnoughExtraCompute() {
        FlexlbConfig config = affinityConfig();
        config.getRouter().getRoles().getPrefill().getHotspotRelease().setMaxExtraComputePermille(10);
        var reporter = mock(EngineHealthReporter.class);
        var ranking = new PrefillCandidateRanking(reporter, budget);
        var survivors = candidates(100, 200, 900, 500);
        sampleRequestLoads(survivors, 30, 4);
        var request = tokenContext(config);

        assertEquals(0, ranking.select(survivors, RoleType.PREFILL, "g", request));
        verify(reporter).reportCacheAffinityDecision(RoleType.PREFILL, "10.1.0.1", "HOTSPOT_RATE_LIMIT");

        recordColdArrival(40_000);
        assertEquals(1, ranking.select(survivors, RoleType.PREFILL, "g", request));
        verify(reporter).reportCacheAffinityDecision(RoleType.PREFILL, "10.1.0.2", "HOTSPOT_RELEASE");
    }

    @Test
    void retainsBaselineWithoutAHealthyAlternativeOrWhenReleaseIsDisabled() {
        FlexlbConfig config = affinityConfig();
        var reporter = mock(EngineHealthReporter.class);
        var ranking = new PrefillCandidateRanking(reporter, budget);
        var request = tokenContext(config);
        var survivors = candidates(100, 200, 500, 500);

        assertEquals(0, ranking.select(survivors, RoleType.PREFILL, "g", request), "healthy baseline");
        sampleRequestLoads(survivors, 30, 30, 4, 4, 4);
        assertEquals(0, ranking.select(survivors, RoleType.PREFILL, "g", request), "all candidates hot");

        var single = new PrefillCandidateSet();
        single.reset(1);
        addCandidate(single, 1, "10.1.0.1", 100, 100, 500);
        sampleRequestLoads(single, 30, 4);
        assertEquals(0, ranking.select(single, RoleType.PREFILL, "g", request), "one candidate");

        sampleRequestLoads(survivors, 30, 4);
        when(survivors.endpoint(1).engineQueueAdmissionBlocked()).thenReturn(true);
        assertEquals(0, ranking.select(survivors, RoleType.PREFILL, "g", request), "peer cannot admit");
        when(survivors.endpoint(1).engineQueueAdmissionBlocked()).thenReturn(false);

        config.getRouter().getRoles().getPrefill().getHotspotRelease().setEnabled(false);
        assertEquals(0, ranking.select(survivors, RoleType.PREFILL, "g", request), "release disabled");
        verify(reporter, times(5))
                .reportCacheAffinityDecision(RoleType.PREFILL, "10.1.0.1", "NO_CACHE_LEAD");
    }

    @Test
    void fastestCoolPeerMustAlsoBeBelowMeanLoadForTtftAndTokenRanking() {
        var survivors = new PrefillCandidateSet();
        survivors.reset(3);
        addCandidate(survivors, 1, "10.1.0.1", 100, 100, 900);
        addCandidate(survivors, 2, "10.1.0.2", 110, 110, 800);
        addCandidate(survivors, 3, "10.1.0.3", 200, 200, 700);
        // 25 is above the pool mean (22.375) but below the source threshold (37.5).
        sampleRequestLoads(survivors, 70, 25, 2, 2, 2, 25, 26, 27);
        for (int i = 0; i < survivors.size(); i++) {
            when(survivors.endpoint(i).canAcceptRequest()).thenReturn(true);
        }
        when(survivors.endpoint(1).tokenWorkloadScore()).thenReturn(1.0);
        when(survivors.endpoint(2).tokenWorkloadScore()).thenReturn(50.0);
        for (FlexlbConfig config : List.of(affinityConfig(), tokenBalanceConfig(true))) {
            config.getRouter().getRoles().getPrefill().getHotspotRelease().setMaxExtraComputePermille(1000);
            var limiter = new HotspotRedirectLimiter(() -> 0L);
            limiter.recordPlacement(tokenContext(config), 0);
            var ranking = new PrefillCandidateRanking(mock(EngineHealthReporter.class), limiter);
            assertEquals(2, ranking.select(survivors, RoleType.PREFILL, "g", tokenContext(config)));
        }
    }

    @Test
    void twoHotGroupsCompeteForOnePacedStreamOfRedirects() {
        var config = affinityConfig();
        config.getRouter().getRoles().getPrefill().getHotspotRelease().setMaxExtraComputePermille(10);
        var ranking = new PrefillCandidateRanking(mock(EngineHealthReporter.class), budget);
        var request = tokenContext(config);
        request.getRequest().setSeqLen(2000L);
        var a = candidates(100, 1000, 0, 1000);
        var b = candidates(100, 1000, 100, 1100);
        sampleRequestLoads(a, 4, 30);
        sampleRequestLoads(b, 4, 30);
        recordColdArrival(500_000);
        for (int ms : new int[] {0, 199, 200, 399, 400, 600, 800, 999}) {
            nowNanos.set(MILLISECONDS.toNanos(ms));
            long grants = java.util.stream.IntStream.range(0, 32).parallel().filter(i -> {
                boolean first = i % 2 == 0;
                return ranking.select(first ? a : b, RoleType.PREFILL,
                        first ? "a" : "b", request) == 0;
            }).count();
            assertEquals(ms % 200 == 0 ? 1 : 0, grants, "redirects at " + ms + "ms");
        }
    }

    /** The pool may include endpoints that did not survive this request's projection. */
    private static void sampleRequestLoads(PrefillCandidateSet survivors, long... loads) {
        var pool = new ArrayList<EndpointRegistry.PrefillRoutingEntry>();
        for (int i = 0; i < loads.length; i++) {
            var endpoint = i < survivors.size() ? survivors.endpoint(i) : mock(PrefillEndpoint.class);
            when(endpoint.observedRequestCount()).thenReturn(loads[i]);
            pool.add(new EndpointRegistry.PrefillRoutingEntry("peer-" + i, endpoint));
        }
        survivors.setHotspots(new PrefillHotspots(() -> 0L).sample(
                RoleType.PREFILL, "g", pool, new RoutingConfig.HotspotReleaseConfig()));
    }

    private static FlexlbConfig tokenBalanceConfig(boolean hotspotRelease) {
        FlexlbConfig config = new FlexlbConfig();
        config.getScheduler().setType(SchedulerConfig.Type.DIRECT);
        config.setDispatcher(DispatcherConfig.nonBatch());
        config.getRouter().getRoles().getPrefill()
                .setTokenBalance(new RoutingConfig.TokenBalanceConfig());
        config.getRouter().getRoles().getPrefill()
                .getHotspotRelease().setEnabled(hotspotRelease);
        return config;
    }

    private static FlexlbConfig configWithHotspotRelease() {
        return tokenBalanceConfig(true);
    }

    private CostBasedPrefillStrategy tokenStrategy(
            FlexlbConfig config, List<EndpointRegistry.PrefillRoutingEntry> fleet) {
        WorkerDirectory directory = mock(WorkerDirectory.class);
        when(directory.prefillRoutingSnapshot(RoleType.PREFILL)).thenReturn(fleet);
        for (EndpointRegistry.PrefillRoutingEntry entry : fleet) {
            PrefillEndpoint endpoint = entry.endpoint();
            long generationId = endpoint.getStatus().getGenerationId();
            var pin = mock(WorkerEndpoint.GenerationPin.class);
            when(pin.endpoint()).thenReturn(endpoint);
            when(pin.generationId()).thenReturn(generationId);
            when(directory.captureEndpoint(RoleType.PREFILL, entry.address())).thenReturn(pin);
        }
        CacheAwareService cache = mock(CacheAwareService.class);
        when(cache.findMatchingEngines(any(), eq(RoleType.PREFILL), any())).thenReturn(
                Map.of("leader:8080", 9, "other:8080", 5));
        return new CostBasedPrefillStrategy(
                directory, cache, mock(EngineHealthReporter.class), budget, new PrefillHotspots(nowNanos::get));
    }

    private static BalanceContext tokenContext(FlexlbConfig config) {
        Request request = new Request();
        request.setRequestId(1L);
        request.setSeqLen(1_000L);
        request.setPriority(50);
        request.setCacheKeyBlockSize(100L);
        BalanceContext context = new BalanceContext(config);
        context.setRequest(request);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(
                50, System.currentTimeMillis() + 60_000L));
        return context;
    }

    /** Mocked endpoint with production-shaped status and projection inputs. */
    private static PrefillEndpoint tokenEndpoint(String ip) {
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(
                WorkerStatus.createDiscovered(
                        RoleType.PREFILL, "g", ip, 8080, 8090, "test"));
        when(endpoint.getIp()).thenReturn(ip);
        when(endpoint.getHttpPort()).thenReturn(8080);
        when(endpoint.canAcceptRequest()).thenReturn(true);
        var predictor = mock(PrefillTimePredictor.class);
        var evaluator = mock(PrefillTimePredictor.Evaluator.class);
        when(evaluator.estimateMs(anyLong(), anyLong())).thenReturn(10L);
        when(predictor.evaluator()).thenReturn(evaluator);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(endpoint.captureRouteProjectionInputs()).thenReturn(new RouteProjection.Inputs(
                new QueueSnapshot(0L, false, null, null, List.of(), null),
                new WorkSnapshot(0L, List.of(), List.of(), 0L)));
        when(endpoint.deliveryProjection()).thenReturn(
                new RouteDeliveryStrategy(
                        mock(RequestRegistry.class), mock(DeliveryMetrics.class))
                        .projectionPolicy());
        when(endpoint.engineQueueAdmissionBlocked()).thenReturn(false);
        when(endpoint.planTokenWork(anyLong()))
                .thenReturn(mock(PrefillTokenWorkload.Reservation.class));
        return endpoint;
    }

    private static FlexlbConfig affinityConfig() {
        FlexlbConfig config = new FlexlbConfig();
        RoutingConfig.CacheAffinityConfig affinity =
                new RoutingConfig.CacheAffinityConfig();
        affinity.setMaxExtraTtftMs(10L);
        affinity.setMinPrefixHitPercent(0.0);
        config.getRouter().getRoles().getPrefill().setCacheAffinity(affinity);
        return config;
    }

    private static PrefillCandidateSet candidates(
            long firstTtftMs, long secondTtftMs, long firstHit, long secondHit) {
        PrefillCandidateSet candidates = new PrefillCandidateSet();
        candidates.reset(2);
        addCandidate(candidates, 1, "10.1.0.1", firstTtftMs, firstTtftMs, firstHit);
        addCandidate(candidates, 2, "10.1.0.2", secondTtftMs, secondTtftMs, secondHit);
        return candidates;
    }

    private static void addCandidate(
            PrefillCandidateSet candidates, int index, String ip,
            long ttftMs, long executionMs, long hitTokens) {
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getIp()).thenReturn(ip);
        candidates.addCandidate(
                ip + ":8080",
                endpoint,
                new RouteProjection.Candidate(
                        ttftMs,
                        executionMs,
                        RouteProjection.Candidate.InitialHeadDisposition.NONE,
                        "",
                        hitTokens,
                        hitTokens),
                index);
    }
}
