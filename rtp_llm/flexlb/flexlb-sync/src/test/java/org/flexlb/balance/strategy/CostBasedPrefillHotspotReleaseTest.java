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

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyBoolean;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Hotspot release switch contracts: with the switch enabled, a cache leader
 * whose integral release line climbed past a request's hit rate stops
 * pinning it in both the token-balance and the cache-affinity ranking;
 * with the switch off both rankings behave exactly as before.
 */
class CostBasedPrefillHotspotReleaseTest {

    @Test
    void releasedTokenBalanceLeaderDriftsToLowerUncachedWork() {
        FlexlbConfig config = configWithHotspotRelease();
        PrefillEndpoint leader = tokenEndpoint("leader");
        PrefillEndpoint other = tokenEndpoint("other");
        when(leader.hotspotReleaseLine()).thenReturn(95.0);
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
                    "a closed line must keep pinning despite leader backlog");
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
        verify(leader, never()).governHotspotRelease(anyLong(), anyBoolean(), anyLong());
    }

    @Test
    void enabledSwitchReplacesTheAffinityTtftCap() {
        FlexlbConfig config = affinityConfig();
        EngineHealthReporter reporter = mock(EngineHealthReporter.class);
        CostBasedPrefillStrategy strategy = new CostBasedPrefillStrategy(
                mock(WorkerDirectory.class), mock(CacheAwareService.class), reporter);

        assertEquals(0, strategy.selectBestCandidate(
                candidates(100L, 1000L, 500L, 600L), 100L,
                RoleType.PREFILL, null, 1_000L, config, false),
                "gate off: the TTFT cap must keep rejecting the leader");
        assertEquals(1, strategy.selectBestCandidate(
                candidates(100L, 1000L, 500L, 600L), 100L,
                RoleType.PREFILL, null, 1_000L, config, true),
                "gate on: the closed line replaces the TTFT cap");
        verify(reporter).reportCacheAffinityDecision(
                RoleType.PREFILL, "10.1.0.2", "CACHE_LEADER");
    }

    @Test
    void releasedAffinityLeaderFallsBackToBaseline() {
        FlexlbConfig config = affinityConfig();
        EngineHealthReporter reporter = mock(EngineHealthReporter.class);
        CostBasedPrefillStrategy strategy = new CostBasedPrefillStrategy(
                mock(WorkerDirectory.class), mock(CacheAwareService.class), reporter);
        PrefillCandidateSet survivors = candidates(100L, 1000L, 500L, 600L);
        when(survivors.endpoint(1).hotspotReleaseLine()).thenReturn(80.0);

        assertEquals(0, strategy.selectBestCandidate(
                survivors, 100L, RoleType.PREFILL, null, 1_000L, config, true),
                "a released leader must lose to the minimum-TTFT baseline");
        verify(reporter).reportCacheAffinityDecision(
                RoleType.PREFILL, "10.1.0.1", "HOTSPOT_RELEASE");
    }

    /** Release is strict: a hit rate exactly on the line is still pinned. */
    @Test
    void releasePredicateIsStrictOnTheLine() {
        assertTrue(PrefillTokenBalance.hotspotReleases(50.0, 499L, 1_000L));
        assertFalse(PrefillTokenBalance.hotspotReleases(50.0, 500L, 1_000L));
        assertFalse(PrefillTokenBalance.hotspotReleases(0.0, 499L, 1_000L));
        assertFalse(PrefillTokenBalance.hotspotReleases(50.0, 499L, 0L));
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

    private static CostBasedPrefillStrategy tokenStrategy(
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
                directory, cache, mock(EngineHealthReporter.class));
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
        addCandidate(candidates, 1, "10.1.0.1", firstTtftMs, firstHit);
        addCandidate(candidates, 2, "10.1.0.2", secondTtftMs, secondHit);
        return candidates;
    }

    private static void addCandidate(
            PrefillCandidateSet candidates, int index, String ip,
            long ttftMs, long hitTokens) {
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getIp()).thenReturn(ip);
        candidates.addCandidate(
                ip + ":8080",
                endpoint,
                new RouteProjection.Candidate(
                        ttftMs,
                        ttftMs,
                        RouteProjection.Candidate.InitialHeadDisposition.NONE,
                        "",
                        hitTokens,
                        hitTokens),
                index);
    }
}
