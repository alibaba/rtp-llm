package org.flexlb.balance.strategy;

import org.flexlb.balance.request.Request;
import org.flexlb.balance.schedule.RequestContext;
import org.flexlb.cache.CacheMatchResult;
import org.flexlb.cache.CacheMatchSource;
import org.flexlb.cache.HostCacheMatch;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.cache.metrics.CacheMetricsReporter;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.routing.RoutingConfig;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.endpoint.PrefillEndpoint;
import org.flexlb.worker.endpoint.WorkerEndpoint;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.flexlb.worker.status.WorkerStatusTestSupport.createDiscovered;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Checks strict cache cohorts separately from ordinary cache-affinity fallback.
 */
class TokenBalancedPrefillStrategyTest {
    @Test
    void hotRequestWaitsForLeaderWhileOrdinaryAffinityMayUseAnotherNode() {
        var config = new FlexlbConfig();
        config.getRouter()
                .getRoles()
                .getPrefill()
                .getSelection()
                .setCacheAffinity(new RoutingConfig.CacheAffinityConfig());
        var service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var endpoints = mock(EndpointRegistry.class);
        var cache = mock(CacheAwareService.class);
        var metrics = mock(CacheMetricsReporter.class);
        var token = new TokenBalancedPrefillStrategy(service, endpoints, cache, metrics);
        var ordinary = new CostBasedPrefillStrategy(service, endpoints, cache, metrics);
        var leader = endpoint("leader", 100, true);
        var alternate = endpoint("alternate", 0, false);
        var matches = matches(Map.of("leader", 80, "alternate", 30));
        assertTrue(
                token.chooseBest(request(), null, List.of(leader, alternate), matches).isEmpty());
        assertEquals(
                "alternate",
                ordinary.chooseBest(request(), null, List.of(leader, alternate), matches)
                        .orElseThrow()
                        .address());
    }

    @Test
    void capacityRetryDoesNotPromoteTheSecondBestCacheNode() {
        var token = strategy();
        var alternate = endpoint("alternate", 0, false);
        var leader = endpoint("leader", 0, true);
        var matches = matches(Map.of("leader", 80, "alternate", 30));
        var candidates =
                token.cacheCandidates(request(), null, List.of(leader, alternate), matches);
        assertTrue(
                token.chooseBest(request(), null, List.of(alternate), matches, candidates)
                        .isEmpty());
    }

    @Test
    void blockedSelectionReportsAllCurrentLeadersWithoutIncludingLowerCacheNodes() {
        var service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        var endpoints = mock(EndpointRegistry.class);
        var cache = mock(CacheAwareService.class);
        var first = endpoint("first", 0, true);
        var second = endpoint("second", 0, true);
        var lower = endpoint("lower", 0, false);
        when(endpoints.<PrefillEndpoint>endpoints(RoleType.PREFILL, null))
                .thenReturn(List.of(lower, first, second));
        when(cache.findMatchingEngines(org.mockito.ArgumentMatchers.any()))
                .thenReturn(matches(Map.of("first", 80, "second", 80, "lower", 30)));
        var context = mock(org.flexlb.balance.schedule.RequestContext.class);
        when(context.getRequest()).thenReturn(request());
        var strategy =
                new TokenBalancedPrefillStrategy(
                        service, endpoints, cache, mock(CacheMetricsReporter.class));
        var result = strategy.selectAndRegister(context, null);
        assertEquals(SelectionResult.Status.BLOCKED, result.status());
        assertEquals(
                java.util.Set.of(
                        first.getStatus().getGenerationId(), second.getStatus().getGenerationId()),
                result.candidateGenerations());
    }

    @ParameterizedTest
    @CsvSource({
        "true, 60, false",
        "true, 80, false",
        "false, 60, false",
        "false, 80, false",
        "false, 80, true"
    })
    void registrationFailureKeepsOnlyAvailableLeadersInWaitingHints(
            boolean leaderUnavailable, int nextHit, boolean nextUnavailable) {
        var service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        var endpoints = mock(EndpointRegistry.class);
        var cache = mock(CacheAwareService.class);
        var leader = endpoint("leader", 0, false);
        var next = endpoint("next", 0, false);
        var cooler = endpoint("cooler", 0, false);
        when(endpoints.<PrefillEndpoint>endpoints(RoleType.PREFILL, null))
                .thenReturn(List.of(leader, next, cooler));
        // The old endpoint may still appear healthy after its registry entry has been removed.
        when(endpoints.getEndpoint(RoleType.PREFILL, "leader"))
                .thenReturn(leaderUnavailable ? null : leader);
        when(endpoints.getEndpoint(RoleType.PREFILL, "next"))
                .thenReturn(nextUnavailable ? null : next);
        when(endpoints.getEndpoint(RoleType.PREFILL, "cooler")).thenReturn(cooler);
        var full =
                WorkerEndpoint.Registration.<PrefillEndpoint.RequestRegistration>rejected(
                        WorkerEndpoint.RegistrationStatus.CAPACITY_FULL);
        when(leader.tryRegisterSelection(anyString(), any(), anyLong(), anyLong()))
                .thenReturn(full);
        when(next.tryRegisterSelection(anyString(), any(), anyLong(), anyLong())).thenReturn(full);
        when(cache.findMatchingEngines(any()))
                .thenReturn(matches(Map.of("leader", 80, "next", nextHit, "cooler", 20)));
        var context = mock(RequestContext.class);
        when(context.getRequest()).thenReturn(request());
        when(context.getRequestId()).thenReturn("test");
        var token =
                new TokenBalancedPrefillStrategy(
                        service, endpoints, cache, mock(CacheMetricsReporter.class));

        var result = token.selectAndRegister(context, null);

        long leaderGeneration = leader.getStatus().getGenerationId();
        long nextGeneration = next.getStatus().getGenerationId();
        var expected =
                leaderUnavailable
                        ? Set.of(nextGeneration)
                        : nextHit == 80 && !nextUnavailable
                                ? Set.of(leaderGeneration, nextGeneration)
                                : Set.of(leaderGeneration);
        assertEquals(SelectionResult.Status.BLOCKED, result.status());
        assertEquals(expected, result.candidateGenerations());
        verify(cache).findMatchingEngines(any());
        verify(cooler, never()).tryRegisterSelection(anyString(), any(), anyLong(), anyLong());
    }

    @Test
    void coldRequestUsesLoadAndCacheThresholdIsStrict() {
        var token = strategy();
        var cached = endpoint("cached", 10, false);
        var idle = endpoint("idle", 0, false);
        assertEquals(
                "idle",
                token.chooseBest(
                                request(),
                                null,
                                List.of(cached, idle),
                                matches(Map.of("cached", 5)))
                        .orElseThrow()
                        .address());
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(ints = {0, 4, 5, 6, 80})
    void queuedAndSelectedWorkUseTheSameColdThreshold(int hit) {
        var service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        var endpoints = mock(EndpointRegistry.class);
        var cache = mock(CacheAwareService.class);
        var node = endpoint("node", 0, false);
        var matches = matches(Map.of("node", hit));
        when(endpoints.<PrefillEndpoint>endpoints(RoleType.PREFILL, null))
                .thenReturn(List.of(node));
        when(cache.findMatchingEngines(org.mockito.ArgumentMatchers.any())).thenReturn(matches);
        var token =
                new TokenBalancedPrefillStrategy(
                        service, endpoints, cache, mock(CacheMetricsReporter.class));
        var selected = token.chooseBest(request(), null, List.of(node), matches).orElseThrow();
        long expected = hit <= 5 ? 100 : 100 - hit;
        assertEquals(expected, selected.computeTokens());
        assertEquals(100 - expected, selected.cacheHitTokens());
        assertEquals(
                expected,
                token.estimateQueuedTokens(request(), RoleType.PREFILL, null).orElseThrow());
        var ordinary =
                new CostBasedPrefillStrategy(
                        service, endpoints, cache, mock(CacheMetricsReporter.class));
        assertEquals(
                100 - hit,
                ordinary.chooseBest(request(), null, List.of(node), matches)
                        .orElseThrow()
                        .computeTokens());
    }

    private static TokenBalancedPrefillStrategy strategy() {
        var service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        return new TokenBalancedPrefillStrategy(
                service,
                mock(EndpointRegistry.class),
                mock(CacheAwareService.class),
                mock(CacheMetricsReporter.class));
    }

    private static PrefillEndpoint endpoint(String address, long work, boolean full) {
        var endpoint = mock(PrefillEndpoint.class);
        var status = createDiscovered(RoleType.PREFILL, null, address, 1, 2, null);
        when(endpoint.getStatus()).thenReturn(status);
        when(endpoint.ipPort()).thenReturn(address);
        when(endpoint.acceptsNewRequests()).thenReturn(true);
        when(endpoint.selectionSnapshot())
                .thenReturn(
                        new PrefillEndpoint.SelectionSnapshot(
                                status.getGenerationId(), 0, work, full ? 128 : 0, 128, false));
        return endpoint;
    }

    private static Request request() {
        return Request.builder().requestId("test").seqLen(100L).blockSize(1L).build();
    }

    private static CacheMatchResult matches(Map<String, Integer> hits) {
        return new CacheMatchResult(
                HostCacheMatch.fromLocalMatches(hits), CacheMatchSource.LOCAL_SYNC, 0, 0);
    }
}
