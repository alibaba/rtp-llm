package org.flexlb.balance.strategy;

import org.flexlb.balance.schedule.SchedulingMetadata;
import org.flexlb.cache.CacheMatchResult;
import org.flexlb.cache.CacheMatchSource;
import org.flexlb.cache.HostCacheMatch;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.routing.RoutingConfig;
import org.flexlb.config.service.RoleType;
import org.flexlb.util.formula.PrefillTimePredictor;
import org.flexlb.worker.endpoint.PrefillEndpoint;
import org.flexlb.worker.endpoint.Request;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Checks compute-token ranking, absolute capacity limits and retrying another eligible worker.
 */
class CostBasedPrefillStrategyBlockerTest {
    @Test
    void customFormulaRanksEligibleWorkersWithoutChangingTokenCapacity() {
        var config = new FlexlbConfig();
        config.getRouter().getRoles().getPrefill().setCostExpression("1000 - computeTokens");
        var strategy = strategy(config, Map.of());
        var result =
                select(
                        strategy,
                        List.of(
                                input("idle", 0, 200),
                                input("busy", 80, 200),
                                input("full", 150, 200)));
        assertEquals("busy", result.orElseThrow().address());
        assertEquals(
                180,
                result.orElseThrow().selectionSnapshot().computeTokens()
                        + result.orElseThrow().computeTokens());
    }

    @Test
    void customFormulaCannotBypassCacheFirstTokenCap() {
        var config = new FlexlbConfig();
        config.getRouter().getRoles().getPrefill().setCostExpression("0");
        var affinity = new RoutingConfig.CacheAffinityConfig();
        affinity.setMaxComputeTokens(100L);
        config.getRouter().getRoles().getPrefill().setCacheAffinity(affinity);
        var result =
                select(
                        strategy(config, Map.of("cached", 80, "other", 30)),
                        List.of(input("cached", 90, 1000), input("other", 0, 1000)));
        assertEquals("other", result.orElseThrow().address());
    }

    @Test
    void invalidFormulaResultsCannotBecomeWorkerScores() {
        for (String expression : List.of("-1", "1 / 0", "sqrt(-1)")) {
            var predictor = new PrefillTimePredictor(expression);
            org.junit.jupiter.api.Assertions.assertThrows(
                    IllegalArgumentException.class, () -> predictor.score(100));
        }
    }

    @Test
    void computeTokensOnlySubtractsReusableInputTokens() {
        var predictor = new PrefillTimePredictor();
        assertEquals(70, predictor.computeTokens(100, 30));
        assertEquals(0, predictor.computeTokens(100, 200));
        assertEquals(100, predictor.computeTokens(100, -20));
        assertEquals(0, predictor.computeTokens(-10, 20));
        assertEquals(Long.MAX_VALUE, predictor.computeTokens(Long.MAX_VALUE, 0));
    }

    @Test
    void selectsLowestWorkAfterAddingIncomingComputeTokens() {
        var strategy = strategy(new FlexlbConfig(), Map.of());
        var result = select(strategy, List.of(input("busy", 70, 1000), input("free", 10, 1000)));
        org.junit.jupiter.api.Assertions.assertTrue(result.isPresent());
        assertEquals("free", result.orElseThrow().address());
        assertEquals(100, result.orElseThrow().computeTokens());
        assertEquals(
                110,
                result.orElseThrow().selectionSnapshot().computeTokens()
                        + result.orElseThrow().computeTokens());
    }

    @Test
    void skipsFullWorkerAndAcceptsExactlyAtCapacity() {
        var strategy = strategy(new FlexlbConfig(), Map.of());
        var result = select(strategy, List.of(input("full", 1, 100), input("fits", 0, 100)));
        assertEquals("fits", result.orElseThrow().address());
        assertEquals(
                100,
                result.orElseThrow().selectionSnapshot().computeTokens()
                        + result.orElseThrow().computeTokens());
    }

    @Test
    void allFullWorkersHaveNoSelection() {
        var result = select(strategy(new FlexlbConfig(), Map.of()), List.of(input("full", 1, 100)));
        org.junit.jupiter.api.Assertions.assertTrue(result.isEmpty());
    }

    @Test
    void cacheFirstCannotExceedItsAbsoluteComputeTokenCap() {
        var config = new FlexlbConfig();
        var affinity = new RoutingConfig.CacheAffinityConfig();
        affinity.setMaxComputeTokens(80L);
        config.getRouter().getRoles().getPrefill().setCacheAffinity(affinity);
        var strategy = strategy(config, Map.of("cached", 80, "other", 30));
        var result = select(strategy, List.of(input("cached", 70, 1000), input("other", 0, 1000)));
        assertEquals("other", result.orElseThrow().address());
        assertEquals(
                70,
                result.orElseThrow().selectionSnapshot().computeTokens()
                        + result.orElseThrow().computeTokens());
        assertEquals(30, result.orElseThrow().cacheHitTokens());
    }

    @Test
    void cacheFirstPrefersMoreHitsAmongWorkersWithinBothCaps() {
        var config = new FlexlbConfig();
        var affinity = new RoutingConfig.CacheAffinityConfig();
        affinity.setMaxComputeTokens(100L);
        config.getRouter().getRoles().getPrefill().setCacheAffinity(affinity);
        var strategy = strategy(config, Map.of("cached", 80, "other", 30));
        var result = select(strategy, List.of(input("cached", 70, 1000), input("other", 0, 1000)));
        assertEquals("cached", result.orElseThrow().address());
        assertEquals(
                90,
                result.orElseThrow().selectionSnapshot().computeTokens()
                        + result.orElseThrow().computeTokens());
    }

    private static Fixture strategy(FlexlbConfig config, Map<String, Integer> hits) {
        var service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        return new Fixture(
                new CostBasedPrefillStrategy(
                        service,
                        org.mockito.Mockito.mock(org.flexlb.worker.endpoint.EndpointRegistry.class),
                        org.mockito.Mockito.mock(org.flexlb.cache.match.CacheAwareService.class),
                        org.mockito.Mockito.mock(
                                org.flexlb.cache.metrics.CacheMetricsReporter.class)),
                new CacheMatchResult(
                        HostCacheMatch.fromLocalMatches(hits),
                        CacheMatchSource.LOCAL_SYNC,
                        0L,
                        0L));
    }

    private static PrefillEndpoint input(String address, long work, long cap) {
        var endpoint = mock(PrefillEndpoint.class);
        when(endpoint.ipPort()).thenReturn(address);
        var status =
                org.flexlb.worker.status.WorkerStatus.createDiscovered(
                        RoleType.PREFILL, null, address, 1, 2, null);
        when(endpoint.getStatus()).thenReturn(status);
        when(endpoint.selectionSnapshot())
                .thenReturn(
                        new PrefillEndpoint.SelectionSnapshot(
                                status.getGenerationId(), 0L, work, cap));
        return endpoint;
    }

    private static java.util.Optional<CostBasedPrefillStrategy.Selection> select(
            Fixture fixture, List<PrefillEndpoint> inputs) {
        return fixture.strategy().chooseBest(request(), inputs, fixture.matches());
    }

    private static Request request() {
        return Request.builder().requestId("request").seqLen(100L).blockSize(1L).build();
    }

    private static SchedulingMetadata metadata() {
        return SchedulingMetadata.explicit(50, Long.MAX_VALUE);
    }

    private record Fixture(CostBasedPrefillStrategy strategy, CacheMatchResult matches) {}
}
