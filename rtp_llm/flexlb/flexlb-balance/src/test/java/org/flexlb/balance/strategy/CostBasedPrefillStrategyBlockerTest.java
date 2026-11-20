package org.flexlb.balance.strategy;

import org.flexlb.balance.request.Request;
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
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.flexlb.worker.status.WorkerStatusTestSupport.createDiscovered;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Checks token ranking, request admission and fallback to another eligible worker.
 */
class CostBasedPrefillStrategyBlockerTest {
    @Test
    void rotatesWithinRequestedGroupRatherThanWholeDirectory() {
        var fixture = strategy(new FlexlbConfig(), Map.of());
        var inputs =
                List.of(
                        input("a", 0, 1000, "g1"),
                        input("b", 0, 1000, "g1"),
                        input("c", 0, 1000, "g2"),
                        input("d", 0, 1000, "g2"));
        for (String address : List.of("a", "b", "a", "b")) {
            assertEquals(
                    address,
                    fixture.strategy()
                            .chooseBest(request(), "g1", inputs, fixture.matches())
                            .orElseThrow()
                            .address());
        }
    }

    @Test
    void emptyCacheStopsAtFirstIdleWorkerAndKeepsRotating() {
        var fixture = strategy(new FlexlbConfig(), Map.of());
        var first = input("first", 0, 1000);
        var second = input("second", 0, 1000);
        var inputs = List.of(first, second);
        assertEquals("first", select(fixture, inputs).orElseThrow().address());
        org.mockito.Mockito.verify(second, org.mockito.Mockito.never()).selectionSnapshot();
        assertEquals("second", select(fixture, inputs).orElseThrow().address());
        assertEquals("first", select(fixture, inputs).orElseThrow().address());
    }

    @Test
    void laterCacheHitStillBeatsAnIdleWorkerWithoutHits() {
        var fixture = strategy(new FlexlbConfig(), Map.of("cached", 80));
        var selected =
                select(fixture, List.of(input("idle", 0, 1000), input("cached", 10, 1000)))
                        .orElseThrow();
        assertEquals("cached", selected.address());
        assertEquals(80L, selected.maximumRoutingCacheMatchTokens());
    }

    @Test
    void emptyCacheUsesSharedRequestAdmission() {
        var config = new FlexlbConfig();
        var affinity = new RoutingConfig.CacheAffinityConfig();
        config.getRouter().getRoles().getPrefill().getSelection().setCacheAffinity(affinity);
        assertEquals(
                "idle",
                select(strategy(config, Map.of()), List.of(input("idle", 0, 1000)))
                        .orElseThrow()
                        .address());
    }

    @Test
    void idleWorkerFromAnotherGroupIsNotSelected() {
        var fixture = strategy(new FlexlbConfig(), Map.of());
        var other = input("other", 0, 1000);
        var selected =
                fixture.strategy()
                        .chooseBest(request(), "required", List.of(other), fixture.matches());
        assertEquals(java.util.Optional.empty(), selected);
        org.mockito.Mockito.verify(other, org.mockito.Mockito.never()).selectionSnapshot();
    }

    @Test
    void customFormulaRanksEligibleWorkersWithoutChangingTokenCapacity() {
        var config = new FlexlbConfig();
        config.getRouter()
                .getRoles()
                .getPrefill()
                .getSelection()
                .setCostExpression("1000 - computeTokens");
        var strategy = strategy(config, Map.of());
        var result =
                select(
                        strategy,
                        List.of(
                                input("idle", 0, 200),
                                input("busy", 80, 200),
                                input("full", 200, 200)));
        assertEquals("busy", result.orElseThrow().address());
        assertEquals(
                180,
                result.orElseThrow().selectionSnapshot().computeTokens()
                        + result.orElseThrow().computeTokens());
    }

    @Test
    void customFormulaCannotBypassRequestAdmission() {
        var config = new FlexlbConfig();
        config.getRouter().getRoles().getPrefill().getSelection().setCostExpression("0");
        var affinity = new RoutingConfig.CacheAffinityConfig();
        config.getRouter().getRoles().getPrefill().getSelection().setCacheAffinity(affinity);
        var result =
                select(
                        strategy(config, Map.of("cached", 80, "other", 30)),
                        List.of(input("cached", 90, 90), input("other", 0, 1000)));
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
        var result = select(strategy, List.of(input("full", 100, 100), input("fits", 0, 100)));
        assertEquals("fits", result.orElseThrow().address());
        assertEquals(
                100,
                result.orElseThrow().selectionSnapshot().computeTokens()
                        + result.orElseThrow().computeTokens());
    }

    @Test
    void allFullWorkersHaveNoSelection() {
        var result =
                select(strategy(new FlexlbConfig(), Map.of()), List.of(input("full", 100, 100)));
        org.junit.jupiter.api.Assertions.assertTrue(result.isEmpty());
    }

    @Test
    void cacheFirstFallsBackWhenTheLeaderHasNoRequestSlot() {
        var config = new FlexlbConfig();
        var affinity = new RoutingConfig.CacheAffinityConfig();
        config.getRouter().getRoles().getPrefill().getSelection().setCacheAffinity(affinity);
        var strategy = strategy(config, Map.of("cached", 80, "other", 30));
        var result = select(strategy, List.of(input("cached", 70, 70), input("other", 0, 1000)));
        assertEquals("other", result.orElseThrow().address());
        assertEquals(
                70,
                result.orElseThrow().selectionSnapshot().computeTokens()
                        + result.orElseThrow().computeTokens());
        assertEquals(30, result.orElseThrow().cacheHitTokens());
    }

    @Test
    void cacheFirstPrefersMoreHitsAmongAdmissibleWorkers() {
        var config = new FlexlbConfig();
        var affinity = new RoutingConfig.CacheAffinityConfig();
        config.getRouter().getRoles().getPrefill().getSelection().setCacheAffinity(affinity);
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
        return input(address, work, cap, null);
    }

    private static PrefillEndpoint input(String address, long work, long cap, String group) {
        var endpoint = mock(PrefillEndpoint.class);
        when(endpoint.ipPort()).thenReturn(address);
        var status = createDiscovered(RoleType.PREFILL, group, address, 1, 2, null);
        when(endpoint.getStatus()).thenReturn(status);
        when(endpoint.selectionSnapshot())
                .thenReturn(
                        new PrefillEndpoint.SelectionSnapshot(
                                status.getGenerationId(),
                                0L,
                                work,
                                work >= cap ? 128L : 0L,
                                128,
                                false));
        return endpoint;
    }

    private static java.util.Optional<CostBasedPrefillStrategy.Selection> select(
            Fixture fixture, List<PrefillEndpoint> inputs) {
        return fixture.strategy().chooseBest(request(), null, inputs, fixture.matches());
    }

    private static Request request() {
        return Request.builder().requestId("request").seqLen(100L).blockSize(1L).build();
    }

    private static SchedulingMetadata metadata() {
        return SchedulingMetadata.of(50, null, Long.MAX_VALUE, 50);
    }

    private record Fixture(CostBasedPrefillStrategy strategy, CacheMatchResult matches) {}
}
