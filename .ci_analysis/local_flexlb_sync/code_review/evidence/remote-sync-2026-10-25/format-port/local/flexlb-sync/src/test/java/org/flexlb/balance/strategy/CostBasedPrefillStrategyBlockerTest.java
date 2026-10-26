package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.planner.GroupingPolicy;
import org.flexlb.balance.prediction.FormulaPredictor;
import org.flexlb.balance.projection.QueueSnapshot;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class CostBasedPrefillStrategyBlockerTest {

    @ParameterizedTest
    @CsvSource({
        "AAAAA,,0 1 2 3 4,5",
        "FAAAA,,0 1 2 3 4,5",
        "AAFAA,,0 1 2 3 4,5",
        "AAAAF,,0 1 2 3 4,5",
        "FFFFF,,0 1 2 3 4,5",
        "FAFAF,,0 1 2 3 4,5",
        "'',,,0",
        "AAAAA,target,0 1 2 3 4,5",
        "OAAAA,target,1 2 3 4,4",
        "AAOAA,target,0 1 3 4,4",
        "AAAAO,target,0 1 2 3,4",
        "OOOOO,target,,0",
        "AOFAO,target,0 2 3,3",
        "OAFOA,target,1 2 4,3"
    })
    void discoveryRanksTheWholeScopedFleetInOrderIncludingFullWorkers(
            String shape, String group, String accepted, int registered) {
        // A: available target member; F: full target member; O: member of another group.
        var expected =
                accepted == null
                        ? List.<String>of()
                        : Arrays.stream(accepted.split(" "))
                                .map(index -> "worker-" + index)
                                .toList();
        var inputs =
                new RouteProjection.Inputs(
                        new QueueSnapshot(
                                0L,
                                true,
                                GroupingPolicy.FIXED_WINDOW,
                                Comparator.comparingLong(GroupPlanner.Item::requestId),
                                new GroupPlanner.Constraints(1, Long.MAX_VALUE, 0L, 0L, 0L),
                                List.of(),
                                null),
                        new WorkSnapshot(0L, List.of(), List.of(), 0L),
                        0L);
        for (RoleType role : List.of(RoleType.PREFILL, RoleType.PDFUSION)) {
            var projectedWorkers = new AtomicInteger();
            var directory = mock(EndpointRegistry.class);
            var cache = mock(CacheAwareService.class);
            when(cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of());
            var strategy =
                    new CostBasedPrefillStrategy(
                            directory,
                            cache,
                            mock(EngineHealthReporter.class),
                            mock(CacheMetricsReporter.class),
                            SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()));
            var entries = new ArrayList<EndpointRegistry.PrefillRoutingEntry>();
            for (int index = 0; index < shape.length(); index++) {
                char kind = shape.charAt(index);
                var endpoint = mock(PrefillEndpoint.class);
                org.springframework.test.util.ReflectionTestUtils.setField(
                        endpoint, "ownership", mock(PrefillState.class));
                org.springframework.test.util.ReflectionTestUtils.setField(
                        endpoint, "requestLimit", 1L);
                when(endpoint.getStatus())
                        .thenReturn(
                                WorkerStatus.createDiscovered(
                                        role,
                                        kind == 'O' ? "other" : "target",
                                        "worker",
                                        8080,
                                        8090,
                                        "test"));
                when(endpoint.ownership.progress())
                        .thenReturn(
                                new PrefillState.Progress(0L, 0L, 0L, 0, 0, kind == 'A' ? 0L : 1L));
                when(endpoint.captureRouteProjectionInputs())
                        .thenAnswer(
                                ignored -> {
                                    projectedWorkers.incrementAndGet();
                                    return inputs;
                                });
                org.springframework.test.util.ReflectionTestUtils.setField(
                        endpoint, "predictor", new FormulaPredictor("sum(computeTokens)"));
                org.springframework.test.util.ReflectionTestUtils.setField(
                        endpoint,
                        "deliveryProjection",
                        RouteProjection.forDelivery(DispatcherConfig.Type.BATCH));
                entries.add(new EndpointRegistry.PrefillRoutingEntry("worker-" + index, endpoint));
            }
            var frozen = List.copyOf(entries);
            when(directory.prefillRoutingSnapshot(role)).thenReturn(frozen);
            var result =
                    StrategyTestSupport.select(
                            strategy,
                            Request.builder().requestId(99L).seqLen(100L).build(),
                            SchedulingMetadata.explicit(50, Long.MAX_VALUE),
                            role,
                            group);

            // Ranking models all scoped workers, including full workers; no generation pin is
            // published by this fixture.
            assertEquals(org.flexlb.balance.PlacementResult.Status.BLOCKED, result.status());
            assertEquals(registered, projectedWorkers.get());
            assertEquals(entries, frozen);
            verify(directory).prefillRoutingSnapshot(role);
            if (expected.isEmpty()) {
                verifyNoInteractions(cache);
            } else {
                verify(cache).findMatchingEngines(any(), eq(role), eq(expected));
            }
            for (int index = 0; index < shape.length(); index++) {
                if (shape.charAt(index) == 'O') {
                    verifyNoInteractions(entries.get(index).endpoint().ownership);
                }
            }
        }
    }

    @Test
    void everyRegisteredEndpointMustProveTheSameBlocker() {
        assertNull(CostBasedPrefillStrategy.provenPoolWideBlocker(Map.of(RoleType.DECODE, 3), 4));
        assertNull(
                CostBasedPrefillStrategy.provenPoolWideBlocker(
                        Map.of(RoleType.DECODE, 2, RoleType.PREFILL, 2), 4));
        assertEquals(
                RoleType.DECODE,
                CostBasedPrefillStrategy.provenPoolWideBlocker(Map.of(RoleType.DECODE, 4), 4));
    }
}
