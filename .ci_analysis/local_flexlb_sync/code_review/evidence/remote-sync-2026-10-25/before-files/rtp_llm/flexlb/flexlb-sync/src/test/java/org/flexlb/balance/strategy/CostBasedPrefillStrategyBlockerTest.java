package org.flexlb.balance.strategy;

import org.flexlb.balance.scheduler.SchedulingTestConfig;


import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.planner.GroupingPolicy;
import org.flexlb.balance.prediction.FormulaPredictor;
import org.flexlb.balance.projection.QueueSnapshot;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
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
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

/** Regression contracts for Decode capacity projected through Prefill. */
class CostBasedPrefillStrategyBlockerTest {

    @ParameterizedTest
    @CsvSource({
            "AAAAA,,0 1 2 3 4,5", "FAAAA,,1 2 3 4,5", "AAFAA,,0 1 3 4,5",
            "AAAAF,,0 1 2 3,5", "FFFFF,,,5", "FAFAF,,1 3,5", "'',,,0",
            "AAAAA,target,0 1 2 3 4,5", "OAAAA,target,1 2 3 4,4", "AAOAA,target,0 1 3 4,4",
            "AAAAO,target,0 1 2 3,4", "OOOOO,target,,0", "AOFAO,target,0 3,3", "OAFOA,target,1 4,3"
    })
    void discoveryPreservesCandidateOrderAndCountsCapacityRejectedGroupMembers(
            String shape, String group, String accepted, int registered) {
        // A: available target member; F: full target member; O: member of another group.
        var expected = accepted == null ? List.<String>of()
                : Arrays.stream(accepted.split(" ")).map(index -> "worker-" + index).toList();
        var inputs = new RouteProjection.Inputs(new QueueSnapshot(0L, true, GroupingPolicy.FIXED_WINDOW,
                Comparator.comparingLong(GroupPlanner.Item::requestId),
                new GroupPlanner.Constraints(1, Long.MAX_VALUE, 0L, 0L, 0L), List.of(), null),
                new WorkSnapshot(0L, List.of(), List.of(), 0L), 0L);
        for (RoleType role : List.of(RoleType.PREFILL, RoleType.PDFUSION)) {
            var directory = mock(EndpointRegistry.class);
            var cache = mock(CacheAwareService.class);
            when(cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of());
            var strategy = new CostBasedPrefillStrategy(directory, cache, mock(EngineHealthReporter.class),
                    mock(CacheMetricsReporter.class), SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()));
            var entries = new ArrayList<EndpointRegistry.PrefillRoutingEntry>();
            for (int index = 0; index < shape.length(); index++) {
                char kind = shape.charAt(index);
                var endpoint = rejectedEndpoint(role, kind == 'O' ? "other" : "target", AdmissionRejectReason.RESOURCE_EXHAUSTED);
                if (kind == 'A') {
                    when(endpoint.ownership.progress()).thenReturn(new PrefillState.Progress(0L, 0L, 0L, 0, 0, 0L));
                    when(endpoint.captureRouteProjectionInputs()).thenReturn(inputs);
                    org.springframework.test.util.ReflectionTestUtils.setField(endpoint, "predictor", new FormulaPredictor("sum(computeTokens)"));
                }
                entries.add(new EndpointRegistry.PrefillRoutingEntry("worker-" + index, endpoint));
            }
            var frozen = List.copyOf(entries);
            when(directory.prefillRoutingSnapshot(role)).thenReturn(frozen);
            var result = StrategyTestSupport.select(strategy, Request.builder().requestId(99L).seqLen(100L).build(),
                    SchedulingMetadata.explicit(50, Long.MAX_VALUE), role, group);

            // The zero KV capacity blocks projection, exposing the registered count even for accepted candidates.
            assertEquals(org.flexlb.balance.PlacementResult.Status.BLOCKED, result.status());
            assertEquals(registered, result.diagnostics().get("workers"));
            assertEquals(entries, frozen);
            verify(directory).prefillRoutingSnapshot(role);
            if (expected.isEmpty()) { verifyNoInteractions(cache); }
            else { verify(cache).findMatchingEngines(any(), eq(role), eq(expected)); }
            for (int index = 0; index < shape.length(); index++) {
                if (shape.charAt(index) == 'O') { verifyNoInteractions(entries.get(index).endpoint().ownership); }
            }
        }
    }

    @Test
    void failureUsesOneScopedFleetAndAggregatesReasonsIndependentlyOfOrder() {
        for (RoleType role : List.of(RoleType.PREFILL, RoleType.PDFUSION)) {
            for (AdmissionRejectReason first : AdmissionRejectReason.values()) {
                for (AdmissionRejectReason second : AdmissionRejectReason.values()) {
                    var directory = mock(EndpointRegistry.class);
                    var cache = mock(CacheAwareService.class);
                    var strategy = new CostBasedPrefillStrategy(directory, cache, mock(EngineHealthReporter.class), org.mockito.Mockito.mock(CacheMetricsReporter.class), SchedulingTestConfig.configService(SchedulingTestConfig.newConfig()));
                    var outside = rejectedEndpoint(role, "other", AdmissionRejectReason.UNSPECIFIED);
                    var firstEndpoint = rejectedEndpoint(role, "target", first);
                    var secondEndpoint = rejectedEndpoint(role, "target", second);
                    when(directory.prefillRoutingSnapshot(role)).thenReturn(List.of(
                            new EndpointRegistry.PrefillRoutingEntry("first", firstEndpoint),
                            new EndpointRegistry.PrefillRoutingEntry("outside", outside),
                            new EndpointRegistry.PrefillRoutingEntry("second", secondEndpoint)));
                    var request = Request.builder()
                            .requestId(1L)
                            .seqLen(100L).build();

                    var result = StrategyTestSupport.select(strategy, request, SchedulingMetadata.explicit(50, Long.MAX_VALUE), role, "target");
                    var failure = result.failure();

                    // Missing provenance dominates; mixed known blockers are capacity, not arbitrary priority.
                    int expectedCode = first == AdmissionRejectReason.UNSPECIFIED
                            || second == AdmissionRejectReason.UNSPECIFIED ? 8432
                            : first != second || first == AdmissionRejectReason.RESOURCE_EXHAUSTED ? 8431 : 8430;
                    assertEquals(expectedCode, failure.getCode(), role + " " + first + " " + second);
                    assertEquals(2, result.diagnostics().get("workers"));
                    verify(directory).prefillRoutingSnapshot(role);
                    verify(outside.ownership, never()).admissionSummary(anyInt(), org.mockito.ArgumentMatchers.anyLong());
                    verifyNoInteractions(cache);
                }
            }
        }
    }

    private static PrefillEndpoint rejectedEndpoint(RoleType role, String group, AdmissionRejectReason reason) {
        var endpoint = mock(PrefillEndpoint.class);
        org.springframework.test.util.ReflectionTestUtils.setField(endpoint, "ownership", mock(PrefillState.class));
        org.springframework.test.util.ReflectionTestUtils.setField(endpoint, "requestLimit", 1L);
        when(endpoint.ownership.progress()).thenReturn(new PrefillState.Progress(0L, 0L, 0L, 0, 0, 1L));
        when(endpoint.getStatus()).thenReturn(
                WorkerStatus.createDiscovered(role, group, "worker", 8080, 8090, "test"));
        when(endpoint.ownership.admissionSummary(50, endpoint.requestLimit)).thenReturn(switch (reason) {
            case HIGHER_PRIORITY_AHEAD -> new PrefillState.AdmissionSummary(1, 1, 0, 0, 1, 0);
            case SAME_PRIORITY_AHEAD -> new PrefillState.AdmissionSummary(1, 1, 0, 1, 0, 0);
            case UNSPECIFIED -> new PrefillState.AdmissionSummary(1, 1, 0, 0, 0, 1);
            case RESOURCE_EXHAUSTED -> new PrefillState.AdmissionSummary(0, 0, 0, 0, 0, 0);
        });
        return endpoint;
    }

    @Test
    void everyRegisteredEndpointMustProveTheSameBlocker() {
        assertNull(CostBasedPrefillStrategy.provenPoolWideBlocker(
                Map.of(RoleType.DECODE, 3), 4));
        assertNull(CostBasedPrefillStrategy.provenPoolWideBlocker(
                Map.of(RoleType.DECODE, 2, RoleType.PREFILL, 2), 4));
        assertEquals(
                RoleType.DECODE,
                CostBasedPrefillStrategy.provenPoolWideBlocker(
                        Map.of(RoleType.DECODE, 4), 4));
    }
}
