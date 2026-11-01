package org.flexlb.balance.strategy;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.planner.BuiltinGroupPlanner;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.projection.QueueSnapshot;
import org.flexlb.balance.projection.RequestTimeEstimation;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.scheduler.PlacementKey;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

/** Observable selection contracts, independent of candidate storage and discovery helpers. */
class PrefillSelectionContractTest {
    private ConfigService service;
    private EndpointRegistry registry;
    private CacheAwareService cache;
    private CacheMetricsReporter metrics;
    private StrategyTestSupport.PrefillRoutingFixture strategy;
    private Request request;
    private final SchedulingMetadata metadata = SchedulingMetadata.explicit(50, Long.MAX_VALUE);

    @BeforeEach
    void setUp() {
        var config = StrategyTestSupport.config();
        config.setScheduler(ConfigTestSupport.direct());
        config.setDispatcher(ConfigTestSupport.nonBatch());
        config.getRouter()
                .getRoles()
                .getPrefill()
                .getExecutionTimeEstimator()
                .setExpression("sum(computeTokens)");
        service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        registry = StrategyTestSupport.endpointRegistry(service);
        cache = mock(CacheAwareService.class);
        when(cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of());
        metrics = mock(CacheMetricsReporter.class);
        strategy = strategy(registry);
        request =
                Request.builder()
                        .requestId(91001L)
                        .seqLen(1000L)
                        .blockCacheKeys(List.of(1L))
                        .build();
    }

    @AfterEach
    void tearDown() {
        registry.close();
    }

    @Test
    void policyReturnsObservedAdmissionBlockWithoutCallerInterpretation() {
        var endpoint = publish(registry, RoleType.PREFILL);
        var queue =
                new QueueSnapshot(
                        1000L,
                        true,
                        BuiltinGroupPlanner.SINGLE,
                        Comparator.comparingLong(GroupPlanner.Item::enqueueSeq),
                        new GroupPlanner.Constraints(1, 10000L, 10000L, 0L, 0L),
                        List.of(new GroupPlanner.Item(1L, 50, 1L, 1000L, Long.MAX_VALUE, 10L, 0L)),
                        new QueueSnapshot.AdmissionBlock(
                                1L,
                                1L,
                                new RequestTimeEstimation.AdmissionBlockSemantics(
                                        "PREFILL_CAPACITY",
                                        RequestTimeEstimation.AfterProbeAdmission.BLOCKED,
                                        "PREFILL_CAPACITY")));
        var input = mock(PrefillSelectionStrategy.CandidateInput.class);
        when(input.address()).thenReturn(endpoint.ipPort());
        when(input.group()).thenReturn("g");
        when(input.projectionInputs())
                .thenReturn(
                        new RequestTimeEstimation.Inputs(
                                queue,
                                new WorkSnapshot(1000L, List.of(), List.of(), 0L),
                                0L,
                                endpoint.predictor.evaluator(),
                                endpoint.deliveryMode));

        var result =
                new CostBasedPrefillStrategy(cache, service)
                        .select(
                                request,
                                metadata,
                                RoleType.PREFILL,
                                "g",
                                List.of(input),
                                () -> 1000L);

        assertEquals(PlacementResult.Status.BLOCKED, result.status());
        assertEquals(
                PlacementKey.exact(RoleType.PREFILL, "g", endpoint.ipPort()), result.blocker());
        assertEquals(
                StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(), result.failure().getCode());
        assertEquals(
                Map.of(
                        "reason",
                        "PREFILL_CAPACITY",
                        "selectedEndpoint",
                        endpoint.ipPort(),
                        "cacheHitTokens",
                        0L,
                        "affinityReason",
                        ""),
                result.diagnostics().get("details"));
        assertNull(result.value());
    }

    @Test
    void cacheLookupDelayDoesNotKeepExpiredWorkInTheProjection() {
        var endpoint = publish(registry, RoleType.PREFILL);
        var clock = new AtomicLong(1000L);
        var queue =
                new QueueSnapshot(
                        1000L,
                        true,
                        BuiltinGroupPlanner.SINGLE,
                        Comparator.comparingLong(GroupPlanner.Item::enqueueSeq),
                        new GroupPlanner.Constraints(1, 1000L, 1000L, 0L, 0L),
                        List.of(new GroupPlanner.Item(1L, 50, 1L, 1000L, 1500L, 10L, 0L)),
                        null);
        var snapshot =
                new RequestTimeEstimation.Inputs(
                        queue,
                        new WorkSnapshot(1000L, List.of(), List.of(), 0L),
                        0L,
                        endpoint.predictor.evaluator(),
                        endpoint.deliveryMode);
        var candidate = mock(PrefillSelectionStrategy.CandidateInput.class);
        when(candidate.address()).thenReturn(endpoint.ipPort());
        when(candidate.projectionInputs()).thenReturn(snapshot);
        var inputs = List.of(candidate);
        when(cache.findMatchingEngines(any(), any(), any()))
                .thenAnswer(
                        call -> {
                            clock.set(2000L);
                            return Map.of();
                        });

        var selected =
                new CostBasedPrefillStrategy(cache, service)
                        .select(
                                request.toBuilder().seqLen(20L).build(),
                                metadata,
                                RoleType.PREFILL,
                                null,
                                inputs,
                                clock::get)
                        .value();

        assertEquals(20L, selected.projectedTtftMs());
        assertEquals(20L, selected.prefillWorkMs());
    }

    @Test
    void filteredCandidatesKeepTheWinnersIdentityWithoutReadingEveryGeneration() {
        var endpoint = publish(registry, RoleType.PREFILL);
        var inputs = new java.util.ArrayList<PrefillSelectionStrategy.CandidateInput>();
        for (int index = 0; index < 4; index++) {
            var candidate = mock(PrefillSelectionStrategy.CandidateInput.class);
            when(candidate.address()).thenReturn("worker-" + index);
            when(candidate.group()).thenReturn(index == 0 ? "other" : "g");
            when(candidate.generationId()).thenReturn(100L + index);
            when(candidate.projectionInputs())
                    .thenAnswer(call -> endpoint.captureRouteProjectionInputs());
            inputs.add(candidate);
        }
        // Index 0 is outside the group; index 1 lacks a predictor. Rank compacts 2 and 3.
        var snapshot = endpoint.captureRouteProjectionInputs();
        when(inputs.get(1).projectionInputs())
                .thenReturn(
                        new RequestTimeEstimation.Inputs(
                                snapshot.queue(),
                                snapshot.work(),
                                snapshot.ownershipVersion(),
                                null,
                                snapshot.deliveryMode()));
        var policy = new CostBasedPrefillStrategy(cache, service);
        var result = policy.select(request, metadata, RoleType.PREFILL, "g", inputs, () -> 1000L);

        assertEquals(PlacementResult.Status.SUCCESS, result.status());
        var selected = result.value();
        int winner = inputs.indexOf(selected.input());
        assertTrue(winner >= 2);
        assertSame(inputs.get(winner), selected.input());
        assertEquals("worker-" + winner, selected.address());
        assertEquals(100L + winner, selected.generationId());
        for (int index = 0; index < inputs.size(); index++) {
            verify(inputs.get(index), org.mockito.Mockito.times(index == winner ? 1 : 0))
                    .generationId();
        }
    }

    @Test
    void reusedScratchPreservesEachInputTypeAndPreviouslySelectedIdentity() {
        var endpoint = publish(registry, RoleType.PREFILL);
        var entry = new EndpointRegistry.PrefillRoutingEntry(endpoint.ipPort(), endpoint);
        var synthetic = mock(PrefillSelectionStrategy.CandidateInput.class);
        when(synthetic.address()).thenReturn("synthetic");
        when(synthetic.projectionInputs()).thenReturn(endpoint.captureRouteProjectionInputs());
        var policy = new CostBasedPrefillStrategy(cache, service);

        var first =
                policy.select(
                                request,
                                metadata,
                                RoleType.PREFILL,
                                null,
                                List.of(entry),
                                () -> 1000L)
                        .value();
        var second =
                policy.select(
                                request,
                                metadata,
                                RoleType.PREFILL,
                                null,
                                List.of(synthetic),
                                () -> 1000L)
                        .value();
        var third =
                policy.select(
                                request,
                                metadata,
                                RoleType.PREFILL,
                                null,
                                List.of(entry),
                                () -> 1000L)
                        .value();

        assertSame(endpoint, first.input().endpoint());
        assertSame(synthetic, second.input());
        assertSame(entry, third.input());
        assertSame(entry, first.input());
    }

    @Test
    void policyResultRetainsOnlyItsOwnScalarPredictionAcrossTheNextProjection() {
        var endpoint = publish(registry, RoleType.PREFILL);
        var candidate = mock(PrefillSelectionStrategy.CandidateInput.class);
        when(candidate.address()).thenReturn(endpoint.ipPort());
        when(candidate.group()).thenReturn("g");
        when(candidate.generationId()).thenReturn(endpoint.getStatus().getGenerationId());
        when(candidate.projectionInputs())
                .thenAnswer(call -> endpoint.captureRouteProjectionInputs());
        var inputs = List.of(candidate);
        when(cache.findMatchingEngines(any(), any(), any()))
                .thenReturn(Map.of(endpoint.ipPort(), 1), Map.of(endpoint.ipPort(), 2));
        var policy = new CostBasedPrefillStrategy(cache, service);
        var firstRequest = request.toBuilder().cacheKeyBlockSize(100L).build();
        var first =
                policy.select(firstRequest, metadata, RoleType.PREFILL, "g", inputs, () -> 1000L)
                        .value();
        var second =
                policy.select(
                                firstRequest.toBuilder().requestId(91002L).seqLen(2000L).build(),
                                metadata,
                                RoleType.PREFILL,
                                "g",
                                inputs,
                                () -> 2000L)
                        .value();
        assertEquals(100L, first.cacheHitTokens());
        assertEquals(900L, first.prefillWorkMs());
        assertEquals(200L, second.cacheHitTokens());
        assertEquals(1800L, second.prefillWorkMs());
        assertNotSame(first, second);
    }

    @ParameterizedTest
    @CsvSource({
        "100,64,1000,5,500,500",
        "0,64,1000,5,320,320",
        "-1,64,1000,16,936,1000",
        "0,0,1000,5,0,0",
        "0,-1,1000,5,0,0",
        "100,64,0,5,0,0",
        "0,9223372036854775807,1000,2147483647,0,1000"
    })
    void cacheHitsUseRequestBlockSizeBeforeWorkerFallback(
            long requestBlock,
            long workerBlock,
            long tokens,
            int prefix,
            long effectiveHit,
            long routingHit) {
        var endpoint = publish(registry, RoleType.PREFILL);
        var status = endpoint.getStatus();
        status.cacheMutationLock.lock();
        try {
            status.publishCacheStatus(CacheStatus.builder().blockSize(workerBlock).build());
        } finally {
            status.cacheMutationLock.unlock();
        }
        request = request.toBuilder().seqLen(tokens).cacheKeyBlockSize(requestBlock).build();
        when(cache.findMatchingEngines(any(), any(), any()))
                .thenReturn(Map.of(endpoint.ipPort(), prefix));

        try (var route = StrategyTestSupport.scope(request)) {
            var result = strategy.select(request, metadata, RoleType.PREFILL, "g", route);
            assertEquals(PlacementResult.Status.SUCCESS, result.status());
            assertEquals(effectiveHit, route.prefill().getDebugInfo().getHitCacheLen());
            assertEquals(Math.max(0L, tokens - effectiveHit), route.prefillWorkMs());
        }
        verify(metrics)
                .reportSelectionCache(
                        RoleType.PREFILL, tokens, effectiveHit, routingHit, routingHit);
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void absentCacheResultsPreserveUncachedSelection(boolean nullResults) {
        publish(registry, RoleType.PREFILL);
        request = request.toBuilder().cacheKeyBlockSize(100L).build();
        when(cache.findMatchingEngines(any(), any(), any()))
                .thenReturn(nullResults ? null : Map.of());
        try (var route = StrategyTestSupport.scope(request)) {
            var result = strategy.select(request, metadata, RoleType.PREFILL, "g", route);
            assertEquals(PlacementResult.Status.SUCCESS, result.status());
            assertEquals(0L, route.prefill().getDebugInfo().getHitCacheLen());
            assertEquals(1000L, route.prefillWorkMs());
        }
    }

    @ParameterizedTest
    @CsvSource({"PREFILL,0", "PREFILL,2", "PDFUSION,0", "PDFUSION,2"})
    void emptyAndUnmodeledPoolsKeepTheirFailureAndDiagnosticScope(RoleType role, int workers) {
        for (int index = 0; index < workers; index++) {
            var source =
                    StrategyTestSupport.workerStatus(
                            role, "g", "10.0.1." + index, 8080, 8081, true, 100000L, 100000L);
            var endpoint =
                    StrategyTestSupport.publishEndpoint(registry, role, source.getIpPort(), source);
            ReflectionTestUtils.setField(endpoint, "predictor", null);
        }
        long started = System.currentTimeMillis();
        try (var route = StrategyTestSupport.scope(request)) {
            var result = strategy.select(request, metadata, role, "g", route);
            assertEquals(PlacementResult.Status.BLOCKED, result.status());
            assertEquals(new PlacementKey(role, "g", null), result.blocker());
            assertEquals(
                    (workers == 0 ? role.getErrorType() : StrategyErrorType.RESOURCE_EXHAUSTED)
                            .getErrorCode(),
                    result.failure().getCode());
            assertEquals(role.name(), result.diagnostics().get("role"));
            assertEquals(workers, result.diagnostics().get("workers"));
            long observed = (long) result.diagnostics().get("observedAtMs");
            assertTrue(observed >= started && observed <= System.currentTimeMillis());
            assertEquals(
                    workers == 0
                            ? Map.of()
                            : Map.of("PROJECTION_UNAVAILABLE_PREDICTOR_MISSING", workers),
                    result.diagnostics().get("details"));
            assertNull(route.prefill());
        }
        if (workers == 0) verifyNoInteractions(cache);
    }

    @ParameterizedTest
    @CsvSource({
        "1,0,0,1,0,PRIORITY_ADMISSION_REJECTED,HIGHER_PRIORITY_AHEAD",
        "1,0,1,0,0,PRIORITY_ADMISSION_REJECTED,SAME_PRIORITY_AHEAD",
        "1,0,0,0,1,ADMISSION_UNAVAILABLE,UNSPECIFIED",
        "1,1,0,0,0,RESOURCE_EXHAUSTED,RESOURCE_EXHAUSTED",
        "2,0,1,0,1,ADMISSION_UNAVAILABLE,UNSPECIFIED"
    })
    void capacityBlockKeepsItsReasonAndClosesTheSelectedGeneration(
            long release,
            long lower,
            long same,
            long higher,
            long unknown,
            StrategyErrorType error,
            AdmissionRejectReason reason) {
        var endpoint = publish(registry, RoleType.PREFILL);
        var ownership = spy(endpoint.ownership);
        ReflectionTestUtils.setField(endpoint, "ownership", ownership);
        doReturn(false).when(ownership).requestCapacityAvailable(endpoint.requestLimit, 1);
        doReturn(
                        new PrefillState.AdmissionSummary(
                                lower + same + higher + unknown,
                                release,
                                lower,
                                same,
                                higher,
                                unknown))
                .when(ownership)
                .admissionSummary(metadata.priority(), endpoint.requestLimit);
        var pin = endpoint.tryPinGeneration();
        var directory = spy(registry);
        doReturn(pin).when(directory).capture(RoleType.PREFILL, endpoint.ipPort());
        strategy = strategy(directory);
        try (pin;
                var route = StrategyTestSupport.scope(request)) {
            var result = strategy.select(request, metadata, RoleType.PREFILL, "g", route);
            assertEquals(PlacementResult.Status.BLOCKED, result.status());
            assertEquals(error.getErrorCode(), result.failure().getCode());
            assertEquals(reason, result.failure().getAdmissionRejectReason());
            assertEquals(
                    PlacementKey.exact(RoleType.PREFILL, "g", endpoint.ipPort()), result.blocker());
            assertEquals(1, result.diagnostics().get("workers"));
            assertEquals(
                    Map.of(
                            "reason",
                            "PREFILL_CAPACITY",
                            "selectedEndpoint",
                            endpoint.ipPort(),
                            "cacheHitTokens",
                            0L),
                    result.diagnostics().get("details"));
            assertFalse(pin.isOpen());
            assertNull(route.prefill());
        }
    }

    @Test
    void sameAddressReplacementCannotReceiveTheOldGenerationsPrediction() {
        var original = publish(registry, RoleType.PREFILL);
        var replacementRegistry = StrategyTestSupport.endpointRegistry(service);
        try {
            var replacement = publish(replacementRegistry, RoleType.PREFILL);
            assertEquals(original.ipPort(), replacement.ipPort());
            assertNotSame(original, replacement);
            var pin = replacement.tryPinGeneration();
            var directory = spy(registry);
            doReturn(pin).when(directory).capture(RoleType.PREFILL, original.ipPort());
            strategy = strategy(directory);
            try (pin;
                    var route = StrategyTestSupport.scope(request)) {
                var result = strategy.select(request, metadata, RoleType.PREFILL, "g", route);
                assertEquals(PlacementResult.Status.BLOCKED, result.status());
                assertEquals(new PlacementKey(RoleType.PREFILL, "g", null), result.blocker());
                assertNull(result.failure());
                assertNull(route.prefill());
                assertFalse(pin.isOpen());
            }
        } finally {
            replacementRegistry.close();
        }
    }

    private StrategyTestSupport.PrefillRoutingFixture strategy(EndpointRegistry directory) {
        return new StrategyTestSupport.PrefillRoutingFixture(
                directory, cache, mock(EngineHealthReporter.class), metrics, service);
    }

    private PrefillEndpoint publish(EndpointRegistry directory, RoleType role) {
        var source =
                StrategyTestSupport.workerStatus(
                        role, "g", "10.0.0.1", 8080, 8081, true, 100000L, 100000L);
        return (PrefillEndpoint)
                StrategyTestSupport.publishEndpoint(directory, role, source.getIpPort(), source);
    }
}
