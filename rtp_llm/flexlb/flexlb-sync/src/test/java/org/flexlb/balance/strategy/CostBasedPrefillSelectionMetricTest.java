package org.flexlb.balance.strategy;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.projection.RequestTimeEstimation;
import org.flexlb.balance.scheduler.RequestContext;
import org.flexlb.balance.scheduler.RequestRoute;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.PreemptionConfig;
import org.flexlb.config.QueueOrderingConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusTestSupport;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.ArgumentCaptor;
import org.mockito.Mockito;

import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Checks Prefill selection observations and metric failures without changing valid selection
 * results.
 */
class CostBasedPrefillSelectionMetricTest {
    private static final int FULL_FLEET_SIZE = 750;

    private FlexlbConfig config;
    private EndpointRegistry registry;
    private CacheAwareService cache;
    private EngineHealthReporter reporter;
    private CacheMetricsReporter cacheReporter;
    private StrategyTestSupport.PrefillRoutingFixture strategy;
    private RequestContext context;

    @BeforeEach
    void setUp() {
        config = StrategyTestSupport.config();
        config.setDispatcher(ConfigTestSupport.nonBatch());
        config.getRouter()
                .getRoles()
                .getPrefill()
                .getExecutionTimeEstimator()
                .setExpression("sum(computeTokens)");
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        registry = StrategyTestSupport.endpointRegistry(configService);
        publish("10.0.0.1", 8080);

        cache = mock(CacheAwareService.class);
        when(cache.findMatchingEngines(any())).thenReturn(localMatches(Map.of()));
        reporter = mock(EngineHealthReporter.class);
        cacheReporter = mock(CacheMetricsReporter.class);
        strategy =
                new StrategyTestSupport.PrefillRoutingFixture(
                        registry,
                        cache,
                        reporter,
                        cacheReporter,
                        SchedulingTestConfig.configService(config));

        Request request =
                Request.builder()
                        .requestId(String.valueOf(10_001L))
                        .seqLen(1_000L)
                        .rawPriority(50)
                        .blockCacheKeys(List.of())
                        .build();
        context = new RequestContext(config);
        context.data().setRequest(request);
        context.data()
                .setSchedulingMetadata(
                        SchedulingMetadata.explicit(50, System.currentTimeMillis() + 60_000L));
    }

    @AfterEach
    void tearDown() {
        try {
            registry.close();
        } finally {
            candidateWorkspace().remove();
        }
    }

    @Test
    void selectionReleasesWorkspaceWithoutChangingItsOwnedResult() {
        PrefillCandidateSet workspace = candidateWorkspace().get();

        try (RequestRoute selected = select()) {
            assertReleased(workspace);
            assertSame(workspace, candidateWorkspace().get());
            assertTrue(selected.prefill().isSuccess());
            assertEquals("10.0.0.1", selected.prefill().getServerIp());
            assertTrue(selected.prefillWorkMs() > 0L);
        }
    }

    @ParameterizedTest
    @ValueSource(longs = {0L, 20L, 300L})
    void firstQueueSelectionUsesTheSameWorkCostAsStartedQueues(long windowMs) {
        registry.close();
        config.decisionPolicy().setMaxCollectionWaitMs(windowMs);
        config.decisionPolicy().setMaxRequests(8);
        config.getRouter().getRoles().getPrefill().setCacheAffinity(null);
        registry = StrategyTestSupport.endpointRegistry(SchedulingTestConfig.configService(config));
        var estimator = config.getRouter().getRoles().getPrefill().getExecutionTimeEstimator();
        estimator.setExpression("10");
        publish("10.0.0.1", 8080);
        estimator.setExpression("5");
        publish("10.0.0.2", 8080);
        var first = (PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0.1:8080");
        var second = (PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0.2:8080");
        var settings =
                org.flexlb.balance.scheduler.QueueExecutionSettings.capture(
                        config, org.flexlb.balance.planner.BuiltinGroupPlanner.from(config));
        var before = first.captureRouteProjectionInputs();
        assertTrue(before.queue().queueScheduling(), "QUEUE policy exists before its first route");
        assertEquals(8, before.queue().constraints().maxRequests());
        assertEquals(windowMs, before.queue().constraints().collectionWindowMs());
        assertNull(org.flexlb.balance.endpoint.EndpointTestSupport.requestExecution(first));
        org.flexlb.balance.endpoint.EndpointTestSupport.enableQueueRuntime(
                second,
                settings,
                org.flexlb.balance.scheduler.SchedulerTestSupport.queueDependencies(
                        mock(org.flexlb.balance.scheduler.RequestLifecycle.class), config),
                StrategyTestSupport.parkedDelivery());
        strategy =
                new StrategyTestSupport.PrefillRoutingFixture(
                        registry,
                        cache,
                        reporter,
                        cacheReporter,
                        SchedulingTestConfig.configService(config));

        for (boolean startFirst : new boolean[] {false, true}) {
            if (startFirst) {
                org.flexlb.balance.endpoint.EndpointTestSupport.enableQueueRuntime(
                        first,
                        settings,
                        org.flexlb.balance.scheduler.SchedulerTestSupport.queueDependencies(
                                mock(org.flexlb.balance.scheduler.RequestLifecycle.class), config),
                        StrategyTestSupport.parkedDelivery());
            }
            try (RequestRoute selected = select()) {
                assertEquals("10.0.0.2", selected.prefill().getServerIp());
                assertEquals(5L, selected.prefill().getPrefillTime());
            }
        }
    }

    @Test
    void lostGenerationCaptureReleasesWorkspace() {
        PrefillCandidateSet workspace = candidateWorkspace().get();
        EndpointRegistry directory = Mockito.spy(registry);
        Mockito.doReturn(null).when(directory).capture(RoleType.PREFILL, "10.0.0.1:8080");
        strategy =
                new StrategyTestSupport.PrefillRoutingFixture(
                        directory,
                        cache,
                        reporter,
                        cacheReporter,
                        SchedulingTestConfig.configService(config));

        var result =
                StrategyTestSupport.select(
                        strategy,
                        context.data().getRequest(),
                        context.data().getSchedulingMetadata(),
                        RoleType.PREFILL,
                        null);

        assertEquals(PlacementResult.Status.ENDPOINT_CHANGED, result.status());
        assertEquals(PlacementResult.Status.SUCCESS, result.selection().status());
        assertReleased(workspace);
    }

    @Test
    void capacityCheckFailureDoesNotAcquireAGeneration() {
        config.setScheduler(ConfigTestSupport.direct());
        refreshContext();
        var endpoint =
                Mockito.spy((PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0.1:8080"));
        var ownership = endpoint;
        org.flexlb.balance.endpoint.EndpointTestSupport.stubResources(endpoint, ownership);
        var failure = new IllegalStateException("capacity unavailable");
        Mockito.doThrow(failure)
                .when(ownership)
                .requestCapacityAvailable(Mockito.anyLong(), Mockito.anyInt());
        var pin = endpoint.tryPinGeneration();
        var directory = Mockito.spy(registry);
        Mockito.doReturn(
                        java.util.List.of(
                                new EndpointRegistry.PrefillRoutingEntry(
                                        "10.0.0.1:8080", endpoint)))
                .when(directory)
                .prefillRoutingSnapshot(RoleType.PREFILL);
        Mockito.doReturn(pin).when(directory).capture(RoleType.PREFILL, "10.0.0.1:8080");
        strategy =
                new StrategyTestSupport.PrefillRoutingFixture(
                        directory,
                        cache,
                        reporter,
                        cacheReporter,
                        SchedulingTestConfig.configService(config));
        try {
            assertSame(failure, assertThrows(IllegalStateException.class, this::select));
            Mockito.verify(directory, Mockito.never()).capture(Mockito.any(), Mockito.any());
            assertTrue(
                    pin.isOpen(), "selection only reads capacity; no generation has been acquired");
            assertReleased(candidateWorkspace().get());
        } finally {
            pin.close();
        }
    }

    @Test
    void projectionFailureReleasesCandidatesAlreadyEvaluated() {
        PrefillCandidateSet workspace = candidateWorkspace().get();
        publish("10.0.0.2", 8080);
        PrefillEndpoint failing =
                Mockito.spy((PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0.2:8080"));
        IllegalStateException failure = new IllegalStateException("projection unavailable");
        Mockito.doAnswer(
                        invocation -> {
                            assertEquals(
                                    1,
                                    workspace.size,
                                    "the first endpoint must already be captured");
                            throw failure;
                        })
                .when(failing)
                .captureRouteProjectionInputs();
        EndpointRegistry directory = Mockito.spy(registry);
        Mockito.doReturn(
                        List.of(
                                registry.prefillRoutingSnapshot(RoleType.PREFILL).getFirst(),
                                new EndpointRegistry.PrefillRoutingEntry("10.0.0.2:8080", failing)))
                .when(directory)
                .prefillRoutingSnapshot(RoleType.PREFILL);
        strategy =
                new StrategyTestSupport.PrefillRoutingFixture(
                        directory,
                        cache,
                        reporter,
                        cacheReporter,
                        SchedulingTestConfig.configService(config));

        assertSame(failure, assertThrows(IllegalStateException.class, this::select));

        assertReleased(workspace);
        assertSame(workspace, candidateWorkspace().get());
    }

    @ParameterizedTest
    @CsvSource({"4096,true", "4097,false"})
    void workspaceRetentionIsBoundedByItsColumnCapacity(int workers, boolean retained) {
        PrefillCandidateSet workspace = candidateWorkspace().get();
        var originalDirectory = registry.prefillRoutingSnapshot(RoleType.PREFILL);
        EndpointRegistry directory = Mockito.spy(registry);
        Mockito.doReturn(Collections.nCopies(workers, originalDirectory.getFirst()))
                .when(directory)
                .prefillRoutingSnapshot(RoleType.PREFILL);
        strategy =
                new StrategyTestSupport.PrefillRoutingFixture(
                        directory,
                        cache,
                        reporter,
                        cacheReporter,
                        SchedulingTestConfig.configService(config));

        try (RequestRoute selected = select()) {
            assertEquals("10.0.0.1", selected.prefill().getServerIp());
        }

        assertTrue(workspace.capturedSources.length >= workers);
        assertReleased(workspace);
        PrefillCandidateSet nextWorkspace = candidateWorkspace().get();
        if (retained) {
            assertSame(workspace, nextWorkspace);
        } else {
            assertNotSame(workspace, nextWorkspace);
            assertEquals(0, nextWorkspace.capturedSources.length);
        }
        Mockito.doReturn(originalDirectory)
                .when(directory)
                .prefillRoutingSnapshot(RoleType.PREFILL);
        try (RequestRoute selected = select()) {
            assertEquals("10.0.0.1", selected.prefill().getServerIp());
        }
        assertReleased(nextWorkspace);
    }

    @Test
    void reentrantTelemetryUsesDetachedSelectionAfterWorkspaceRelease() {
        PrefillCandidateSet workspace = candidateWorkspace().get();
        AtomicBoolean reentered = new AtomicBoolean();
        Mockito.doAnswer(
                        invocation -> {
                            if (reentered.compareAndSet(false, true)) {
                                assertReleased(workspace);
                                try (RequestRoute selected = select()) {
                                    assertEquals("10.0.0.1", selected.prefill().getServerIp());
                                }
                                assertEquals(1, workspace.size);
                                assertFalse(workspace.inUse);
                            }
                            return null;
                        })
                .when(reporter)
                .reportPrefillSelectedEstimates(
                        any(), any(), any(), Mockito.anyLong(), Mockito.anyLong());

        try (RequestRoute selected = select()) {
            assertEquals("10.0.0.1", selected.prefill().getServerIp());
        }

        assertTrue(reentered.get());
        assertReleased(workspace);
        assertSame(workspace, candidateWorkspace().get());
    }

    @SuppressWarnings("unchecked")
    private static ThreadLocal<PrefillCandidateSet> candidateWorkspace() {
        return (ThreadLocal<PrefillCandidateSet>)
                org.springframework.test.util.ReflectionTestUtils.getField(
                        CostBasedPrefillStrategy.class, "CANDIDATES");
    }

    @Test
    void cacheLookupReentryCannotChangeTheOuterFleet() {
        publish("10.0.0.2", 8080);
        AtomicBoolean reentered = new AtomicBoolean();
        when(cache.findMatchingEngines(any()))
                .thenAnswer(
                        invocation -> {
                            var query =
                                    (org.flexlb.cache.domain.CacheMatchQuery)
                                            invocation.getArgument(0);
                            var keys = List.copyOf(query.blockCacheKeys());
                            assertEquals(keys, query.blockCacheKeys());
                            if (reentered.compareAndSet(false, true)) {
                                try (RequestRoute inner = select()) {
                                    assertTrue(inner.prefill().isSuccess());
                                }
                                assertEquals(keys, query.blockCacheKeys());
                            }
                            return localMatches(Map.of());
                        });
        try (RequestRoute outer = select()) {
            assertTrue(outer.prefill().isSuccess());
        }
        assertTrue(reentered.get());
        assertReleased(candidateWorkspace().get());
    }

    @Test
    void cacheLookupFailureReleasesTheEntireCapturedFleet() {
        publish("10.0.0.2", 8080);
        var failure = new IllegalStateException("cache unavailable");
        when(cache.findMatchingEngines(any())).thenThrow(failure);
        assertSame(failure, assertThrows(IllegalStateException.class, this::select));
        assertReleased(candidateWorkspace().get());
    }

    @Test
    void rejectedCandidatesDoNotOverwriteUnvisitedWorkersOrRetainTheTail() {
        publish("10.0.0.2", 8080);
        publish("10.0.0.3", 8080);
        for (String address : List.of("10.0.0.1:8080", "10.0.0.3:8080")) {
            org.springframework.test.util.ReflectionTestUtils.setField(
                    registry.get(RoleType.PREFILL, address), "predictor", null);
        }
        try (RequestRoute selected = select()) {
            assertEquals("10.0.0.2", selected.prefill().getServerIp());
        }
        assertReleased(candidateWorkspace().get());
    }

    @Test
    void cacheLookupAndRankingUseTheSameCapturedGroup() {
        publish("10.0.0.2", 8080);
        var first = registry.get(RoleType.PREFILL, "10.0.0.1:8080").getStatus();
        var second = registry.get(RoleType.PREFILL, "10.0.0.2:8080").getStatus();
        java.util.function.BiConsumer<WorkerStatus, String> regroup =
                (status, group) -> {
                    status.lock.lock();
                    try {
                        status.updateDiscoveryLabels("test", group);
                    } finally {
                        status.lock.unlock();
                    }
                };
        regroup.accept(first, "target");
        regroup.accept(second, "other");
        when(cache.findMatchingEngines(any()))
                .thenAnswer(
                        invocation -> {
                            assertEquals(
                                    "target",
                                    ((org.flexlb.cache.domain.CacheMatchQuery)
                                                    invocation.getArgument(0))
                                            .group());
                            regroup.accept(first, "other");
                            regroup.accept(second, "target");
                            return localMatches(Map.of());
                        });
        try (RequestRoute selected =
                StrategyTestSupport.select(
                                strategy,
                                context.data().getRequest(),
                                context.data().getSchedulingMetadata(),
                                RoleType.PREFILL,
                                "target")
                        .value()) {
            assertEquals("10.0.0.1", selected.prefill().getServerIp());
        }
        assertReleased(candidateWorkspace().get());
    }

    @ParameterizedTest
    @CsvSource({
        "100,0,0,0",
        "100,-1,0,0",
        "100,5,500,500",
        "100,10,900,1000",
        "100,15,900,1000",
        "2000,1,0,1000",
        "9223372036854775807,2147483647,0,1000"
    })
    void cacheRankingPreservesFullHitAndOverflowBoundaries(
            long blockSize, int prefix, long effectiveHit, long routingHit) {
        context.data()
                .setRequest(
                        context.data().getRequest().toBuilder()
                                .cacheKeyBlockSize(blockSize)
                                .build());
        when(cache.findMatchingEngines(any()))
                .thenReturn(localMatches(Map.of("10.0.0.1:8080", prefix)));
        try (RequestRoute selected = select()) {
            assertEquals(effectiveHit, selected.prefill().getDebugInfo().getHitCacheLen());
        }
        verify(cacheReporter)
                .reportSelectionCache(
                        RoleType.PREFILL, 1000L, effectiveHit, routingHit, routingHit);
    }

    private static void assertReleased(PrefillCandidateSet workspace) {
        assertEquals(0, workspace.size);
        assertTrue(workspace.addresses.isEmpty());
        assertFalse(workspace.inUse);
        assertEquals(Long.MAX_VALUE, workspace.minimumTtftRank);
        assertFalse(workspace.hasKnownTtft);
        assertEquals(0L, workspace.maximumCacheHit);
        assertEquals(0L, workspace.maximumRoutingCacheMatchTokens);
        for (int index = 0; index < workspace.capturedSources.length; index++) {
            assertNull(workspace.capturedSources[index]);
            assertNull(workspace.endpointAddresses[index]);
            assertNull(workspace.admissionBlocks[index]);
        }
    }

    @Test
    void selectionDoesNotOwnTheRequestDeadline() {
        context.data()
                .setSchedulingMetadata(
                        SchedulingMetadata.explicit(50, System.currentTimeMillis() - 1L));

        try (RequestRoute ignored = select()) {
            assertTrue(ignored.prefill().isSuccess());
        }
    }

    @ParameterizedTest
    @CsvSource({
        "5,1,false,SUCCESS",
        "50,1,false,SUCCESS",
        "5,2,false,SUCCESS",
        "50,1,true,SUCCESS",
        "10,1,true,SUCCESS",
        "5,1,true,SUCCESS"
    })
    void waitingOwnersDoNotConsumeDeliveryCapacityOrRequirePreemption(
            int incomingPriority,
            int requestLimit,
            boolean preemptQueued,
            PlacementResult.Status expectedStatus) {
        registry.close();
        config.queueScheduler().getDecision().setMaxRequests(2);
        config.getDispatcher().setMaxInflightPerPrefillWorker(requestLimit);
        config.queueScheduler().getDecision().setMaxCollectionWaitMs(60_000L);
        QueueOrderingConfig ordering = ConfigTestSupport.priority();
        ordering.setPreemption(null);
        if (preemptQueued) {
            PreemptionConfig preemption = new PreemptionConfig();
            preemption.setAllowedVictimStages(Set.of(VictimStage.PREFILL_QUEUED));
            ordering.setPreemption(preemption);
        }
        config.queueScheduler().setOrdering(ordering);
        refreshContext();
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        registry = StrategyTestSupport.endpointRegistry(service);
        publish("10.0.0.1", 8080);
        strategy =
                new StrategyTestSupport.PrefillRoutingFixture(
                        registry,
                        cache,
                        reporter,
                        cacheReporter,
                        SchedulingTestConfig.configService(config));
        PrefillEndpoint endpoint =
                (PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0.1:8080");
        Request queuedRequest =
                Request.builder()
                        .requestId(String.valueOf(9_001L))
                        .seqLen(1_000L)
                        .rawPriority(10)
                        .build();
        RequestContext queuedContext = new RequestContext(config);
        queuedContext.data().setRequest(queuedRequest);
        queuedContext
                .data()
                .setSchedulingMetadata(
                        SchedulingMetadata.explicit(10, System.currentTimeMillis() + 120_000L));
        queuedContext.data().setFuture(new CompletableFuture<>());
        RequestRoute queued =
                SchedulingTestConfig.createRoute(
                        queuedContext,
                        null,
                        null,
                        null,
                        endpoint,
                        null,
                        null,
                        System.currentTimeMillis());
        assertTrue(StrategyTestSupport.offer(endpoint, queued));
        context.data()
                .setSchedulingMetadata(
                        SchedulingMetadata.explicit(
                                incomingPriority, System.currentTimeMillis() + 120_000L));

        PlacementResult<RequestRoute, org.flexlb.balance.scheduler.PlacementKey> result =
                StrategyTestSupport.select(
                        strategy,
                        context.data().getRequest(),
                        context.data().getSchedulingMetadata(),
                        RoleType.PREFILL,
                        null);

        assertEquals(expectedStatus, result.status());
        verify(cache).findMatchingEngines(any());
        if (result.status() == PlacementResult.Status.SUCCESS) {
            try (RequestRoute selected = result.value()) {
                assertEquals("10.0.0.1", selected.prefill().getServerIp());
                assertTrue(selected.prefillWorkMs() > 0L);
            }
        }
        assertEquals(
                1,
                endpoint.captureRouteProjectionInputs().queue().activeItems().size(),
                "selection must not evict a lower-priority queued request");
    }

    @ParameterizedTest
    @CsvSource({"false,NON_BATCH", "true,BATCH"})
    void everyDeliveryModeReportsItsSelectionEstimates(boolean batchDelivery, String deliveryMode) {
        config.setDispatcher(batchDelivery ? new DispatcherConfig() : ConfigTestSupport.nonBatch());
        refreshContext();

        try (RequestRoute selected = select()) {
            ArgumentCaptor<Long> ttft = ArgumentCaptor.forClass(Long.class);
            ArgumentCaptor<Long> execution = ArgumentCaptor.forClass(Long.class);
            verify(reporter)
                    .reportPrefillSelectedEstimates(
                            Mockito.eq(RoleType.PREFILL),
                            Mockito.eq("10.0.0.1"),
                            Mockito.eq(deliveryMode),
                            ttft.capture(),
                            execution.capture());
            assertEquals(selected.prefill().getPrefillTime(), ttft.getValue());
            assertEquals(selected.prefillWorkMs(), execution.getValue());
        }
    }

    @Test
    void telemetryFailureNeverChangesAValidSelection() {
        doThrow(new IllegalStateException("metrics unavailable"))
                .when(reporter)
                .reportPrefillSelectedEstimates(
                        any(), any(), any(), Mockito.anyLong(), Mockito.anyLong());

        try (RequestRoute selected = select()) {
            assertTrue(selected.prefill().isSuccess());
            assertEquals("10.0.0.1", selected.prefill().getServerIp());
        }
    }

    @Test
    void failedSelectionReportingDoesNotReleaseTheSelectedGenerationBeforeItsOwner() {
        IllegalStateException failure = new IllegalStateException("metrics unavailable");
        doThrow(failure)
                .when(cacheReporter)
                .reportSelectionCache(
                        any(),
                        Mockito.anyLong(),
                        Mockito.anyLong(),
                        Mockito.anyLong(),
                        Mockito.anyLong());
        var endpoint = registry.get(RoleType.PREFILL, "10.0.0.1:8080");
        var pin = Mockito.spy(endpoint.tryPinGeneration());
        var directory = Mockito.spy(registry);
        Mockito.doReturn(pin).when(directory).capture(RoleType.PREFILL, "10.0.0.1:8080");
        strategy =
                new StrategyTestSupport.PrefillRoutingFixture(
                        directory,
                        cache,
                        reporter,
                        cacheReporter,
                        SchedulingTestConfig.configService(config));
        try (var selected = select()) {
            assertTrue(selected.prefill().isSuccess());
            assertTrue(pin.isOpen());
            verify(pin, Mockito.never()).close();
        }
        verify(pin).close();
    }

    @Test
    void cacheLeaderInsideTtftCapOverridesTheBaselineCandidate() {
        configureAffinity(600L, 5.0);

        try (RequestRoute selected = select()) {
            assertEquals("10.0.0.2", selected.prefill().getServerIp());
            verify(cacheReporter)
                    .reportCacheAffinityDecision(RoleType.PREFILL, "10.0.0.2", "CACHE_LEADER");
        }
    }

    @Test
    void fullKvCacheDoesNotHideTheBestPrefixOwner() {
        configureAffinity(600L, 5.0);
        var leader = (PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0.2:8080");
        var status = leader.getStatus();
        var response =
                StrategyTestSupport.response(
                        RoleType.PREFILL,
                        true,
                        0L,
                        1_000_000L,
                        status.appliedStatusCursor().statusVersion() + 1L);
        status.lock.lock();
        try {
            leader.publishStatus(
                    leader.applyPreparedStatus(
                            status,
                            status.prepareNewStatus(
                                    WorkerStatusTestSupport.freezeStatusResponse(
                                            status, response))));
        } finally {
            status.lock.unlock();
        }
        var settings =
                org.flexlb.balance.scheduler.QueueExecutionSettings.capture(
                        config, org.flexlb.balance.planner.BuiltinGroupPlanner.from(config));
        var dependencies =
                org.flexlb.balance.scheduler.SchedulerTestSupport.queueDependencies(
                        mock(org.flexlb.balance.scheduler.RequestLifecycle.class), config);
        org.flexlb.balance.endpoint.EndpointTestSupport.enableQueueRuntime(
                leader, settings, dependencies, StrategyTestSupport.parkedDelivery());
        org.flexlb.balance.endpoint.EndpointTestSupport.enableQueueRuntime(
                (PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0.1:8080"),
                settings,
                dependencies,
                StrategyTestSupport.parkedDelivery());
        assertEquals(
                0L, leader.captureRouteProjectionInputs().queue().constraints().batchKvCapacity());
        try (RequestRoute selected = select()) {
            assertEquals("10.0.0.2", selected.prefill().getServerIp());
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void busyCacheLeaderIsRankedBeforeDeliveryCapacity(boolean direct) {
        if (direct) {
            config.setScheduler(ConfigTestSupport.direct());
        }
        config.getDispatcher().setMaxInflightPerPrefillWorker(1);
        configureAffinity(600L, 5.0);
        PrefillEndpoint leader = (PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0.2:8080");
        WorkerStatus status = leader.getStatus();
        var response =
                StrategyTestSupport.response(
                        RoleType.PREFILL,
                        true,
                        1_000_000L,
                        1_000_000L,
                        status.appliedStatusCursor().statusVersion() + 1L);
        response.setRunningQueryLen(1L);
        status.lock.lock();
        try {
            leader.publishStatus(
                    leader.applyPreparedStatus(
                            status,
                            status.prepareNewStatus(
                                    WorkerStatusTestSupport.freezeStatusResponse(
                                            status, response))));
        } finally {
            status.lock.unlock();
        }
        assertFalse(leader.requestCapacityAvailable(leader.requestLimit, 1));

        var result =
                StrategyTestSupport.select(
                        strategy,
                        context.data().getRequest(),
                        context.data().getSchedulingMetadata(),
                        RoleType.PREFILL,
                        null);
        if (direct) {
            assertEquals(PlacementResult.Status.BLOCKED, result.status());
            assertEquals(
                    org.flexlb.balance.scheduler.PlacementKey.exact(
                            RoleType.PREFILL, null, "10.0.0.2:8080"),
                    result.blocker());
        } else {
            assertEquals(PlacementResult.Status.SUCCESS, result.status());
            try (RequestRoute selected = result.value()) {
                assertEquals("10.0.0.2", selected.prefill().getServerIp());
            }
        }
        verify(cacheReporter)
                .reportCacheAffinityDecision(RoleType.PREFILL, "10.0.0.2", "CACHE_LEADER");
    }

    @Test
    void cacheLeaderAtEndOfFullFleetRemainsEligible() {
        configureAffinity(600L, 5.0);
        String cacheLeader = null;
        for (int index = 2; index < FULL_FLEET_SIZE; index++) {
            String ip = "10.2." + index / 250 + '.' + index % 250;
            publish(ip, 8080);
            cacheLeader = ip;
        }
        when(cache.findMatchingEngines(any()))
                .thenReturn(localMatches(Map.of(cacheLeader + ":8080", 5)));

        try (RequestRoute selected = select()) {
            assertEquals(
                    cacheLeader,
                    selected.prefill().getServerIp(),
                    "cache-first must inspect the complete 750-node fleet");
        }
    }

    @Test
    void exactCostTiesRotateWithoutShrinkingTheCandidateFleet() {
        publish("10.0.0.2", 8080);
        publish("10.0.0.3", 8080);

        Set<String> selectedIps = new HashSet<>();
        for (int index = 0; index < 3; index++) {
            context.data()
                    .setRequest(
                            context.data().getRequest().toBuilder()
                                    .requestId(String.valueOf(30_000L + index))
                                    .build());
            try (RequestRoute selected = select()) {
                selectedIps.add(selected.prefill().getServerIp());
            }
        }

        assertEquals(Set.of("10.0.0.1", "10.0.0.2", "10.0.0.3"), selectedIps);
    }

    @ParameterizedTest
    @CsvSource({"400,5.0,OVER_CAP", "600,60.0,LOW_CACHE_HIT"})
    void cacheAffinityGateUsesTheConfiguredBaselineCandidate(
            long maxExtraTtftMs, double minPrefixHitPercent, String reason) {
        configureAffinity(maxExtraTtftMs, minPrefixHitPercent);

        try (RequestRoute selected = select()) {
            assertEquals("10.0.0.1", selected.prefill().getServerIp());
            verify(cacheReporter).reportCacheAffinityDecision(RoleType.PREFILL, "10.0.0.1", reason);
        }
    }

    @Test
    void equalMaximumCacheHitPreservesBestOnlySelection() {
        configureFocusedAffinity();
        PrefillCandidateSet candidates =
                candidates(new long[] {100L, 105L, 1_000L}, new long[] {500L, 500L, 0L});

        int selectedIndex =
                strategy.selectBestCandidate(candidates, 100L, RoleType.PREFILL, null, 1_000L);

        assertEquals(0, selectedIndex, "equal cache hit must preserve the minimum-TTFT baseline");
        assertEquals("NO_CACHE_LEAD", candidates.affinityReason);
    }

    @Test
    void strictlyGreaterCacheHitOverridesTheBaselineCandidate() {
        configureFocusedAffinity();
        PrefillCandidateSet candidates =
                candidates(new long[] {100L, 105L, 1_000L}, new long[] {500L, 600L, 0L});

        int selectedIndex =
                strategy.selectBestCandidate(candidates, 100L, RoleType.PREFILL, null, 1_000L);

        assertEquals(1, selectedIndex);
        assertEquals("CACHE_LEADER", candidates.affinityReason);
    }

    @Test
    void cacheLeaderRemainsPreferredAcrossRepeatedSelections() {
        configureAffinity(600L, 5.0);

        try (RequestRoute selected = select()) {
            assertEquals("10.0.0.2", selected.prefill().getServerIp());
        }
        context.data()
                .setRequest(
                        context.data().getRequest().toBuilder()
                                .requestId(String.valueOf(20_002L))
                                .build());

        try (RequestRoute selected = select()) {
            assertEquals("10.0.0.2", selected.prefill().getServerIp());
        }
        verify(cacheReporter, times(2))
                .reportCacheAffinityDecision(RoleType.PREFILL, "10.0.0.2", "CACHE_LEADER");
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void unknownEngineTokenDemandDoesNotPreventCountBasedAdmission(boolean missingTokenDemand) {
        var endpoint = (PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0.1:8080");
        var status = endpoint.getStatus();
        var response =
                StrategyTestSupport.response(
                        RoleType.PREFILL,
                        true,
                        1_000_000L,
                        1_000_000L,
                        status.appliedStatusCursor().statusVersion() + 1L);
        var task = new TaskInfo();
        task.setRequestId(String.valueOf(1L));
        task.setPhase(TaskPhase.RUNNING);
        task.setInputLength(missingTokenDemand ? 0L : 100L);
        response.setRunningQueryLen(1L);
        response.setRunningTaskInfo(Map.of("1", task));
        status.lock.lock();
        try {
            endpoint.publishStatus(
                    endpoint.applyPreparedStatus(
                            status,
                            status.prepareNewStatus(
                                    WorkerStatusTestSupport.freezeStatusResponse(
                                            status, response))));
        } finally {
            status.lock.unlock();
        }
        assertTrue(endpoint.captureRouteProjectionInputs().work().hasUnknownWork());

        context.data()
                .setRequest(
                        context.data().getRequest().toBuilder()
                                .requestId(String.valueOf(40_001L))
                                .build());
        var result =
                StrategyTestSupport.select(
                        strategy,
                        context.data().getRequest(),
                        context.data().getSchedulingMetadata(),
                        RoleType.PREFILL,
                        null);
        assertEquals(PlacementResult.Status.SUCCESS, result.status());
        try (RequestRoute selected = result.value()) {
            assertEquals("10.0.0.1", selected.prefill().getServerIp());
        }
    }

    @Test
    void sparseCacheLeadersAcrossTheFleetRotateOnlyAmongExactTies() {
        configureFocusedAffinity();
        long[] ttft = new long[130];
        long[] hits = new long[130];
        java.util.Arrays.fill(ttft, 105L);
        ttft[0] = 100L;
        hits[12] = 550L;
        hits[63] = 700L;
        ttft[63] = 111L;
        hits[64] = 600L;
        hits[129] = 600L;
        PrefillCandidateSet candidates = candidates(ttft, hits);

        for (int expected : new int[] {129, 64, 129, 64}) {
            assertEquals(
                    "10.1.0." + (expected + 1) + ":8080",
                    candidates
                            .endpointAddresses[
                            strategy.selectBestCandidate(
                                    candidates, 100L, RoleType.PREFILL, null, 1_000L)]);
        }
    }

    @Test
    void saturatedAffinityTiesExcludeUnknownWorkWithTheSameRank() {
        configureFocusedAffinity();
        PrefillCandidateSet candidates =
                candidates(
                        new long[] {Long.MAX_VALUE - 5L, Long.MAX_VALUE}, new long[] {100L, 600L});
        candidates.addCandidate(
                "unknown:8080",
                null,
                new RequestTimeEstimation.Candidate(
                        RequestTimeEstimation.Candidate.State.UNMODELED_ENGINE_WORK,
                        RequestTimeEstimation.Candidate.UNKNOWN,
                        1L,
                        RequestTimeEstimation.Candidate.InitialHeadDisposition.NONE,
                        "",
                        600L,
                        600L),
                3L);

        for (int i = 0; i < 4; i++) {
            assertEquals(
                    1,
                    strategy.selectBestCandidate(
                            candidates, Long.MAX_VALUE - 5L, RoleType.PREFILL, null, 1_000L));
        }
        assertEquals("CACHE_LEADER", candidates.affinityReason);
    }

    private RequestRoute select() {
        PlacementResult<RequestRoute, org.flexlb.balance.scheduler.PlacementKey> result =
                StrategyTestSupport.select(
                        strategy,
                        context.data().getRequest(),
                        context.data().getSchedulingMetadata(),
                        RoleType.PREFILL,
                        null);
        assertEquals(PlacementResult.Status.SUCCESS, result.status());
        return result.value();
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void knownSaturatedTtftWinsOverUnknownRegardlessOfCaptureOrder(boolean unknownFirst) {
        var candidates = new PrefillCandidateSet();
        for (int i = 0; i < 2; i++) {
            boolean unknown = (i == 0) == unknownFirst;
            candidates.addCandidate(
                    unknown ? "unknown:8080" : "known:8080",
                    null,
                    new RequestTimeEstimation.Candidate(
                            unknown
                                    ? RequestTimeEstimation.Candidate.State.UNMODELED_ENGINE_WORK
                                    : RequestTimeEstimation.Candidate.State.MODELED,
                            unknown ? RequestTimeEstimation.Candidate.UNKNOWN : Long.MAX_VALUE,
                            1L,
                            RequestTimeEstimation.Candidate.InitialHeadDisposition.NONE,
                            "",
                            0L,
                            0L),
                    i + 1L);
        }
        for (int i = 0; i < 4; i++) {
            int selected =
                    strategy.selectBestCandidate(
                            candidates, Long.MAX_VALUE, RoleType.PREFILL, null, 1_000L);
            assertEquals("known:8080", candidates.endpointAddresses[selected]);
        }
    }

    private void configureFocusedAffinity() {
        RoutingConfig.CacheAffinityConfig affinity = new RoutingConfig.CacheAffinityConfig();
        affinity.setMaxExtraTtftMs(10L);
        affinity.setMinPrefixHitPercent(0.0);
        config.getRouter().getRoles().getPrefill().setCacheAffinity(affinity);
        refreshContext();
    }

    private PrefillCandidateSet candidates(long[] ttftMs, long[] hitTokens) {
        assertEquals(ttftMs.length, hitTokens.length);
        PrefillCandidateSet candidates = new PrefillCandidateSet();
        for (int i :
                java.util.stream.IntStream.range(0, ttftMs.length)
                        .boxed()
                        .sorted(
                                java.util.Comparator.comparing(
                                        index -> "10.1.0." + (index + 1) + ":8080"))
                        .toList()) {
            String ip = "10.1.0." + (i + 1);
            candidates.addCandidate(
                    ip + ":8080",
                    null,
                    new RequestTimeEstimation.Candidate(
                            RequestTimeEstimation.Candidate.State.MODELED,
                            ttftMs[i],
                            ttftMs[i],
                            RequestTimeEstimation.Candidate.InitialHeadDisposition.NONE,
                            "",
                            hitTokens[i],
                            hitTokens[i]),
                    i + 1L);
        }
        return candidates;
    }

    private void refreshContext() {
        var refreshed = new RequestContext(config);
        refreshed.data().setRequest(context.data().getRequest());
        refreshed.data().setSchedulingMetadata(context.data().getSchedulingMetadata());
        org.springframework.test.util.ReflectionTestUtils.setField(
                refreshed.data(), "startTime", context.data().getStartTime());
        context = refreshed;
        var currentRegistry =
                (EndpointRegistry)
                        org.springframework.test.util.ReflectionTestUtils.getField(
                                strategy.routing, "endpoints");
        strategy =
                new StrategyTestSupport.PrefillRoutingFixture(
                        currentRegistry,
                        cache,
                        reporter,
                        cacheReporter,
                        SchedulingTestConfig.configService(config));
    }

    private void configureAffinity(long maxExtraTtftMs, double minPrefixHitPercent) {
        config.getRouter()
                .getRoles()
                .getPrefill()
                .getExecutionTimeEstimator()
                .setExpression("sum(computeTokens) + 2*sum(hitCacheTokens)");
        RoutingConfig.CacheAffinityConfig affinity = new RoutingConfig.CacheAffinityConfig();
        affinity.setMaxExtraTtftMs(maxExtraTtftMs);
        affinity.setMinPrefixHitPercent(minPrefixHitPercent);
        config.getRouter().getRoles().getPrefill().setCacheAffinity(affinity);
        refreshContext();

        publish("10.0.0.2", 8080);
        context.data()
                .setRequest(
                        context.data().getRequest().toBuilder()
                                .requestId(String.valueOf(20_001L))
                                .build());
        context.data()
                .setRequest(
                        context.data().getRequest().toBuilder()
                                .blockCacheKeys(List.of(1L, 2L, 3L, 4L, 5L))
                                .build());
        context.data()
                .setRequest(
                        context.data().getRequest().toBuilder().cacheKeyBlockSize(100L).build());
        when(cache.findMatchingEngines(any())).thenReturn(localMatches(Map.of("10.0.0.2:8080", 5)));
    }

    private void publish(String ip, int port) {
        WorkerStatus status =
                StrategyTestSupport.workerStatus(
                        RoleType.PREFILL, null, ip, port, port + 1, true, 1_000_000L, 1_000_000L);
        StrategyTestSupport.publishEndpoint(registry, RoleType.PREFILL, ip + ":" + port, status);
    }

    private static org.flexlb.cache.domain.CacheMatchResult localMatches(
            Map<String, Integer> matches) {
        return matches == null
                ? null
                : new org.flexlb.cache.domain.CacheMatchResult(
                        org.flexlb.dao.cache.HostCacheMatch.fromLocalMatches(matches),
                        org.flexlb.cache.domain.CacheMatchSource.LOCAL_SYNC,
                        0L,
                        0L);
    }
}
