package org.flexlb.balance.strategy;

import org.flexlb.balance.RoutingResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.EndpointTestSupport;
import org.flexlb.balance.planner.BuiltinGroupPlanner;
import org.flexlb.balance.scheduler.QueueExecutionSettings;
import org.flexlb.balance.scheduler.RequestContext;
import org.flexlb.balance.scheduler.RequestRouteSelector;
import org.flexlb.balance.scheduler.RouteResult;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.master.WorkerStatusTestSupport;
import org.flexlb.dao.route.RoleType;
import org.flexlb.metric.NoOpFlexMonitor;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Checks selection registers P/D resources before publication and rolls back incomplete combinations.
 * Concurrent selections and publication cannot duplicate or replace those registrations.
 */
class SelectionRegistrationTest {
    private final ConfigService config =
            new ConfigService(
                    """
            {"requestLifecycle":{"request":{"timeoutMs":60000}},"dispatcher":{"type":"BATCH"},"router":{"roles":{
              "prefill":{"executionTimeEstimator":{"expression":"sum(computeTokens)"}},
              "decode":{"availability":{"maxEngineRequests":10,"maxKvUsagePercent":100}}}}}
            """);

    @Test
    void selectedRequestsAreVisibleBeforePublicationAndDoNotEnterDispatchQueue() throws Exception {
        var registry = StrategyTestSupport.endpointRegistry(config);
        try (AutoCloseable cleanup = registry::close) {
            add(registry, RoleType.PREFILL, "p1");
            add(registry, RoleType.PREFILL, "p2");
            add(registry, RoleType.DECODE, "d1");
            var selector = selector(registry);
            var first = selector.select(context("a"), null);
            assertEquals(RoutingResult.Status.SUCCESS, first.status());
            var a = first.value();
            assertNotNull(a.prefillReservation());
            assertNotNull(a.decodeReservation());
            assertTrue(a.prefillEndpoint().captureQueue(10).items().isEmpty());
            assertEquals(1, a.decodeEndpoint().routingView().totalLoad());
            var second = selector.select(context("b"), null);
            assertEquals(RoutingResult.Status.SUCCESS, second.status());
            assertNotSame(a.prefillEndpoint(), second.value().prefillEndpoint());
            release(a);
            release(second.value());
            assertEquals(0, a.decodeEndpoint().routingView().totalLoad());
        }
    }

    @Test
    void successfulPolicyResultCannotReplaceActualDecodeRegistration() throws Exception {
        var registry = StrategyTestSupport.endpointRegistry(config);
        try (AutoCloseable cleanup = registry::close) {
            add(registry, RoleType.PREFILL, "p1");
            add(registry, RoleType.DECODE, "d1");
            var routing = selector(registry);
            var policy = org.mockito.Mockito.mock(DecodeSelectionStrategy.class);
            when(policy.selectAndRegister(any(), any(), any(), any(), any(), any(), any()))
                    .thenAnswer(
                            call ->
                                    RoutingResult.success(
                                            ((java.util.List<?>) call.getArgument(3)).getFirst()));
            org.springframework.test.util.ReflectionTestUtils.setField(
                    routing, "decodeSelector", policy);
            org.junit.jupiter.api.Assertions.assertThrows(
                    IllegalStateException.class,
                    () -> routing.select(context("unregistered"), null));
            assertEquals(
                    0,
                    registry.prefillRoutingSnapshot(RoleType.PREFILL)
                            .getFirst()
                            .endpoint()
                            .queueCounters()
                            .outstandingRequests());
            var views = new java.util.ArrayList<DecodeResources.DecodeRoutingView>();
            registry.decodeRoutingSnapshot(null, views);
            assertEquals(0, views.getFirst().totalLoad());
        }
    }

    @Test
    void missingDecodeRollsBackAlreadySelectedPrefill() throws Exception {
        var registry = StrategyTestSupport.endpointRegistry(config);
        try (AutoCloseable cleanup = registry::close) {
            add(registry, RoleType.PREFILL, "p1");
            var endpoint = registry.prefillRoutingSnapshot(RoleType.PREFILL).getFirst().endpoint();
            long initial = endpoint.queueCounters().outstandingRequests();
            var selector = selector(registry);
            assertEquals(
                    RoutingResult.Status.BLOCKED, selector.select(context("a"), null).status());
            assertEquals(initial, endpoint.queueCounters().outstandingRequests());
            add(registry, RoleType.DECODE, "d1");
            var next = selector.select(context("a"), null);
            assertEquals(RoutingResult.Status.SUCCESS, next.status());
            release(next.value());
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void selectedButUnsentDecodeConsumesCapacityInBothModes(boolean direct) throws Exception {
        var settings = config.loadBalanceConfig();
        settings.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(1L);
        if (direct) {
            settings.setScheduler(org.flexlb.config.ConfigTestSupport.direct());
            settings.setDispatcher(org.flexlb.config.ConfigTestSupport.nonBatch());
        }
        var registry = StrategyTestSupport.endpointRegistry(config);
        try (AutoCloseable cleanup = registry::close) {
            add(registry, RoleType.PREFILL, "p1");
            add(registry, RoleType.DECODE, "d1");
            var selector = selector(registry);
            var first = selector.select(context("first"), null);
            assertEquals(RoutingResult.Status.SUCCESS, first.status());
            var route = first.value();
            try {
                var blocked = selector.select(context("second"), null);
                assertEquals(RoutingResult.Status.BLOCKED, blocked.status());
                assertEquals(RoleType.DECODE, blocked.blocker().role());
                assertEquals(1, route.decodeEndpoint().routingView().totalLoad());
                assertEquals(1, route.prefillEndpoint().queueCounters().outstandingRequests());
                assertEquals(0, route.decodeEndpoint().resourceSnapshot().activeDispatchPermits());
            } finally {
                release(route);
            }
            var retry = selector.select(context("second"), null);
            assertEquals(RoutingResult.Status.SUCCESS, retry.status());
            release(retry.value());
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void queuedHeadsDecodePressureDoesNotDetermineTheNextRequestsPlacement(boolean spareDecode)
            throws Exception {
        var settings = config.loadBalanceConfig();
        settings.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(1L);
        var registry = StrategyTestSupport.endpointRegistry(config);
        try (AutoCloseable cleanup = registry::close) {
            add(registry, RoleType.PREFILL, "p1");
            add(registry, RoleType.DECODE, "d1");
            var selector = selector(registry);
            var selected = selector.select(context("head"), null);
            assertEquals(RoutingResult.Status.SUCCESS, selected.status());
            var head = selected.value();
            var prefill = head.prefillEndpoint();
            var decode = head.decodeEndpoint();
            var headDecode = head.decodeReservation();
            try {
                assertTrue(
                        org.flexlb.balance.endpoint.EndpointTestSupport.enqueueRoute(
                                prefill,
                                head,
                                QueueExecutionSettings.capture(
                                        settings, BuiltinGroupPlanner.from(settings))));
                assertSame(head, prefill.captureQueue(10).items().getFirst());

                // The engine consumes its KV after the queued head has registered D1.
                var status = decode.getStatus();
                var full = new WorkerStatusResponse();
                full.setRole(RoleType.DECODE);
                full.setAlive(true);
                full.setStatusVersion(status.appliedStatusCursor().statusVersion() + 1L);
                full.setLatestFinishedVersion(0L);
                full.setTotalKvCacheTokens(1_000_000L);
                full.setAvailableKvCacheTokens(0L);
                org.flexlb.balance.endpoint.WorkerEndpoint.StatusChange change;
                status.lock.lock();
                try {
                    change =
                            decode.applyWorkerStatus(
                                    status,
                                    status.prepareStatus(
                                            WorkerStatusTestSupport.freezeStatusResponse(
                                                    status, full)));
                } finally {
                    status.lock.unlock();
                }
                decode.publishStatus(change);
                assertEquals(
                        DecodeResources.EngineDispatchPermitAcquireStatus.CAPACITY_FULL,
                        decode.acquireDispatchPermit(
                                        headDecode, SchedulingTestConfig.capacity(settings))
                                .status());
                if (spareDecode) add(registry, RoleType.DECODE, "d2");

                var result = selector.select(context("next"), null);
                try {
                    if (spareDecode) {
                        assertEquals(RoutingResult.Status.SUCCESS, result.status());
                        var next = result.value();
                        assertSame(prefill, next.prefillEndpoint());
                        assertEquals("d2", next.decodeEndpoint().getIp());
                        assertNotSame(decode, next.decodeEndpoint());
                        assertEquals(
                                next.decodeReservation(),
                                EndpointTestSupport.decodeReservation(
                                        next.decodeEndpoint(), "next"));
                        assertNotNull(next.prefillReservation());
                        assertEquals(2, prefill.queueCounters().outstandingRequests());
                        assertEquals(1, next.decodeEndpoint().routingView().totalLoad());
                    } else {
                        assertEquals(RoutingResult.Status.BLOCKED, result.status());
                        assertEquals(RoleType.DECODE, result.blocker().role());
                        assertEquals("g", result.blocker().group());
                        assertEquals(decode.ipPort(), result.blocker().endpoint());
                        assertEquals(
                                StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(),
                                result.failure().getCode());
                        assertNull(result.value());
                        assertEquals(1, prefill.queueCounters().outstandingRequests());
                    }
                    assertSame(head, prefill.captureQueue(10).items().getFirst());
                    assertEquals(1, prefill.captureQueue(10).items().size());
                    assertEquals(headDecode, EndpointTestSupport.decodeReservation(decode, "head"));
                    assertNull(EndpointTestSupport.decodeReservation(decode, "next"));
                    assertEquals(1, decode.resourceSnapshot().queuedCount());
                    assertEquals(0, decode.resourceSnapshot().activeDispatchPermits());
                } finally {
                    if (result.value() != null) release(result.value());
                }
                assertEquals(1, prefill.queueCounters().outstandingRequests());
            } finally {
                release(head);
            }
            assertEquals(0, prefill.queueCounters().outstandingRequests());
            assertTrue(prefill.captureQueue(10).items().isEmpty());
            assertNull(EndpointTestSupport.decodeReservation(decode, "head"));
        }
    }

    @Test
    void completedRequestRollsBackPartialSelectionBeforeDecode() throws Exception {
        var registry = StrategyTestSupport.endpointRegistry(config);
        try (AutoCloseable cleanup = registry::close) {
            add(registry, RoleType.PREFILL, "p1");
            add(registry, RoleType.DECODE, "d1");
            var selector = selector(registry);
            var context = context("finished");
            context.data().setFuture(new java.util.concurrent.CompletableFuture<>());
            var prefill = registry.prefillRoutingSnapshot(RoleType.PREFILL).getFirst().endpoint();
            var cache = (CacheAwareService) ReflectionTestUtils.getField(selector, "cacheService");
            doAnswer(
                            call -> {
                                assertEquals(1, prefill.queueCounters().outstandingRequests());
                                context.data()
                                        .getFuture()
                                        .complete(new org.flexlb.dao.loadbalance.Response());
                                return null;
                            })
                    .when(cache)
                    .trackRoutingPrediction(
                            any(), any(), any(), any(), anyLong(), anyLong(), any());
            var result = selector.select(context, null);
            assertEquals(RoutingResult.Status.CLOSED, result.status());
            assertEquals(0, prefill.queueCounters().outstandingRequests());
            var decode =
                    (DecodeEndpoint)
                            registry.selectEndpoint(
                                    RoleType.DECODE,
                                    registry.endpointAddressSnapshot(RoleType.DECODE).getFirst());
            assertEquals(0, decode.routingView().totalLoad());
        }
    }

    @Test
    void sameSnapshotCanRegisterOnlyOneConcurrentRequest() throws Exception {
        var registry = StrategyTestSupport.endpointRegistry(config);
        try (AutoCloseable cleanup = registry::close) {
            add(registry, RoleType.DECODE, "d1");
            var endpoint =
                    (DecodeEndpoint)
                            registry.selectEndpoint(
                                    RoleType.DECODE,
                                    registry.endpointAddressSnapshot(RoleType.DECODE).getFirst());
            var snapshot = endpoint.routingView();
            var capacity = new DecodeResources.AdmissionCapacity(10, 100);
            var barrier = new java.util.concurrent.CyclicBarrier(2);
            try (var executor = java.util.concurrent.Executors.newFixedThreadPool(2)) {
                var results =
                        new java.util.ArrayList<
                                java.util.concurrent.Future<DecodeResources.ReserveStatus>>();
                for (int i = 0; i < 2; i++) {
                    String id = "request-" + i;
                    results.add(
                            executor.submit(
                                    () -> {
                                        barrier.await();
                                        return endpoint.tryRegisterSelection(
                                                        snapshot, id, 100, 100, 50, capacity)
                                                .status();
                                    }));
                }
                var statuses = java.util.EnumSet.noneOf(DecodeResources.ReserveStatus.class);
                for (var result : results)
                    statuses.add(result.get(5, java.util.concurrent.TimeUnit.SECONDS));
                assertEquals(
                        java.util.EnumSet.of(
                                DecodeResources.ReserveStatus.ACQUIRED,
                                DecodeResources.ReserveStatus.VERSION_CONFLICT),
                        statuses);
                assertEquals(1, endpoint.routingView().totalLoad());
            }
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void publicationCannotCreateOrReplaceSelectionRegistration(boolean direct) throws Exception {
        var settings = config.loadBalanceConfig();
        if (direct) {
            settings.setScheduler(org.flexlb.config.ConfigTestSupport.direct());
            settings.setDispatcher(org.flexlb.config.ConfigTestSupport.nonBatch());
        }
        var registry = StrategyTestSupport.endpointRegistry(config);
        try (AutoCloseable cleanup = registry::close) {
            add(registry, RoleType.PREFILL, "p1");
            add(registry, RoleType.DECODE, "d1");
            var selected = selector(registry).select(context("registered"), null).value();
            var endpoint = selected.prefillEndpoint();
            org.junit.jupiter.api.Assertions.assertThrows(
                    IllegalStateException.class,
                    () -> new RouteResult(selected.ctx(), selected.roles()));
            var withoutRegistration = org.mockito.Mockito.spy(selected);
            org.mockito.Mockito.doReturn(null).when(withoutRegistration).prefillReservation();
            if (direct) {
                org.junit.jupiter.api.Assertions.assertNull(
                        endpoint.commitDirect(withoutRegistration));
                org.junit.jupiter.api.Assertions.assertNotNull(endpoint.commitDirect(selected));
            } else {
                var queue =
                        org.flexlb.balance.scheduler.QueueExecutionSettings.capture(
                                settings,
                                org.flexlb.balance.planner.BuiltinGroupPlanner.from(settings));
                org.junit.jupiter.api.Assertions.assertFalse(
                        org.flexlb.balance.endpoint.EndpointTestSupport.enqueueRoute(
                                endpoint, withoutRegistration, queue));
                assertTrue(
                        org.flexlb.balance.endpoint.EndpointTestSupport.enqueueRoute(
                                endpoint, selected, queue));
            }
            assertEquals(1, endpoint.queueCounters().outstandingRequests());
            release(selected);
            org.junit.jupiter.api.Assertions.assertNull(endpoint.commitDirect(selected));
            org.junit.jupiter.api.Assertions.assertFalse(
                    org.flexlb.balance.endpoint.EndpointTestSupport.tryEnqueueRoute(
                            endpoint, selected));
            assertEquals(0, endpoint.queueCounters().outstandingRequests());
        }
    }

    @Test
    void retiringPrefillBeforeCombinationCompletesRollsBackDecode() throws Exception {
        var registry = StrategyTestSupport.endpointRegistry(config);
        try (AutoCloseable cleanup = registry::close) {
            add(registry, RoleType.PREFILL, "p1");
            add(registry, RoleType.DECODE, "d1");
            var selector = selector(registry);
            var prefill = registry.prefillRoutingSnapshot(RoleType.PREFILL).getFirst().endpoint();
            var cache = (CacheAwareService) ReflectionTestUtils.getField(selector, "cacheService");
            doAnswer(
                            call -> {
                                assertEquals(1, prefill.queueCounters().outstandingRequests());
                                assertEquals(
                                        "retired",
                                        prefill.captureRouteProjectionInputs()
                                                .queue()
                                                .activeItems()
                                                .getFirst()
                                                .requestId());
                                prefill.close();
                                return null;
                            })
                    .when(cache)
                    .trackRoutingPrediction(
                            any(), any(), any(), any(), anyLong(), anyLong(), any());
            assertEquals(
                    RoutingResult.Status.RETRY, selector.select(context("retired"), null).status());
            assertEquals(0, prefill.queueCounters().outstandingRequests());
            var decode =
                    (DecodeEndpoint)
                            registry.selectEndpoint(
                                    RoleType.DECODE,
                                    registry.endpointAddressSnapshot(RoleType.DECODE).getFirst());
            assertEquals(0, decode.routingView().totalLoad());
        }
    }

    private RequestContext context(String id) {
        var context = new RequestContext(config.loadBalanceConfig());
        context.data()
                .setRequest(Request.builder().requestId(id).seqLen(1000L).maxNewTokens(1).build());
        context.data().setSchedulingMetadata(SchedulingMetadata.explicit(50, Long.MAX_VALUE));
        return context;
    }

    private void add(EndpointRegistry registry, RoleType role, String address) {
        var worker =
                StrategyTestSupport.workerStatus(
                        role, "g", address, 8080, 8081, true, 1000000, 1000000);
        StrategyTestSupport.publishEndpoint(registry, role, worker.getLogicalIpPort(), worker);
    }

    private RequestRouteSelector selector(EndpointRegistry registry) {
        var cache = mock(CacheAwareService.class);
        when(cache.findMatchingEngines(any()))
                .thenReturn(
                        org.flexlb.cache.domain.CacheMatchResult.empty(
                                org.flexlb.cache.domain.CacheMatchSource.LOCAL_SYNC));
        var monitor = new NoOpFlexMonitor();
        var metrics = new CacheMetricsReporter();
        ReflectionTestUtils.setField(metrics, "monitor", monitor);
        var model = mock(ModelMetaConfig.class);
        when(model.requiredRoles()).thenReturn(List.of(RoleType.PREFILL, RoleType.DECODE));
        return new RequestRouteSelector(
                registry,
                new CostBasedPrefillStrategy(cache, config),
                new DecodeSelector(
                        config,
                        org.flexlb.balance.scheduler.SchedulingTestConfig.capacity(
                                config.loadBalanceConfig())),
                new VitWorkerSelector(),
                mock(EncoderSelectionStrategy.class),
                cache,
                null,
                new EngineHealthReporter(monitor, registry),
                metrics,
                config,
                model);
    }

    private void release(RouteResult route) {
        route.prefillEndpoint().releaseRequest(route);
        route.decodeEndpoint()
                .release(route.decodeReservation(), DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
    }
}
