package org.flexlb.balance.strategy;

import org.flexlb.balance.schedule.RequestContext;
import org.flexlb.balance.schedule.RequestResult;
import org.flexlb.balance.schedule.SchedulingMetadata;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.cache.metrics.CacheMetricsReporter;
import org.flexlb.config.ConfigService;
import org.flexlb.config.service.ModelMetaConfig;
import org.flexlb.config.service.RoleType;
import org.flexlb.util.metrics.NoOpFlexMonitor;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.DecodeResources;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.endpoint.Request;
import org.flexlb.worker.endpoint.WorkerEndpoint;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
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
            {"requestLifecycle":{"request":{"timeoutMs":60000}},"router":{"roles":{
              "prefill":{"maxComputeTokens":10000},
              "decode":{"availability":{"maxEngineRequests":10,"maxKvUsagePercent":100}}}}}
            """);

    @Test
    void successfulPolicyResultStillRequiresActualDecodeRegistration() throws Exception {
        var registry = StrategyTestSupport.endpointRegistry(config);
        try (AutoCloseable cleanup = registry::close) {
            add(registry, RoleType.PREFILL, "p1");
            add(registry, RoleType.DECODE, "d1");
            var routing = selector(registry);
            var policy =
                    org.mockito.Mockito.spy(
                            new DecodeStrategy(
                                    config,
                                    org.flexlb.balance.schedule.SchedulingTestConfig.capacity(
                                            config.loadBalanceConfig()),
                                    registry));
            org.mockito.Mockito.doAnswer(
                            call ->
                                    java.util.Optional.of(
                                            ((java.util.List<DecodeEndpoint.SelectionSnapshot>)
                                                            call.getArgument(1))
                                                    .getFirst()))
                    .when(policy)
                    .chooseBest(any(), any());
            org.springframework.test.util.ReflectionTestUtils.setField(
                    routing, "decodeStrategy", policy);
            var accepted = routing.select(context("registered"), null);
            assertEquals(RoutingResult.Status.SUCCESS, accepted.status());
            assertEquals(
                    1,
                    accepted.decodeEndpoint().resourceSnapshot().requests().values().stream()
                            .filter(snapshotRequest -> !snapshotRequest.phase().isEngineConfirmed())
                            .count());
            accepted.prefillEndpoint().rollbackSelection(accepted.prefillRegistration());
            accepted.decodeEndpoint().releaseRequest(accepted.decodeRegistration());
            assertEquals(
                    0,
                    registry.<org.flexlb.worker.endpoint.PrefillEndpoint>endpoints(
                                    RoleType.PREFILL, null)
                            .getFirst()
                            .selectionSnapshot()
                            .computeTokens());
            var views = new java.util.ArrayList<DecodeEndpoint.SelectionSnapshot>();
            views.addAll(
                    org.flexlb.worker.endpoint.EndpointTestSupport.decodeRoutingSnapshot(
                            registry, null));
            assertEquals(0, views.getFirst().usage().occupiedRequests());
        }
    }

    @Test
    void missingDecodeRollsBackAlreadySelectedPrefill() throws Exception {
        var registry = StrategyTestSupport.endpointRegistry(config);
        try (AutoCloseable cleanup = registry::close) {
            add(registry, RoleType.PREFILL, "p1");
            var endpoint =
                    registry.<org.flexlb.worker.endpoint.PrefillEndpoint>endpoints(
                                    RoleType.PREFILL, null)
                            .getFirst();
            long initial = endpoint.selectionSnapshot().computeTokens();
            var selector = selector(registry);
            assertEquals(
                    RoutingResult.Status.BLOCKED, selector.select(context("a"), null).status());
            assertEquals(initial, endpoint.selectionSnapshot().computeTokens());
            add(registry, RoleType.DECODE, "d1");
            var next = selector.select(context("a"), null);
            assertEquals(RoutingResult.Status.SUCCESS, next.status());
            release(next);
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void selectedButUnsentDecodeConsumesCapacityInBothModes(boolean direct) throws Exception {
        var settings = config.loadBalanceConfig();
        settings.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(1L);
        if (direct) {
            settings.setScheduler(org.flexlb.config.ConfigTestSupport.direct());
        }
        var registry = StrategyTestSupport.endpointRegistry(config);
        try (AutoCloseable cleanup = registry::close) {
            add(registry, RoleType.PREFILL, "p1");
            add(registry, RoleType.DECODE, "d1");
            var selector = selector(registry);
            var first = selector.select(context("first"), null);
            assertEquals(RoutingResult.Status.SUCCESS, first.status());
            var route = first;
            try {
                var blocked = selector.select(context("second"), null);
                assertEquals(RoutingResult.Status.BLOCKED, blocked.status());
                assertEquals(RoleType.DECODE, blocked.blocker().role());
                assertEquals(
                        1, route.decodeEndpoint().selectionSnapshot().usage().occupiedRequests());
                assertEquals(1, route.prefillEndpoint().stats().locallyOwnedRequests());
            } finally {
                release(route);
            }
            var retry = selector.select(context("second"), null);
            assertEquals(RoutingResult.Status.SUCCESS, retry.status());
            release(retry);
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
            context.setFuture(new java.util.concurrent.CompletableFuture<>());
            var prefill =
                    registry.<org.flexlb.worker.endpoint.PrefillEndpoint>endpoints(
                                    RoleType.PREFILL, null)
                            .getFirst();
            var cache =
                    (CacheAwareService)
                            ReflectionTestUtils.getField(
                                    ReflectionTestUtils.getField(selector, "prefillStrategy"),
                                    "cacheService");
            doAnswer(
                            call -> {
                                assertEquals(1, prefill.stats().locallyOwnedRequests());
                                context.getFuture()
                                        .complete(new org.flexlb.worker.endpoint.Response());
                                return null;
                            })
                    .when(cache)
                    .trackRoutingPrediction(
                            any(), any(), any(), any(), anyLong(), anyLong(), any());
            var result = selector.select(context, null);
            assertEquals(RoutingResult.Status.CLOSED, result.status());
            assertEquals(0, prefill.selectionSnapshot().computeTokens());
            var decode =
                    (DecodeEndpoint)
                            registry.getEndpoint(
                                    RoleType.DECODE,
                                    registry.endpoints(RoleType.DECODE, null).getFirst().ipPort());
            assertEquals(0, decode.selectionSnapshot().usage().occupiedRequests());
        }
    }

    @Test
    void sameSnapshotCanRegisterOnlyOneConcurrentRequest() throws Exception {
        var registry = StrategyTestSupport.endpointRegistry(config);
        try (AutoCloseable cleanup = registry::close) {
            add(registry, RoleType.DECODE, "d1");
            var endpoint =
                    (DecodeEndpoint)
                            registry.getEndpoint(
                                    RoleType.DECODE,
                                    registry.endpoints(RoleType.DECODE, null).getFirst().ipPort());
            var snapshot = endpoint.selectionSnapshot();
            var capacity = new DecodeResources.AdmissionCapacity(10, 100);
            var barrier = new java.util.concurrent.CyclicBarrier(2);
            try (var executor = java.util.concurrent.Executors.newFixedThreadPool(2)) {
                var results =
                        new java.util.ArrayList<
                                java.util.concurrent.Future<WorkerEndpoint.RegistrationStatus>>();
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
                var statuses = java.util.EnumSet.noneOf(WorkerEndpoint.RegistrationStatus.class);
                for (var result : results)
                    statuses.add(result.get(5, java.util.concurrent.TimeUnit.SECONDS));
                assertEquals(
                        java.util.EnumSet.of(
                                WorkerEndpoint.RegistrationStatus.REGISTERED,
                                WorkerEndpoint.RegistrationStatus.VERSION_CONFLICT),
                        statuses);
                assertEquals(1, endpoint.selectionSnapshot().usage().occupiedRequests());
            }
        }
    }

    @Test
    void completeRouteTransfersResourcesForSchedulerValidationAndRollback() throws Exception {
        var registry = StrategyTestSupport.endpointRegistry(config);
        try (AutoCloseable cleanup = registry::close) {
            add(registry, RoleType.PREFILL, "p1");
            add(registry, RoleType.DECODE, "d1");
            var selector = selector(registry);
            var prefill =
                    registry.<org.flexlb.worker.endpoint.PrefillEndpoint>endpoints(
                                    RoleType.PREFILL, null)
                            .getFirst();
            var cache =
                    (CacheAwareService)
                            ReflectionTestUtils.getField(
                                    ReflectionTestUtils.getField(selector, "prefillStrategy"),
                                    "cacheService");
            doAnswer(
                            call -> {
                                assertEquals(1, prefill.stats().locallyOwnedRequests());
                                org.flexlb.worker.endpoint.EndpointTestSupport.closeAndWait(
                                        prefill);
                                return null;
                            })
                    .when(cache)
                    .trackRoutingPrediction(
                            any(), any(), any(), any(), anyLong(), anyLong(), any());
            var context = context("retired");
            var route = selector.select(context, null);
            assertEquals(RoutingResult.Status.SUCCESS, route.status());
            org.junit.jupiter.api.Assertions.assertNull(context.snapshot().route());
            assertEquals(1, route.decodeEndpoint().selectionSnapshot().usage().occupiedRequests());
            route.prefillEndpoint().rollbackSelection(route.prefillRegistration());
            route.decodeEndpoint().rollbackSelection(route.decodeRegistration());
            assertEquals(0, prefill.selectionSnapshot().computeTokens());
            var decode =
                    (DecodeEndpoint)
                            registry.getEndpoint(
                                    RoleType.DECODE,
                                    registry.endpoints(RoleType.DECODE, null).getFirst().ipPort());
            assertEquals(0, decode.selectionSnapshot().usage().occupiedRequests());
        }
    }

    private RequestContext context(String id) {
        var context = new RequestContext(config.loadBalanceConfig());
        context.setRequest(Request.builder().requestId(id).seqLen(1000L).maxNewTokens(1).build());
        context.setSchedulingMetadata(SchedulingMetadata.explicit(50, Long.MAX_VALUE));
        return context;
    }

    private void add(EndpointRegistry registry, RoleType role, String address) {
        var worker =
                StrategyTestSupport.workerStatus(
                        role, "g", address, 8080, 8081, true, 1000000, 1000000);
        StrategyTestSupport.publishEndpoint(registry, worker);
    }

    private RequestRouteSelector selector(EndpointRegistry registry) {
        var cache = mock(CacheAwareService.class);
        when(cache.findMatchingEngines(any()))
                .thenReturn(
                        org.flexlb.cache.CacheMatchResult.empty(
                                org.flexlb.cache.CacheMatchSource.LOCAL_SYNC));
        var monitor = new NoOpFlexMonitor();
        var metrics = new CacheMetricsReporter();
        ReflectionTestUtils.setField(metrics, "monitor", monitor);
        var model = mock(ModelMetaConfig.class);
        when(model.requiredRoles()).thenReturn(List.of(RoleType.PREFILL, RoleType.DECODE));
        return new RequestRouteSelector(
                registry,
                new CostBasedPrefillStrategy(config, registry, cache, metrics),
                new DecodeStrategy(
                        config,
                        org.flexlb.balance.schedule.SchedulingTestConfig.capacity(
                                config.loadBalanceConfig()),
                        registry),
                new VitStrategy(
                        org.mockito.Mockito.mock(org.flexlb.cache.VitCacheDirectory.class),
                        registry),
                mock(EncoderStrategy.class),
                model);
    }

    private void release(RequestResult route) {
        route.prefillEndpoint().releaseRequest(route);
        route.decodeEndpoint().releaseRequest(route.decodeRegistration());
    }
}
