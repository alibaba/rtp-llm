package org.flexlb.balance.strategy;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.RequestContext;
import org.flexlb.balance.scheduler.RequestRoute;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

class SessionPrefillRoutingTest {
    FlexlbConfig config;
    EndpointRegistry registry;
    CacheAwareService cache;
    CostBasedPrefillStrategy strategy;
    PrefillEndpoint a, b;
    long id;

    @BeforeEach void setup() {
        config = StrategyTestSupport.config();
        config.setDispatcher(ConfigTestSupport.nonBatch());
        config.getDispatcher().setMaxInflightPerPrefillWorker(32);
        config.getRouter().getRoles().getPrefill().setTokenBalance(new RoutingConfig.TokenBalanceConfig());
        config.getRouter().getRoles().getPrefill().getExecutionTimeEstimator().setExpression("sum(computeTokens) / 70");
        registry = StrategyTestSupport.endpointRegistry(SchedulingTestConfig.configService(config));
        cache = mock(CacheAwareService.class);
        when(cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of("10.0.0.1:8080", 9));
        a = publish("10.0.0.1"); b = publish("10.0.0.2");
        strategy = new CostBasedPrefillStrategy(registry, cache, mock(EngineHealthReporter.class),
                mock(CacheMetricsReporter.class), SchedulingTestConfig.configService(config));
    }

    @AfterEach void close() { registry.close(); }

    PrefillEndpoint publish(String ip) {
        return (PrefillEndpoint) StrategyTestSupport.publishEndpoint(registry, RoleType.PREFILL, ip + ":8080",
                StrategyTestSupport.workerStatus(RoleType.PREFILL, null, ip, 8080, 8081, true, 10_000_000, 10_000_000));
    }

    Request request(String session, long length) {
        return Request.builder().requestId(++id).sessionId(session).apiKey("tenant").model("model")
                .seqLen(length).cacheKeyBlockSize(length / 10).blockCacheKeys(List.of(1L, 2L)).build();
    }

    PlacementResult<RequestRoute, RoleType> select(Request request, long ageMs) {
        return select(request, ageMs, null);
    }

    PlacementResult<RequestRoute, RoleType> select(Request request, long ageMs, PrefillEndpoint source) {
        var ctx = new RequestContext(config);
        if (source != null) {
            try {
                var field = RequestContext.class.getDeclaredField("prefillMigrationSource");
                field.setAccessible(true);
                field.set(ctx, source);
            } catch (ReflectiveOperationException error) { throw new AssertionError(error); }
        }
        ctx.data().setRequest(request);
        long now = System.currentTimeMillis();
        ctx.data().setSchedulingMetadata(SchedulingMetadata.explicit(50, now + 60_000));
        ctx.register(request.getRequestId(), now - ageMs, now + 60_000, 60_000);
        var route = new RequestRoute(ctx);
        var result = strategy.select(request, ctx.data().getSchedulingMetadata(), RoleType.PREFILL, null, route);
        if (result.status() == PlacementResult.Status.SUCCESS) { return PlacementResult.success(route); }
        route.close();
        return PlacementResult.blocked(RoleType.PREFILL);
    }

    void remember() {
        try (var route = select(request("session", 1_000), 0).value()) {
            assertSame(a, route.prefillEp());
            route.rememberSessionWorker();
        }
    }

    @ParameterizedTest @ValueSource(ints = {0, 1, 20, 50, 80, 90, 100})
    void committedSessionWinsEvenWhenOtherWorkerHasMoreCache(int hitPercent) {
        remember();
        when(cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of(b.ipPort(), hitPercent));
        var request = request("session", 10_000).toBuilder().cacheKeyBlockSize(100).build();
        try (var route = select(request, 0).value()) {
            assertSame(a, route.prefillEp());
            assertEquals(0, route.hitCache(), "session affinity must never invent cached tokens");
            assertEquals(10_000, a.ownership.tokenWorkload(System.nanoTime() / 1_000_000).plannedTokens());
        }
    }

    @Test void speculativeSelectionDoesNotRecordSessionAndBlankSessionsUseCache() {
        try (var ignored = select(request("session", 1_000), 0).value()) { assertSame(a, ignored.prefillEp()); }
        when(cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of(b.ipPort(), 9));
        for (String session : new String[]{"session", "", " ", null}) {
            try (var route = select(request(session, 1_000), 0).value()) { assertSame(b, route.prefillEp()); }
        }
    }

    @Test void tenantAndModelDoNotShareSessionHints() {
        remember();
        when(cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of(b.ipPort(), 9));
        for (Request request : List.of(request("session", 1_000).toBuilder().apiKey("other").build(),
                request("session", 1_000).toBuilder().model("other").build())) {
            try (var route = select(request, 0).value()) { assertSame(b, route.prefillEp()); }
        }
    }

    @Test void fullSessionWorkerMovesOnlyAgedShortWorkToIdleTargetAndChargesColdInput() {
        remember();
        var running = mock(RequestRoute.class);
        when(running.requestId()).thenReturn(100_000L);
        when(running.seqLen()).thenReturn(200_000L);
        var lease = a.ownership.reserveUnqueuedRoute(running, 1, 0).reservation();
        try {
            assertEquals(PlacementResult.Status.BLOCKED, select(request("session", 1_000), 0).status());
            assertEquals(PlacementResult.Status.BLOCKED, select(request("session", 500_000), 10_000).status());
            var busy = mock(RequestRoute.class);
            when(busy.requestId()).thenReturn(100_001L);
            when(busy.seqLen()).thenReturn(1L);
            var targetLease = b.ownership.reserveUnqueuedRoute(busy, 1, 0).reservation();
            try { assertEquals(PlacementResult.Status.BLOCKED, select(request("session", 1_000), 10_000).status()); }
            finally { b.ownership.release(targetLease, null); }
            try (var route = select(request("session", 1_000), 10_000).value()) {
                assertSame(b, route.prefillEp());
                assertEquals(0, route.hitCache());
                assertEquals(1_000, b.ownership.tokenWorkload(System.nanoTime() / 1_000_000).plannedTokens());
                var target = b.ownership.reserveUnqueuedRoute(route, 1, 0).reservation();
                assertEquals(0, b.ownership.tokenWorkload(System.nanoTime() / 1_000_000).plannedTokens());
                assertEquals(1_000, b.ownership.tokenWorkload(System.nanoTime() / 1_000_000).pendingTokens());
                b.ownership.release(target, null);
            }
            assertEquals(0, b.ownership.tokenWorkload(System.nanoTime() / 1_000_000).score());
        } finally { a.ownership.release(lease, null); }
    }

    @Test void queuedEstimateUsesSessionOwnerAndColdMigrationDestination() {
        remember();
        when(cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of(b.ipPort(), 9));
        var estimate = strategy.estimateQueuedWork(request("session", 1_000), RoleType.PREFILL, null);
        assertEquals(0, estimate.hitTokens());
        when(cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of(a.ipPort(), 9));
        estimate = strategy.estimateQueuedWork(request("session", 1_000), RoleType.PREFILL, null, a);
        assertEquals(0, estimate.hitTokens(), "withdrawal cannot retain the old owner's 90% hit estimate");
        assertEquals(14, estimate.workMs());
    }

    @Test void concurrentSessionUpdateCannotSendMigratingRequestBackToItsSource() {
        remember();
        var running = mock(RequestRoute.class);
        when(running.requestId()).thenReturn(100_000L);
        when(running.seqLen()).thenReturn(200_000L);
        var lease = a.ownership.reserveUnqueuedRoute(running, 1, 0).reservation();
        try (var otherTurn = select(request("session", 1_000), 10_000).value()) {
            assertSame(b, otherTurn.prefillEp());
            otherTurn.rememberSessionWorker();
        } finally { a.ownership.release(lease, null); }
        // Both endpoints are now idle, but A remains forbidden for this withdrawn request.
        try (var migrating = select(request("session", 1_000), 10_000, a).value()) {
            assertSame(b, migrating.prefillEp());
            assertEquals(0, migrating.hitCache());
        }
    }

    @Test void scaleInAndSameAddressReplacementDiscardOldGenerationAffinity() {
        remember();
        var status = a.getStatus();
        EndpointRegistry.Retirement retirement;
        status.lock.lock();
        try { retirement = registry.beginRetirement(RoleType.PREFILL, a.ipPort(), status); }
        finally { status.lock.unlock(); }
        retirement.complete(cache, org.slf4j.LoggerFactory.getLogger(getClass()));
        assertEquals(1, strategy.currentPrefillWorkerCount(null));
        when(cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of(b.ipPort(), 9));
        try (var route = select(request("session", 1_000), 0).value()) { assertSame(b, route.prefillEp()); }
        var replacement = publish("10.0.0.1");
        assertNotSame(a, replacement);
        assertEquals(2, strategy.currentPrefillWorkerCount(null));
        try (var route = select(request("session", 1_000), 0).value()) { assertSame(b, route.prefillEp()); }
    }
}
