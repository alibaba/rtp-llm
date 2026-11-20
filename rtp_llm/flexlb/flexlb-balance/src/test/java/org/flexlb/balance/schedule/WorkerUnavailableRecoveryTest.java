package org.flexlb.balance.schedule;

import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.metrics.RequestSchedulerReporter;
import org.flexlb.balance.request.Request;
import org.flexlb.balance.request.Response;
import org.flexlb.balance.requestlog.TheoryCacheHitReporter;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.DecodeStrategy;
import org.flexlb.balance.strategy.EncoderStrategy;
import org.flexlb.balance.strategy.PrefillSelectionStrategy;
import org.flexlb.balance.strategy.RequestRouteSelector;
import org.flexlb.balance.strategy.SelectionResult;
import org.flexlb.balance.strategy.TokenBalancedPrefillStrategy;
import org.flexlb.balance.strategy.VitStrategy;
import org.flexlb.cache.CacheMatchResult;
import org.flexlb.cache.CacheMatchSource;
import org.flexlb.cache.HostCacheMatch;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.cache.metrics.CacheMetricsReporter;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.routing.RoutingConfig;
import org.flexlb.config.scheduler.QueueOrderingConfig;
import org.flexlb.config.scheduler.SchedulerConfig;
import org.flexlb.config.service.ModelMetaConfig;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.CapacityNotifier;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.endpoint.EndpointTestSupport;
import org.flexlb.worker.endpoint.PrefillEndpoint;
import org.flexlb.worker.status.WorkerStatusResponse;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Injects worker loss at real selection and publication boundaries, then requires a successful
 * response on B within one second. Real scheduler retries and endpoint ledgers own recovery.
 */
class WorkerUnavailableRecoveryTest {
    @ParameterizedTest(name = "{0}/{1}/{2}")
    @MethodSource("selectionCases")
    void unavailableBeforeResponseMustRecover(String strategy, String mode, String stage)
            throws Exception {
        try (var f = new Fixture(strategy, mode)) {
            var injected = new AtomicBoolean();
            String prefillAddress = f.a.ipPort();

            Runnable loseA =
                    () -> {
                        if (injected.compareAndSet(false, true))
                            f.a.getStatus().markRemovedFromDiscovery();
                    };
            switch (stage) {
                case "BEFORE_SELECTION" -> loseA.run();
                case "BEFORE_REGISTRATION" ->
                        doAnswer(
                                        call -> {
                                            loseA.run();
                                            return f.a;
                                        })
                                .when(f.registry)
                                .getEndpoint(RoleType.PREFILL, prefillAddress);
                case "AFTER_REGISTRATION" ->
                        doAnswer(
                                        call -> {
                                            loseA.run();
                                            return null;
                                        })
                                .when(f.cache)
                                .trackRoutingPrediction(
                                        any(), any(), any(), any(), anyLong(), anyLong(), any());
                case "BEFORE_PUBLICATION" ->
                        doAnswer(
                                        call -> {
                                            var result = call.callRealMethod();
                                            loseA.run();
                                            return result;
                                        })
                                .when(f.router)
                                .selectAndRegister(any(), any());
                case "AFTER_CONFIRMATION", "AFTER_CONFIRMATION_CLOSE" ->
                        doAnswer(
                                        call -> {
                                            var result = call.callRealMethod();
                                            loseA.run();
                                            if (stage.equals("AFTER_CONFIRMATION_CLOSE"))
                                                EndpointTestSupport.closeAndWait(f.a);
                                            return result;
                                        })
                                .when(f.a)
                                .confirmSelection(any(), any(), anyLong());
                default -> throw new AssertionError(stage);
            }
            f.submitAndAssertB();
            assertTrue(injected.get(), "fault injection must be reached");
            if (stage.equals("BEFORE_REGISTRATION"))
                verify(f.router).selectAndRegister(any(), any());
        }
    }

    @ParameterizedTest(name = "{0}/{1}/WAITING")
    @MethodSource("waitingCases")
    void unavailableWhileWaitingMustRecoverWithoutNotification(String strategy, String mode)
            throws Exception {
        try (var f = new Fixture(strategy, mode)) {
            var occupiedA =
                    f.a.tryRegisterSelection("occupied-a", f.a.selectionSnapshot(), 0, -1L).value();
            var occupiedB =
                    f.b.tryRegisterSelection("occupied-b", f.b.selectionSnapshot(), 0, -1L).value();
            assertNotNull(occupiedA);
            assertNotNull(occupiedB);
            var response = f.submitAndWaitForCapacity();
            f.a.getStatus().markRemovedFromDiscovery();
            assertTrue(f.b.rollbackSelection(occupiedB));
            f.assertB(response.get(1, TimeUnit.SECONDS));
            assertTrue(f.a.rollbackSelection(occupiedA));
            assertEquals(0, f.a.selectionSnapshot().outstandingRequests());
        }
    }

    @ParameterizedTest(name = "{0}/HOT_PREFILL_CAPACITY_RELEASED")
    @MethodSource("queueModes")
    void tokenBalanceWaitsForTheHotPrefillToReleaseCapacity(String mode) throws Exception {
        try (var f = new Fixture("TOKEN_BALANCE", mode)) {
            var occupied =
                    f.a.tryRegisterSelection("occupied-a", f.a.selectionSnapshot(), 0, -1L).value();
            assertNotNull(occupied);
            var response = f.submitAndWaitForCapacity();

            assertTrue(f.a.rollbackSelection(occupied));
            assertTrue(response.get(1, TimeUnit.SECONDS).isSuccess());
            assertSame(f.a, f.request.snapshot().route().prefillEndpoint());
        }
    }

    @ParameterizedTest(name = "{0}/NEW_HOT_PREFILL")
    @MethodSource("queueModes")
    void tokenBalanceReconsidersNewPrefillWhileTheOriginalLeaderIsFull(String mode)
            throws Exception {
        try (var f = new Fixture("TOKEN_BALANCE", mode)) {
            var discovered = new AtomicBoolean();
            when(f.registry.<PrefillEndpoint>endpoints(RoleType.PREFILL, null))
                    .thenAnswer(call -> discovered.get() ? List.of(f.a, f.b) : List.of(f.a));
            var initialCache = HostCacheMatch.fromLocalMatches(Map.of(f.a.ipPort(), 80));
            var updatedCache =
                    HostCacheMatch.fromLocalMatches(Map.of(f.a.ipPort(), 80, f.b.ipPort(), 80));
            when(f.cache.findMatchingEngines(any()))
                    .thenAnswer(
                            call ->
                                    new CacheMatchResult(
                                            discovered.get() ? updatedCache : initialCache,
                                            CacheMatchSource.LOCAL_SYNC,
                                            0,
                                            1));
            var occupied =
                    f.a.tryRegisterSelection("occupied-a", f.a.selectionSnapshot(), 0, -1L).value();
            assertNotNull(occupied);
            var response = f.submitAndWaitForCapacity();

            discovered.set(true);
            f.assertB(response.get(1, TimeUnit.SECONDS));
            assertEquals(1, f.a.selectionSnapshot().outstandingRequests());
            assertTrue(f.a.rollbackSelection(occupied));
        }
    }

    @ParameterizedTest(name = "{0}/{1}/DECODE/{2}")
    @MethodSource("selectionCases")
    void unavailableDecodeMustRecover(String strategy, String mode, String stage) throws Exception {
        try (var f = new Fixture(strategy, mode)) {
            var injected = new AtomicBoolean();

            String decodeAddress = f.decode.ipPort();
            Runnable loseA =
                    () -> {
                        if (injected.compareAndSet(false, true))
                            f.decode.getStatus().markRemovedFromDiscovery();
                    };
            switch (stage) {
                case "BEFORE_SELECTION" -> loseA.run();
                case "BEFORE_REGISTRATION" ->
                        doAnswer(
                                        call -> {
                                            loseA.run();
                                            return f.decode;
                                        })
                                .when(f.registry)
                                .getEndpoint(RoleType.DECODE, decodeAddress);
                case "AFTER_REGISTRATION" ->
                        doAnswer(
                                        call -> {
                                            var result = call.callRealMethod();
                                            loseA.run();
                                            return result;
                                        })
                                .when(f.decode)
                                .tryRegisterSelection(
                                        anyString(), any(), anyLong(), anyLong(), anyInt(), any());
                case "BEFORE_PUBLICATION" ->
                        doAnswer(
                                        call -> {
                                            var result = call.callRealMethod();
                                            loseA.run();
                                            return result;
                                        })
                                .when(f.router)
                                .selectAndRegister(any(), any());
                case "AFTER_CONFIRMATION", "AFTER_CONFIRMATION_CLOSE" ->
                        doAnswer(
                                        call -> {
                                            var result = call.callRealMethod();
                                            loseA.run();
                                            if (stage.equals("AFTER_CONFIRMATION_CLOSE"))
                                                EndpointTestSupport.closeAndWait(f.decode);
                                            return result;
                                        })
                                .when(f.decode)
                                .confirmSelection(any(), any());
                default -> throw new AssertionError(stage);
            }
            var response = f.scheduler.submit(f.request).get(1, TimeUnit.SECONDS);
            assertTrue(injected.get(), "fault injection must be reached");
            assertTrue(response.isSuccess(), "must reselect Decode B: " + response);
            assertSame(f.decodeB, f.request.snapshot().route().decodeEndpoint());
            assertEquals(0, f.decode.selectionSnapshot().usage().occupiedRequests());
            assertEquals(2, f.decodeB.selectionSnapshot().usage().occupiedRequests());
            assertEquals(1, f.a.selectionSnapshot().outstandingRequests());
        }
    }

    @ParameterizedTest(name = "{0}/{1}/DECODE_WAITING")
    @MethodSource("waitingCases")
    void unavailableDecodeWhileWaitingMustRecover(String strategy, String mode) throws Exception {
        try (var f = new Fixture(strategy, mode)) {
            var capacity = SchedulingTestConfig.capacity(f.config);
            var first =
                    f.decode
                            .tryRegisterSelection(
                                    "occupied-a1", f.decode.selectionSnapshot(), 1, 1, 50, capacity)
                            .value();
            var second =
                    f.decode
                            .tryRegisterSelection(
                                    "occupied-a2", f.decode.selectionSnapshot(), 1, 1, 50, capacity)
                            .value();
            var occupiedB =
                    f.decodeB
                            .tryRegisterSelection(
                                    "occupied-b", f.decodeB.selectionSnapshot(), 1, 1, 50, capacity)
                            .value();
            assertNotNull(first);
            assertNotNull(second);
            assertNotNull(occupiedB);
            var blocked = new CountDownLatch(1);
            doAnswer(
                            call -> {
                                var result = (SelectionResult<?, ?>) call.callRealMethod();
                                if (result.status() == SelectionResult.Status.BLOCKED)
                                    blocked.countDown();
                                return result;
                            })
                    .when(f.router)
                    .selectAndRegister(any(), any());
            var future = f.scheduler.submit(f.request);
            assertTrue(blocked.await(1, TimeUnit.SECONDS));
            RequestProtocolTestSupport.awaitCondition(
                    () -> f.request.currentSelectionAttempt() == null);
            assertFalse(future.isDone());
            assertEquals(
                    0,
                    f.a.selectionSnapshot().outstandingRequests(),
                    "partial Prefill must be rolled back before waiting for Decode");
            f.decode.getStatus().markRemovedFromDiscovery();
            assertTrue(f.decodeB.rollbackSelection(occupiedB));
            assertTrue(future.get(1, TimeUnit.SECONDS).isSuccess());
            assertSame(f.decodeB, f.request.snapshot().route().decodeEndpoint());
            assertTrue(f.decode.rollbackSelection(first));
            assertTrue(f.decode.rollbackSelection(second));
            assertEquals(0, f.decode.selectionSnapshot().usage().occupiedRequests());
        }
    }

    static Stream<Arguments> selectionCases() {
        return profiles()
                .flatMap(
                        strategy ->
                                modes().flatMap(
                                                mode ->
                                                        Stream.of(
                                                                        "BEFORE_SELECTION",
                                                                        "BEFORE_REGISTRATION",
                                                                        "AFTER_REGISTRATION",
                                                                        "BEFORE_PUBLICATION",
                                                                        "AFTER_CONFIRMATION",
                                                                        "AFTER_CONFIRMATION_CLOSE")
                                                                .map(
                                                                        stage ->
                                                                                Arguments.of(
                                                                                        strategy,
                                                                                        mode,
                                                                                        stage))));
    }

    static Stream<Arguments> waitingCases() {
        return profiles()
                .flatMap(strategy -> queueModes().map(mode -> Arguments.of(strategy, mode)));
    }

    static Stream<String> queueModes() {
        return modes().filter(mode -> !mode.equals("DIRECT"));
    }

    private static Stream<String> profiles() {
        return Stream.of("TOKEN_BALANCE", "COST_BASED", "CACHE_FIRST");
    }

    private static Stream<String> modes() {
        return Stream.of(
                "DIRECT", "FIFO", "FIFO_PRIORITY", "PREFILL_COST", "PREFILL_COST_PRIORITY");
    }

    /**
     * Runs ordinary capacity recovery with real P/D strategies, ledgers, queue and publication.
     * Decode preemption is tested separately; cache and registry lookups are controlled here.
     * Cache deliberately keeps advertising A after loss, so availability must override stale cache affinity.
     */
    private static final class Fixture implements AutoCloseable {
        final FlexlbConfig config = new FlexlbConfig();
        final EndpointRegistry registry = mock(EndpointRegistry.class);
        final CacheAwareService cache = mock(CacheAwareService.class);
        final PrefillEndpoint a;
        final PrefillEndpoint b;
        final DecodeEndpoint decode;
        final DecodeEndpoint decodeB;
        final RequestRouteSelector router;
        final AbstractRequestScheduler scheduler;
        final RequestContext request;

        Fixture(String strategy, String mode) {
            config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(2L);
            var prefill = config.getRouter().getRoles().getPrefill();
            prefill.getAdmission().setMaxInflightRequests(1);
            if (strategy.equals("TOKEN_BALANCE"))
                prefill.getSelection().setType(RoutingConfig.PrefillSelectionType.TOKEN_BALANCE);
            if (strategy.equals("CACHE_FIRST"))
                prefill.getSelection().setCacheAffinity(new RoutingConfig.CacheAffinityConfig());
            config.getScheduler()
                    .setType(
                            mode.equals("DIRECT")
                                    ? SchedulerConfig.Type.DIRECT
                                    : SchedulerConfig.Type.QUEUE);
            config.getScheduler().getOrdering().getPriority().setEnabled(mode.endsWith("PRIORITY"));
            config.getScheduler()
                    .getOrdering()
                    .getPolicy()
                    .setType(
                            mode.startsWith("PREFILL_COST")
                                    ? QueueOrderingConfig.PolicyType.PREFILL_COST
                                    : QueueOrderingConfig.PolicyType.FIFO);
            var service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(config);
            // Endpoint notifications are intentionally disconnected: periodic recovery must
            // suffice.
            a = spy(prefill("127.0.0.1"));
            b = prefill("127.0.0.2");
            decode =
                    spy(
                            EndpointTestSupport.decode(
                                    EndpointTestSupport.workerStatus(
                                            RoleType.DECODE, "127.0.0.3", 8080, 8081)));
            decodeB =
                    EndpointTestSupport.decode(
                            EndpointTestSupport.workerStatus(
                                    RoleType.DECODE, "127.0.0.4", 8080, 8081));
            var status = new WorkerStatusResponse();
            status.setAlive(true);
            status.setTotalKvCacheTokens(100000L);
            status.setAvailableKvCacheTokens(100000L);
            EndpointTestSupport.applyStatus(decode, status).run();
            EndpointTestSupport.applyStatus(decodeB, status).run();
            assertNotNull(
                    decodeB.tryRegisterSelection(
                                    "background",
                                    decodeB.selectionSnapshot(),
                                    1,
                                    1,
                                    10,
                                    SchedulingTestConfig.capacity(config))
                            .value());
            when(registry.<PrefillEndpoint>endpoints(RoleType.PREFILL, null))
                    .thenReturn(List.of(a, b));
            when(registry.<DecodeEndpoint>endpoints(RoleType.DECODE, null))
                    .thenReturn(List.of(decode, decodeB));
            when(registry.getEndpoint(RoleType.PREFILL, a.ipPort())).thenReturn(a);
            when(registry.getEndpoint(RoleType.PREFILL, b.ipPort())).thenReturn(b);
            when(registry.getEndpoint(RoleType.DECODE, decode.ipPort())).thenReturn(decode);
            when(registry.getEndpoint(RoleType.DECODE, decodeB.ipPort())).thenReturn(decodeB);
            var matches =
                    new CacheMatchResult(
                            HostCacheMatch.fromLocalMatches(Map.of(a.ipPort(), 80, b.ipPort(), 20)),
                            CacheMatchSource.LOCAL_SYNC,
                            0,
                            1);
            when(cache.findMatchingEngines(any())).thenReturn(matches);
            var metrics = mock(CacheMetricsReporter.class);
            PrefillSelectionStrategy policy =
                    strategy.equals("TOKEN_BALANCE")
                            ? new TokenBalancedPrefillStrategy(service, registry, cache, metrics)
                            : new CostBasedPrefillStrategy(service, registry, cache, metrics);
            var model = mock(ModelMetaConfig.class);
            when(model.requiredRoles()).thenReturn(List.of(RoleType.PREFILL, RoleType.DECODE));
            router =
                    spy(
                            new RequestRouteSelector(
                                    registry,
                                    policy,
                                    new DecodeStrategy(
                                            service,
                                            SchedulingTestConfig.capacity(config),
                                            registry),
                                    mock(VitStrategy.class),
                                    mock(EncoderStrategy.class),
                                    model));
            scheduler =
                    SchedulerTestSupport.create(
                            service,
                            mock(DeliveryMetricsReporter.class),
                            mock(RequestSchedulerReporter.class),
                            mock(TheoryCacheHitReporter.class),
                            router,
                            RequestPreemptionTestSupport.mock(),
                            new CapacityNotifier());
            request = new RequestContext(config);
            request.setRequest(
                    Request.builder()
                            .requestId("99010")
                            .seqLen(100L)
                            .blockSize(1L)
                            .maxNewTokens(1)
                            .build());
            request.setSchedulingMetadata(SchedulingMetadata.of(90, null, Long.MAX_VALUE, 50));
        }

        @Override
        public void close() {
            scheduler.cancel(request.getRequestId(), CancelReason.CLIENT_CANCELLED);
            EndpointTestSupport.closeAndWait(a);
            EndpointTestSupport.closeAndWait(b);
            EndpointTestSupport.closeAndWait(decode);
            EndpointTestSupport.closeAndWait(decodeB);
            SchedulerTestSupport.shutdown(SchedulerTestSupport.runtime(scheduler));
        }

        void submitAndAssertB() throws Exception {
            assertB(scheduler.submit(request).get(1, TimeUnit.SECONDS));
            assertEquals(
                    0,
                    a.selectionSnapshot().outstandingRequests(),
                    "A must not retain the failed attempt");
        }

        void assertB(Response response) {
            assertTrue(
                    response.isSuccess(),
                    "must recover instead of returning an error: " + response);
            assertSame(b, request.snapshot().route().prefillEndpoint());
            assertEquals(1, b.selectionSnapshot().outstandingRequests());
            assertEquals(1, decode.selectionSnapshot().usage().occupiedRequests());
        }

        CompletableFuture<Response> submitAndWaitForCapacity() throws InterruptedException {
            var blocked = new CountDownLatch(1);
            doAnswer(
                            call -> {
                                var result = (SelectionResult<?, ?>) call.callRealMethod();
                                if (result.status() == SelectionResult.Status.BLOCKED)
                                    blocked.countDown();
                                return result;
                            })
                    .when(router)
                    .selectAndRegister(any(), any());
            var response = scheduler.submit(request);
            assertTrue(
                    blocked.await(1, TimeUnit.SECONDS),
                    "request must actually encounter full capacity");
            // Selection finishes only after the scheduler has recorded the blocked result.
            RequestProtocolTestSupport.awaitCondition(
                    () -> request.currentSelectionAttempt() == null);
            assertFalse(response.isDone(), "capacity waiting must not publish an error");
            return response;
        }

        private PrefillEndpoint prefill(String ip) {
            var endpoint =
                    EndpointTestSupport.realPrefill(
                            EndpointTestSupport.workerStatus(RoleType.PREFILL, ip, 8080, 8081),
                            config,
                            null,
                            new RequestRepository(),
                            mock(DeliveryMetricsReporter.class));
            var status = new WorkerStatusResponse();
            status.setAlive(true);
            EndpointTestSupport.applyStatus(endpoint, status).run();
            return endpoint;
        }
    }
}
