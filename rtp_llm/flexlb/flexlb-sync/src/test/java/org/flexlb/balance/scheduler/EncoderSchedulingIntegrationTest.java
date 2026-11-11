package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.EncoderEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.preemption.DecodePreemption;
import org.flexlb.balance.strategy.DecodeSelectionStrategy;
import org.flexlb.balance.strategy.EncoderStrategy;
import org.flexlb.balance.strategy.PrefillSelectionStrategy;
import org.flexlb.balance.strategy.VitSelectionStrategy;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DecisionPolicyConfig;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.config.TrafficPolicyConfig;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RequestPhase;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.TheoryCacheHitReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Exercises Encoder selection, admission limits, and capacity wakeups through the request schedulers.
 */
class EncoderSchedulingIntegrationTest {
    private final ConfigService configService = mock(ConfigService.class);
    private final EndpointRegistry endpoints = mock(EndpointRegistry.class);
    private final List<EncoderEndpoint> workers = new CopyOnWriteArrayList<>();
    private final CapacityNotifier availability = new CapacityNotifier();

    private AbstractRequestScheduler lifecycle;
    private RequestRepository requests;
    private FlexlbConfig config;

    @ParameterizedTest
    @CsvSource({"false,false", "false,true", "true,false", "true,true"})
    void cancellationDuringEncoderCommitRejectsDeliveryAndReleasesAdmission(
            boolean queue, boolean beforeCommit) throws Exception {
        configure(queue, 1, List.of(RoleType.ENCODER));
        var original = worker("encoder", 8001, true, "group", Map.of(), 0, 0);
        var endpoint = org.mockito.Mockito.spy(original);
        workers.set(0, endpoint);
        var entered = new java.util.concurrent.CountDownLatch(1);
        var resume = new java.util.concurrent.CountDownLatch(1);
        org.mockito.Mockito.doAnswer(
                        call -> {
                            RouteResult route = call.getArgument(0);
                            assertFalse(Thread.holdsLock(route.ctx()));
                            boolean committed = !beforeCommit && (boolean) call.callRealMethod();
                            entered.countDown();
                            assertTrue(resume.await(5, TimeUnit.SECONDS));
                            return beforeCommit ? call.callRealMethod() : committed;
                        })
                .when(endpoint)
                .commit(
                        any(),
                        org.mockito.ArgumentMatchers.anyInt(),
                        org.mockito.ArgumentMatchers.anyLong());
        try (var executor = java.util.concurrent.Executors.newVirtualThreadPerTaskExecutor()) {
            try {
                var context = context("9900", 100, 0L);
                var submitted = executor.submit(() -> lifecycle.submit(context));
                assertTrue(entered.await(2, TimeUnit.SECONDS));
                assertEquals(beforeCommit ? 0L : 1L, endpoint.requestCount());
                executor.submit(
                                () ->
                                        lifecycle.cancel(
                                                "9900",
                                                0,
                                                CancelReason.CLIENT_CANCELLED,
                                                RequestPhase.ENCODER))
                        .get(2, TimeUnit.SECONDS);
                assertNotNull(context.snapshot().cancellation());
                assertNotNull(context.snapshot().routingContext());
                assertEquals(beforeCommit ? 0L : 1L, endpoint.requestCount());
                resume.countDown();
                assertEquals(
                        StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                        submitted.get(2, TimeUnit.SECONDS).get(2, TimeUnit.SECONDS).getCode());
                SchedulerTestSupport.awaitRequestWork(SchedulerTestSupport.runtime(lifecycle));
                assertEquals(0L, endpoint.requestCount());
                assertNull(context.snapshot().delivery());
                assertNull(requests.findActive("9900", RequestPhase.ENCODER));
            } finally {
                resume.countDown();
            }
        }
    }

    @ParameterizedTest
    @CsvSource({"false,false", "false,true", "true,false", "true,true"})
    void retirementDuringEncoderCommitReselectsWithoutLeakingOldAdmission(
            boolean queue, boolean beforeCommit) throws Exception {
        configure(queue, 1, List.of(RoleType.ENCODER));
        var first = org.mockito.Mockito.spy(worker("first", 8001, true, "group", Map.of(), 0, 0));
        workers.set(0, first);
        var entered = new java.util.concurrent.CountDownLatch(1);
        var resume = new java.util.concurrent.CountDownLatch(1);
        org.mockito.Mockito.doAnswer(
                        call -> {
                            RouteResult route = call.getArgument(0);
                            assertFalse(Thread.holdsLock(route.ctx()));
                            boolean committed = !beforeCommit && (boolean) call.callRealMethod();
                            entered.countDown();
                            assertTrue(resume.await(5, TimeUnit.SECONDS));
                            return beforeCommit ? call.callRealMethod() : committed;
                        })
                .when(first)
                .commit(
                        any(),
                        org.mockito.ArgumentMatchers.anyInt(),
                        org.mockito.ArgumentMatchers.anyLong());
        try (var executor = java.util.concurrent.Executors.newVirtualThreadPerTaskExecutor()) {
            try {
                var submitted = executor.submit(() -> submit(9901, 100, 0L));
                assertTrue(entered.await(2, TimeUnit.SECONDS));
                assertEquals(beforeCommit ? 0L : 1L, first.requestCount());
                workers.remove(first);
                var replacement = worker("replacement", 8002, true, "group", Map.of(), 0, 0);
                executor.submit(first::close).get(2, TimeUnit.SECONDS);
                assertTrue(first.isRetired());
                assertEquals(0L, first.requestCount());
                resume.countDown();
                assertEquals("replacement", selected(submitted.get(2, TimeUnit.SECONDS)));
                SchedulerTestSupport.awaitRequestWork(SchedulerTestSupport.runtime(lifecycle));
                assertEquals(0L, first.requestCount());
                assertEquals(1L, replacement.requestCount());
            } finally {
                resume.countDown();
            }
        }
    }

    @ParameterizedTest
    @CsvSource({"false,false", "false,true", "true,false", "true,true"})
    void retiredEncoderIsReselectedDuringConstructionOrAdmission(
            boolean queue, boolean beforeCommit) throws Exception {
        configure(queue, 1, List.of(RoleType.ENCODER), false);
        var first = worker("first", 8001, true, "group", Map.of(), 0, 0);
        var replacement = new java.util.concurrent.atomic.AtomicReference<EncoderEndpoint>();
        Runnable retire =
                () -> {
                    first.close();
                    workers.remove(first);
                    replacement.set(worker("replacement", 8002, true, "group", Map.of(), 0, 0));
                };
        if (beforeCommit) {
            org.mockito.Mockito.doAnswer(
                            call -> {
                                if (replacement.get() == null) retire.run();
                                return call.callRealMethod();
                            })
                    .when(lifecycle)
                    .currentRoutingContext(any());
        } else {
            org.mockito.Mockito.doAnswer(
                            call -> {
                                retire.run();
                                return first;
                            })
                    .when(endpoints)
                    .selectEndpoint(RoleType.ENCODER, first.address());
        }
        lifecycle.start();
        assertEquals("replacement", selected(submit(9001, 100, 0L)));
        assertEquals(0L, first.requestCount());
        assertEquals(1L, replacement.get().requestCount());
    }

    @BeforeEach
    void setUp() {
        when(endpoints.encoderRoutingSnapshot()).thenAnswer(call -> List.copyOf(workers));
        when(endpoints.selectEndpoint(eq(RoleType.ENCODER), anyString()))
                .thenAnswer(
                        call ->
                                workers.stream()
                                        .filter(
                                                worker ->
                                                        worker.address()
                                                                .equals(call.getArgument(1)))
                                        .findFirst()
                                        .orElse(null));
        configure(false, 1, List.of(RoleType.ENCODER));
    }

    @AfterEach
    void stop() {
        SchedulerTestServices.forOwner(lifecycle).shutdown(ShutdownMode.IMMEDIATE, Duration.ZERO);
        for (var endpoint : workers) {
            endpoint.close();
        }
        workers.clear();
    }

    @Test
    void queueLimitsEachEncoderAndResumesAfterWorkerFinished() throws Exception {
        configure(true, 1, List.of(RoleType.ENCODER));
        var first = worker("first", 8001, true, "group", Map.of(), 0, 0);
        worker("second", 8002, true, "group", Map.of(), 0, 0);
        assertEquals("first", selected(submit(1, 100, 0L)));
        assertEquals("second", selected(submit(2, 100, 0L)));
        var waiting = submit(3, 100, 0L);
        assertFalse(waiting.isDone());
        publish(first, Map.of(), Map.of("1", task("1", 100)), 0, 0);
        assertEquals("first", selected(waiting));
        assertNotNull(requests.getRequestState("1", 0, RequestPhase.ENCODER));
    }

    @ParameterizedTest
    @CsvSource({"false", "true"})
    void encoderPublishesOnlyItsRouteWithoutGenerationInput(boolean queued) throws Exception {
        configure(queued, 1, List.of(RoleType.ENCODER));
        var encoder = worker("encoder", 8001, true, "group", Map.of(), 0, 0);
        var context = context("encoder-route", 100, 0L);
        assertFalse(context.data().hasGenerateInput());

        var response = lifecycle.submit(context).get(2, TimeUnit.SECONDS);

        assertTrue(response.isSuccess(), response.getErrorMessage());
        assertEquals(1, response.getServerStatus().size());
        assertEquals("encoder", response.getServerStatus().getFirst().getServerIp());
        var delivery = context.snapshot().delivery();
        assertNotNull(delivery);
        assertEquals(Delivery.SendMode.ROUTE_DECISION, delivery.identity().kind());
        assertNull(delivery.identity().route().prefillEndpoint());
        assertNull(delivery.identity().route().decodeEndpoint());
        assertEquals(1, encoder.requestCount());

        publish(encoder, Map.of(), Map.of("encoder-route", task("encoder-route", 100)), 0, 0);
        SchedulerTestSupport.awaitRequestWork(SchedulerTestServices.forOwner(lifecycle));
        assertEquals(0, encoder.requestCount());
    }

    @Test
    void queueUsesRunningAndWaitingCountsAndCancelsWaitingRequest() throws Exception {
        configure(true, 2, List.of(RoleType.ENCODER));
        worker("busy", 8001, true, "group", Map.of(), 1, 1);
        worker("open", 8002, true, "group", Map.of(), 0, 1);
        assertEquals("open", selected(submit(4, 100, 0L)));
        var waiting = submit(5, 100, 0L);
        assertFalse(waiting.isDone());
        lifecycle.cancel("5", 0, CancelReason.CLIENT_CANCELLED, RequestPhase.ENCODER);
        assertEquals(
                StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                waiting.get(2, TimeUnit.SECONDS).getCode());
    }

    @ParameterizedTest
    @CsvSource({"SINGLE,NON_BATCH", "SINGLE,BATCH", "FIXED_WINDOW,NON_BATCH", "FIXED_WINDOW,BATCH"})
    void generationDecisionAndTransportDoNotDelayEncoderRoute(
            DecisionPolicyConfig.Type decision, DispatcherConfig.Type transport) throws Exception {
        configure(
                true,
                1,
                List.of(RoleType.ENCODER, RoleType.PREFILL, RoleType.DECODE),
                true,
                configuration -> {
                    configuration.getDispatcher().setType(transport);
                    configuration.getScheduler().getDecision().setType(decision);
                    if (decision == DecisionPolicyConfig.Type.FIXED_WINDOW) {
                        configuration.getScheduler().getDecision().setMaxRequests(8);
                        configuration.getScheduler().getDecision().setMaxCollectionWaitMs(60_000L);
                    }
                });
        var encoder = worker("encoder", 8001, true, "group", Map.of(), 0, 0);
        var request = context("encoder-policy", 100, 0L);

        var response = lifecycle.submit(request).get(2, TimeUnit.SECONDS);

        assertTrue(response.isSuccess(), response.getErrorMessage());
        assertFalse(request.data().hasGenerateInput());
        assertEquals(1, response.getServerStatus().size());
        assertEquals("encoder", response.getServerStatus().getFirst().getServerIp());
        assertEquals(
                Delivery.SendMode.ROUTE_DECISION, request.snapshot().delivery().identity().kind());
        assertEquals(1, encoder.requestCount());
        org.mockito.Mockito.verify(
                        SchedulerTestSupport.engineClient(lifecycle), org.mockito.Mockito.never())
                .batchEnqueueAsync(anyString(), org.mockito.ArgumentMatchers.anyInt(), any());
    }

    @ParameterizedTest
    @CsvSource({"false", "true"})
    void encoderPhaseDefaultsToEncoderEvenWhenModelAlsoHasPrefillAndDecode(boolean queued)
            throws Exception {
        configure(queued, 1, List.of(RoleType.ENCODER, RoleType.PREFILL, RoleType.DECODE));
        worker("encoder", 8001, true, "group", Map.of(), 0, 0);
        var request = context("implicit-encoder", 100, 0L);
        request.data().setRequestedRoles(List.of());

        var response = lifecycle.submit(request).get(2, TimeUnit.SECONDS);

        assertTrue(response.isSuccess(), response.getErrorMessage());
        var route = request.snapshot().delivery().identity().route();
        assertEquals(1, route.roles().size());
        assertNotNull(route.encoderEndpoint());
        assertNull(route.prefillEndpoint());
        assertNull(route.decodeEndpoint());
        assertNull(requests.findActive("implicit-encoder", RequestPhase.GENERATION));
    }

    @Test
    void cancelledWaiterAndRepeatedFinishedReportDoNotConsumeOrReleaseAnotherAdmission()
            throws Exception {
        configure(true, 1, List.of(RoleType.ENCODER));
        var encoder = worker("encoder", 8001, true, "group", Map.of(), 0, 0);
        assertEquals("encoder", selected(submit(101, 100, 0L)));
        var cancelled = context("102", 100, 0L);
        var cancelledResponse = lifecycle.submit(cancelled);
        var next = submit(103, 100, 0L);
        assertFalse(cancelledResponse.isDone());
        assertFalse(next.isDone());
        RequestProtocolTestSupport.awaitGlobalCapacityWaiters(lifecycle, 2);

        lifecycle.cancel("102", 0, CancelReason.CLIENT_CANCELLED, RequestPhase.ENCODER);
        assertEquals(
                StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                cancelledResponse.get(2, TimeUnit.SECONDS).getCode());
        SchedulerTestSupport.awaitRequestWork(SchedulerTestServices.forOwner(lifecycle));
        assertNull(cancelled.snapshot().delivery());
        assertNull(requests.findActive("102", RequestPhase.ENCODER));
        assertEquals(1, encoder.requestCount());
        RequestProtocolTestSupport.awaitGlobalCapacityWaiters(lifecycle, 1);

        publish(encoder, Map.of(), Map.of("101", task("101", 100)), 0, 0);
        assertEquals("encoder", selected(next));
        publish(encoder, Map.of(), Map.of("101", task("101", 100)), 0, 0);
        SchedulerTestSupport.awaitRequestWork(SchedulerTestServices.forOwner(lifecycle));
        assertEquals(1, encoder.requestCount());
        assertNotNull(requests.findActive("103", RequestPhase.ENCODER));

        publish(encoder, Map.of(), Map.of("103", task("103", 100)), 0, 0);
        SchedulerTestSupport.awaitRequestWork(SchedulerTestServices.forOwner(lifecycle));
        assertEquals(0, encoder.requestCount());
        assertEquals(0, encoder.uncachedTokens());
        assertNull(requests.findActive("103", RequestPhase.ENCODER));
    }

    @Test
    void statusCapacityChangeWakesWaitingEncoderWithoutAFinishedTask() throws Exception {
        configure(true, 1, List.of(RoleType.ENCODER));
        var worker = worker("encoder", 8001, true, "group", Map.of(), 1, 0);
        var waiting = submit(10, 100, 0L);
        assertFalse(waiting.isDone());
        publish(worker, Map.of(), Map.of(), 0, 0);
        assertEquals("encoder", selected(waiting));
    }

    @Test
    void queuedEncoderExpiresAndLeavesNoQueueMembership() throws Exception {
        configure(true, 1, List.of(RoleType.ENCODER));
        var worker = worker("busy", 8001, true, "group", Map.of(), 1, 0);
        var context = context("7", 100, 0L);
        context.data()
                .setSchedulingMetadata(
                        SchedulingMetadata.explicit(50, System.currentTimeMillis() + 300));
        var response = lifecycle.submit(context).get(2, TimeUnit.SECONDS);
        assertEquals(StrategyErrorType.QUEUE_TIMEOUT.getErrorCode(), response.getCode());
        publish(worker, Map.of(), Map.of(), 0, 0);
        assertEquals("busy", selected(submit(8, 100, 0L)));
        assertEquals(1, worker.requestCount());
    }

    @Test
    void rejectedEncoderDoesNotLeaveQueueMembership() throws Exception {
        configure(true, 1, List.of(RoleType.PREFILL));
        var response = lifecycle.submit(context("rejected", 100, 0L)).get(2, TimeUnit.SECONDS);
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(), response.getCode());
        assertNull(requests.findActive("rejected", RequestPhase.ENCODER));
    }

    @Test
    void shutdownCompletesWaitingEncoder() throws Exception {
        configure(true, 1, List.of(RoleType.ENCODER));
        worker("busy", 8001, true, "group", Map.of(), 1, 0);
        var waiting = submit(8, 100, 0L);
        assertFalse(waiting.isDone());
        SchedulerTestServices.forOwner(lifecycle).shutdown(ShutdownMode.IMMEDIATE, Duration.ZERO);
        assertFalse(waiting.get(2, TimeUnit.SECONDS).isSuccess());
    }

    @Test
    void weightedRequestChoosesLessObservedUncachedWorkDespiteMoreConcurrentTasks()
            throws Exception {
        worker("long", 8001, true, "group", Map.of("old", task("old", 1000)), 1, 0);
        worker("short", 8002, true, "group", Map.of("a", task("a", 10), "b", task("b", 10)), 4, 0);
        assertEquals("short", selected(submit(1, 800, 200L)));
        assertEquals(
                RequestPhase.ENCODER,
                requests.getRequestState("1", 0, RequestPhase.ENCODER).requestPhase());
    }

    @Test
    void absentCacheEstimateUsesConcurrencyIncludingLocallyPendingSelections() throws Exception {
        var idle = worker("idle", 8001, true, "group", Map.of(), 0, 0);
        worker("busy", 8002, true, "group", Map.of("old", task("old", 1)), 1, 0);
        assertEquals("idle", selected(submit(2, 800, null)));
        assertEquals(1, idle.requestCount());
        assertEquals("idle", selected(submit(3, 800, null)));
        assertEquals(2, idle.requestCount());
        assertEquals("busy", selected(submit(6, 800, null)));
    }

    @Test
    void weightedTieUsesConcurrencyAndFullHitStillRoutesToEncoder() throws Exception {
        worker("busy", 8001, true, "group", Map.of("old", task("old", 0)), 1, 0);
        worker("idle", 8002, true, "group", Map.of(), 0, 0);
        assertEquals("idle", selected(submit(4, 800, 800L)));
    }

    @Test
    void actualWorkerInputReplacesPendingEstimateWithoutDoubleCounting() throws Exception {
        var encoder = worker("encoder", 8001, true, "group", Map.of(), 0, 0);
        worker("other", 8002, true, "group", Map.of("old", task("old", 500)), 1, 0);
        assertEquals("encoder", selected(submit(7, 1000, 200L)));
        assertEquals(800, encoder.uncachedTokens());
        assertEquals("other", selected(submit(8, 1000, 200L)));
        publish(encoder, Map.of("7", task("7", 100)), Map.of(), 1, 0);
        assertEquals(1, encoder.requestCount());
        assertEquals(100, encoder.uncachedTokens());
        assertEquals("encoder", selected(submit(9, 1000, 200L)));
    }

    @Test
    void deadAndWrongGroupWorkersCannotWin() throws Exception {
        worker("dead", 8001, false, "group", Map.of(), 0, 0);
        worker("foreign", 8002, true, "other", Map.of(), 0, 0);
        var group = mock(TrafficPolicyConfig.class);
        config.getRouter().setGroupSelector(group);
        when(group.resolveTargetGroup(anyString(), any(), anyLong()))
                .thenReturn(java.util.Optional.of("group"));
        var result = submit(5, 800, 0L).get(2, TimeUnit.SECONDS);
        assertEquals(StrategyErrorType.NO_ENCODER_WORKER.getErrorCode(), result.getCode());
    }

    @Test
    void stringEncoderIdCanBeCancelledWithoutTouchingGenerationRequest() throws Exception {
        configure(true, 1, List.of(RoleType.ENCODER));
        worker("busy", 8001, true, "group", Map.of(), 1, 0);
        var context = context("shared-id", 100, 0L);
        var pending = lifecycle.submit(context);
        assertNull(
                lifecycle.cancel(
                        "shared-id", 0, CancelReason.CLIENT_CANCELLED, RequestPhase.GENERATION));
        assertFalse(pending.isDone());
        lifecycle.cancel("shared-id", 0, CancelReason.CLIENT_CANCELLED, RequestPhase.ENCODER);
        assertEquals(
                StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                pending.get(2, TimeUnit.SECONDS).getCode());
    }

    private void configure(boolean queue, int limit, List<RoleType> roles) {
        configure(queue, limit, roles, true);
    }

    private void configure(boolean queue, int limit, List<RoleType> roles, boolean start) {
        configure(queue, limit, roles, start, configuration -> {});
    }

    private void configure(
            boolean queue,
            int limit,
            List<RoleType> roles,
            boolean start,
            java.util.function.Consumer<FlexlbConfig> customize) {
        if (lifecycle != null) stop();
        config = SchedulingTestConfig.batchConfig();
        if (queue) SchedulingTestConfig.useFifoQueue(config);
        else config.setScheduler(org.flexlb.config.ConfigTestSupport.direct());
        SchedulingTestConfig.useNonBatchDispatcher(config);
        config.getDispatcher().setMaxInflightPerEncoderWorker(limit);
        customize.accept(config);
        when(configService.loadBalanceConfig()).thenReturn(config);
        var model = mock(ModelMetaConfig.class);
        when(model.requiredRoles()).thenReturn(roles);
        var selector =
                new RequestRouteSelector(
                        endpoints,
                        mock(
                                PrefillSelectionStrategy.class,
                                org.mockito.Mockito.CALLS_REAL_METHODS),
                        mock(DecodeSelectionStrategy.class, org.mockito.Mockito.CALLS_REAL_METHODS),
                        mock(VitSelectionStrategy.class),
                        new EncoderStrategy(),
                        mock(CacheAwareService.class),
                        null,
                        mock(EngineHealthReporter.class),
                        mock(CacheMetricsReporter.class),
                        configService,
                        model);
        lifecycle =
                SchedulerTestSupport.create(
                        configService,
                        mock(DeliveryMetricsReporter.class),
                        mock(RequestSchedulerReporter.class),
                        mock(TheoryCacheHitReporter.class),
                        selector,
                        mock(DecodePreemption.class),
                        availability,
                        start);
        requests = SchedulerTestSupport.repository(lifecycle);
    }

    private CompletableFuture<Response> submit(long id, long tokens, Long hit) {
        return lifecycle.submit(context(Long.toString(id), tokens, hit));
    }

    private RequestContext context(String id, long tokens, Long hit) {
        var context = new RequestContext(config);
        context.data()
                .setRequest(
                        Request.builder()
                                .requestId(id)
                                .seqLen(tokens)
                                .encoderCacheHitLen(hit)
                                .maxNewTokens(16)
                                .build());
        context.data().setPhase(RequestPhase.ENCODER);
        context.data().setRequestedRoles(List.of(RoleType.ENCODER));
        context.data()
                .setSchedulingMetadata(
                        SchedulingMetadata.explicit(50, System.currentTimeMillis() + 60_000));
        return context;
    }

    private static String selected(CompletableFuture<Response> future) throws Exception {
        var result = future.get(2, TimeUnit.SECONDS);
        assertTrue(result.isSuccess(), result.getErrorMessage());
        return result.getServerStatus().getFirst().getServerIp();
    }

    private EncoderEndpoint worker(
            String ip,
            int port,
            boolean alive,
            String group,
            Map<String, WorkerStatus.TaskObservation> tasks,
            long running,
            long waiting) {
        var status =
                WorkerStatus.createDiscovered(RoleType.ENCODER, group, ip, port, port + 1, "site");
        var endpoint = new EncoderEndpoint(status, availability);
        publish(endpoint, tasks, Map.of(), running, waiting);
        status.lock.lock();
        try {
            status.recordSuccessfulPoll(alive);
        } finally {
            status.lock.unlock();
        }
        workers.add(endpoint);
        return endpoint;
    }

    private static void publish(
            EncoderEndpoint endpoint,
            Map<String, WorkerStatus.TaskObservation> active,
            Map<String, WorkerStatus.TaskObservation> finished,
            long running,
            long waiting) {
        var status = endpoint.getStatus();
        org.flexlb.balance.endpoint.WorkerEndpoint.StatusChange change;
        status.lock.lock();
        try {
            long version = Math.max(1L, status.appliedStatusCursor().statusVersion() + 1);
            var engine =
                    new WorkerStatus.EngineObservation(
                            RoleType.ENCODER,
                            null,
                            0,
                            0,
                            active,
                            0,
                            0,
                            1,
                            1,
                            0,
                            0,
                            0,
                            running,
                            waiting);
            var observation =
                    status.bindStatusObservation(engine, true, version, version, finished);
            change = endpoint.applyWorkerStatus(status, status.prepareStatus(observation));
        } finally {
            status.lock.unlock();
        }
        endpoint.publishStatus(change);
    }

    private static WorkerStatus.TaskObservation task(String id, long tokens) {
        var task = mock(WorkerStatus.TaskObservation.class);
        when(task.requestId()).thenReturn(id);
        when(task.inputLength()).thenReturn(tokens);
        return task;
    }
}
