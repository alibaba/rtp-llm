package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.DeliverySettlementTestSupport;
import org.flexlb.balance.endpoint.EndpointTestSupport;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.prediction.FormulaPredictor;
import org.flexlb.balance.scheduler.ExpirationTimer.Token;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DecisionPolicyConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.TheoryCacheHitReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;
import java.util.OptionalLong;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import static org.flexlb.balance.endpoint.EndpointTestSupport.runWithStateLock;
import static org.flexlb.balance.scheduler.RequestProtocolTestSupport.await;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.argThat;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

/**
 * Real executor, request context and endpoint ledgers; only the Engine transport is simulated.
 */
class QueuedBatchDeliveryTest {
    private static final long TIMEOUT_MS = TimeUnit.HOURS.toMillis(1);

    private final CountDownLatch releaseExecutor = new CountDownLatch(1);
    private final CompletableFuture<EngineRpcService.EnqueueBatchResponsePB> reply =
            new CompletableFuture<>();

    private DeliverySettlementTestSupport ledger = new DeliverySettlementTestSupport();
    private FlexlbConfig config;
    private AbstractRequestScheduler registry;
    private DeliveryCoordinator dispatcher;
    private EngineGrpcClient grpc;
    private PrefillEndpoint prefill;
    private DecodeEndpoint decode;
    private PrefillQueueScheduler worker;

    private final java.util.concurrent.CopyOnWriteArrayList<EngineRpcService.EnqueueBatchRequestPB>
            sent = new java.util.concurrent.CopyOnWriteArrayList<>();

    private PrefillQueueScheduler manualScheduling;

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void withdrawalFinishesBeforeTheOldDeliveryTaskResumes(boolean cancel) throws Exception {
        var route = item(99003L);
        var context = route.ctx();
        var encoding = new CountDownLatch(1);
        var resumeSend = new CountDownLatch(1);
        var prepared = new java.util.concurrent.atomic.AtomicReference<Delivery>();
        var requeues = new java.util.concurrent.atomic.AtomicInteger();
        var scheduler = (QueuedRequestScheduler) SchedulerTestServices.scheduler(registry);
        doAnswer(
                        call -> {
                            requeues.incrementAndGet();
                            return true;
                        })
                .when(scheduler)
                .enqueue(any());
        doAnswer(
                        call -> {
                            prepared.set(call.getArgument(0));
                            return call.callRealMethod();
                        })
                .when(org.flexlb.balance.scheduler.DeliveryTestSupport.transport(dispatcher))
                .dispatch(any());
        var dispatchConfig = org.mockito.Mockito.spy(config.getDispatcher());
        org.mockito.Mockito.doReturn(dispatchConfig).when(config).getDispatcher();
        doAnswer(
                        call -> {
                            encoding.countDown();
                            assertTrue(resumeSend.await(2, TimeUnit.SECONDS));
                            return call.callRealMethod();
                        })
                .when(dispatchConfig)
                .getFetchAttachTimeoutMs();
        try {
            submit(List.of(route));
            releaseExecutor.countDown();
            assertTrue(encoding.await(2, TimeUnit.SECONDS));
            var oldClaim = prepared.get().members.getFirst();
            assertNull(context.snapshot().delivery());
            assertEquals(Delivery.SendMode.NONE, oldClaim.kind());
            assertEquals(1, decode.resourceSnapshot().activeDispatchPermits());
            assertTrue(
                    registry.onRouteFailure(
                            route,
                            ContextResult.RouteFailureReason.ENDPOINT_UNAVAILABLE,
                            "Endpoint failed before send"));
            RequestProtocolTestSupport.awaitCondition(() -> requeues.get() == 1);
            assertNull(context.snapshot().route());
            assertEquals(0, decode.resourceSnapshot().reservedCount());
            assertEquals(0, decode.resourceSnapshot().activeDispatchPermits());
            assertOccupancy(0, 0);
            assertFalse(
                    registry.onRouteFailure(
                            route,
                            ContextResult.RouteFailureReason.ENDPOINT_UNAVAILABLE,
                            "Duplicate failure during cleanup"));
            if (cancel) registry.cancel(context, 0L, CancelReason.CLIENT_CANCELLED);
            resumeSend.countDown();
            awaitDispatchTasks();
            if (cancel)
                assertFalse(context.data().getFuture().get(2, TimeUnit.SECONDS).isSuccess());
            else RequestProtocolTestSupport.awaitCondition(() -> requeues.get() == 1);
            assertTrue(sent.isEmpty());
            assertNull(context.snapshot().route());
            assertEquals(0, decode.resourceSnapshot().reservedCount());
            assertOccupancy(0, 0);
            registry.onDeliveryResult(
                    new ContextResult.TransportResult(
                            oldClaim,
                            Delivery.Result.notSent(
                                    new IllegalStateException("Late old completion")),
                            System.currentTimeMillis()));
            assertEquals(1, requeues.get());
            assertEquals(cancel, context.data().getFuture().isDone());
        } finally {
            resumeSend.countDown();
        }
    }

    @Test
    void endpointFailureAfterBatchPreparationWithdrawsAllRolesAndPreventsOldSend()
            throws Exception {
        var route = item(99001L);
        var context = route.ctx();
        var requeues = new java.util.concurrent.atomic.AtomicInteger();
        var scheduler = (QueuedRequestScheduler) SchedulerTestServices.scheduler(registry);
        doAnswer(
                        call -> {
                            assertNull(context.snapshot().route());
                            assertEquals(0, decode.resourceSnapshot().reservedCount());
                            assertEquals(0, decode.resourceSnapshot().activeDispatchPermits());
                            assertOccupancy(0, 0);
                            requeues.incrementAndGet();
                            return true;
                        })
                .when(scheduler)
                .enqueue(any());
        submit(List.of(route));
        assertEquals(1, ledger.prefill.stats().batchCount());
        decode.getStatus().markRemovedFromDiscovery();
        assertTrue(
                registry.onRouteFailure(
                        route,
                        ContextResult.RouteFailureReason.ENDPOINT_UNAVAILABLE,
                        "Decode unavailable after batch preparation"));
        RequestProtocolTestSupport.awaitCondition(() -> requeues.get() == 1);
        assertNull(context.snapshot().route());
        assertEquals(0, decode.resourceSnapshot().activeDispatchPermits());
        assertFalse(context.data().getFuture().isDone());
        releaseExecutor.countDown();
        awaitDispatchTasks();
        RequestProtocolTestSupport.awaitCondition(() -> requeues.get() == 1);
        assertTrue(sent.isEmpty());
        assertEquals(1, requeues.get());
        assertNull(context.snapshot().route());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void oldPreparedPermitCannotChangeAReplacementRegistration(boolean closeOldPermitFirst)
            throws Exception {
        var route = item(99004L);
        var context = route.ctx();
        var prepared = new java.util.concurrent.atomic.AtomicReference<Delivery>();
        var requeues = new java.util.concurrent.atomic.AtomicInteger();
        doAnswer(
                        call -> {
                            prepared.set(call.getArgument(0));
                            return call.callRealMethod();
                        })
                .when(org.flexlb.balance.scheduler.DeliveryTestSupport.transport(dispatcher))
                .dispatch(any());
        doAnswer(
                        call -> {
                            requeues.incrementAndGet();
                            return true;
                        })
                .when((QueuedRequestScheduler) SchedulerTestServices.scheduler(registry))
                .enqueue(any());
        submit(List.of(route));
        var oldPermit = prepared.get().members.getFirst().decode();
        assertNotNull(oldPermit);
        assertEquals(1, decode.resourceSnapshot().activeDispatchPermits());
        assertTrue(
                registry.onRouteFailure(
                        route,
                        ContextResult.RouteFailureReason.ENDPOINT_UNAVAILABLE,
                        "withdraw before the prepared delivery task starts"));
        RequestProtocolTestSupport.awaitCondition(() -> requeues.get() == 1);
        assertNull(context.snapshot().route());
        assertEquals(0, decode.resourceSnapshot().activeDispatchPermits());
        assertEquals(0, decode.resourceSnapshot().reservedCount());
        assertOccupancy(0, 0);

        var replacementRecord = EndpointTestSupport.registerDecode(decode, 99004L, 1L, 2L, 50);
        var replacement =
                RequestExecutionTestSupport.registerPrefillRoute(
                        context,
                        prefill,
                        decode,
                        replacementRecord,
                        0L,
                        System.currentTimeMillis());
        RequestProtocolTestSupport.bind(
                registry,
                new RequestProtocolTestSupport.Registered(replacement, context.data().getFuture()));
        ledger.enqueue(replacement);
        var acquired =
                decode.acquireDispatchPermit(
                        replacementRecord, SchedulingTestConfig.capacity(config));
        assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.ACQUIRED, acquired.status());
        var prefillBeforeOldTask = ledger.prefill.stats();
        var queueBeforeOldTask = ledger.prefill.captureQueue(10).items();
        var decodeBeforeOldTask = decode.resourceSnapshot();
        try {
            if (closeOldPermitFirst) oldPermit.release();
            assertEquals(
                    DecodeResources.EngineDispatchPermitTransferStatus.OWNERSHIP_LOST,
                    oldPermit.dispatch().status());
            oldPermit.release();
            assertEquals(1, decode.resourceSnapshot().activeDispatchPermits());
            assertEquals(replacementRecord, EndpointTestSupport.decodeReservation(decode, 99004L));
            releaseExecutor.countDown();
            awaitDispatchTasks();
            assertTrue(sent.isEmpty());
            assertSame(replacement, context.snapshot().route());
            assertEquals(1, decode.resourceSnapshot().activeDispatchPermits());
            assertEquals(1, decode.resourceSnapshot().reservedCount());
            assertEquals(1, decode.routingView().engineCapacityUsed());
            assertOccupancy(0, 0);
            assertEquals(prefillBeforeOldTask, ledger.prefill.stats());
            assertEquals(queueBeforeOldTask, ledger.prefill.captureQueue(10).items());
            assertEquals(decodeBeforeOldTask, decode.resourceSnapshot());
            assertEquals(
                    1L,
                    org.flexlb.balance.endpoint.EndpointTestSupport.occupiedRequestCount(
                            ledger.prefill));
            assertEquals(List.of(replacement), ledger.prefill.captureQueue(10).items());
            assertEquals(1, requeues.get());
        } finally {
            acquired.value().release();
            registry.cancel(context, 0L, CancelReason.CLIENT_CANCELLED);
        }
    }

    @Test
    void failureAfterSendStartsCannotWithdrawOrRequeue() throws Exception {
        var route = item(99002L);
        submit(List.of(route));
        releaseExecutor.countDown();
        RequestProtocolTestSupport.awaitCondition(() -> !sent.isEmpty());
        assertFalse(
                registry.onRouteFailure(
                        route,
                        ContextResult.RouteFailureReason.ENDPOINT_UNAVAILABLE,
                        "Endpoint unavailable after invocation"));
        assertSame(route, route.ctx().snapshot().route());
        verify(
                        (QueuedRequestScheduler) SchedulerTestServices.scheduler(registry),
                        org.mockito.Mockito.never())
                .enqueue(any());
        reply.complete(ack(99002L));
    }

    @Test
    void dispatchCapacityAndListenerUseTheBoundFacilitiesWithoutReadingTheRequest() {
        var item = mock(RouteResult.class);
        var boundary = CapacityBoundary.unavailable(CapacityBoundary.Resource.DISPATCH_EXECUTOR);
        assertEquals(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry)
                        .deliveries()
                        .deliveryCapacityAvailable(),
                worker.capacityAvailable(item, boundary));
        worker.watchCapacity(item, boundary, () -> {}).run();
        verifyNoInteractions(item);
    }

    @BeforeEach
    void setUp(org.junit.jupiter.api.TestInfo test) {
        config = org.mockito.Mockito.spy(SchedulingTestConfig.batchConfig());
        if (test.getTags().contains("non-batch")) {
            config.getDispatcher().setType(org.flexlb.config.DispatcherConfig.Type.NON_BATCH);
        }
        var sizing = org.mockito.Mockito.spy(config.getInternalRuntime());
        org.mockito.Mockito.doReturn(1).when(sizing).getBatchDispatchThreads();
        org.mockito.Mockito.doReturn(1).when(sizing).getBatchDispatchQueueCapacity();
        org.mockito.Mockito.doReturn(sizing).when(config).getInternalRuntime();
        config.getRequestLifecycle().getRequest().setTimeoutMs(TIMEOUT_MS);
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        DeliveryMetricsReporter reporter = mock(DeliveryMetricsReporter.class);
        registry =
                SchedulerTestSupport.create(
                        service,
                        reporter,
                        mock(RequestSchedulerReporter.class),
                        mock(TheoryCacheHitReporter.class));
        grpc = mock(EngineGrpcClient.class);
        when(grpc.batchEnqueueAsync(anyString(), anyInt(), any()))
                .thenAnswer(
                        call -> {
                            sent.add(call.getArgument(2));
                            return reply;
                        });
        dispatcher =
                org.mockito.Mockito.spy(
                        new DeliveryCoordinator(
                                new org.flexlb.balance.scheduler.RequestRepository(),
                                grpc,
                                service,
                                reporter,
                                new org.flexlb.balance.scheduler.SchedulerConfiguration()
                                        .decodeAdmissionCapacity(service),
                                org.mockito.Mockito.mock(DeliveryExecution.class)));
        var execution =
                org.mockito.Mockito.spy(
                        org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry)
                                .deliveries());
        org.flexlb.balance.scheduler.SchedulerTestServices.setDependency(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry),
                "deliveries",
                execution);
        org.mockito.Mockito.doReturn(201L).when(execution).nextBatchId();
        org.flexlb.balance.scheduler.SchedulerTestServices.setDependency(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry),
                "dispatcher",
                dispatcher);
        prefill = mock(PrefillEndpoint.class);
        when(prefill.acceptsNewRequests()).thenReturn(true);
        when(prefill.getIp()).thenReturn("127.0.0.1");
        when(prefill.getHttpPort()).thenReturn(8080);
        when(prefill.getGrpcPort()).thenReturn(8090);
        when(prefill.batchCapacityAvailable(anyInt()))
                .thenAnswer(call -> ledger.prefill.batchCapacityAvailable(call.getArgument(0)));
        org.mockito.Mockito.doAnswer(
                        call -> {
                            RouteResult item = call.getArgument(0);
                            assertFalse(
                                    Thread.holdsLock(
                                            registry.repository.findActive(item.requestId())));
                            return org.flexlb.balance.endpoint.EndpointTestSupport.releaseRequest(
                                    ledger.prefill, item);
                        })
                .when(prefill)
                .releaseRequest(any());

        when(prefill.tryStartQueuedDelivery(any(Delivery.Selection.class)))
                .thenAnswer(call -> ledger.commit(call.getArgument(0)));
        decode =
                EndpointTestSupport.decode(
                        WorkerStatus.createDiscovered(
                                RoleType.DECODE, null, "127.0.0.1", 8080, 8081, null),
                        SchedulerTestSupport.repository(registry));
        worker =
                org.mockito.Mockito.spy(
                        org.flexlb.balance.endpoint.EndpointTestSupport.attach(
                                org.flexlb.balance.endpoint.EndpointTestSupport.worker(
                                        "queued-delivery",
                                        prefill,
                                        QueueExecutionSettings.capture(
                                                config,
                                                org.flexlb.balance.planner.BuiltinGroupPlanner.from(
                                                        config)),
                                        dispatcher,
                                        ledger.prefill,
                                        SchedulerTestSupport.queueDependencies(
                                                org.flexlb.balance.scheduler.SchedulerTestSupport
                                                        .runtime(registry)))));
        org.flexlb.balance.endpoint.EndpointTestSupport.attachWork(prefill, worker);
        org.flexlb.balance.endpoint.EndpointTestSupport.stubResources(prefill, ledger.prefill);
        if (config.getDispatcher().getType() == org.flexlb.config.DispatcherConfig.Type.NON_BATCH) {
            return;
        }
        CountDownLatch entered = new CountDownLatch(1);
        org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry)
                .deliveries()
                .tryReserveDelivery()
                .value()
                .submit(
                        () -> {
                            entered.countDown();
                            await(releaseExecutor);
                        },
                        failure -> {
                            throw new AssertionError(failure);
                        });
        await(entered);
    }

    @AfterEach
    void close() throws Exception {
        releaseExecutor.countDown();
        awaitDispatchTasks();
        reply.complete(
                EngineRpcService.EnqueueBatchResponsePB.newBuilder().setBatchId(201L).build());
        org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry)
                .deliveries()
                .awaitDeliveries();
        if (RequestProtocolTestSupport.closeRegistrationAndAwaitScheduling(registry)) {
            registry.closeOutstandingAndTerminalize();
            SchedulerTestSupport.timer(SchedulerTestSupport.runtime(registry)).close();
            SchedulerTestSupport.closeRequestExecutors(SchedulerTestSupport.runtime(registry));
        }
    }

    @Test
    void blockedHeadPreventsSendingALaterRequestOnAnAvailableDecode() {
        configureDelivery(true);
        var head = item(1L);
        var other = anotherDecode();
        var later = item(2L, null, other, 50);
        var blocker = blockDecode();
        try {
            cycle();
            cycle();
            assertEquals(List.of(head, later), ledger.prefill.captureQueue(10).items());
            assertEquals(0, other.resourceSnapshot().activeDispatchPermits());
            assertEquals(0, ledger.prefill.stats().batchCount());
        } finally {
            decode.release(blocker, DecodeResources.ReleaseReason.REMOTE_CLEANUP);
        }
    }

    @ParameterizedTest
    @CsvSource({
        "CANCEL, false",
        "CANCEL, true",
        "TIMER, false",
        "TIMER, true",
        "LATE_DEADLINE, false",
        "LATE_DEADLINE, true",
        "LATE_INACTIVITY, false",
        "LATE_INACTIVITY, true"
    })
    void invalidQueuedMembersNeverCrossHandoffAndReleaseExactResources(String expiry, boolean all)
            throws Exception {
        RouteResult first = item(1L);
        RouteResult second = item(2L);
        submit(List.of(first, second));
        assertEquals(
                Delivery.SendMode.NONE,
                SchedulerTestSupport.repository(registry)
                        .getRequestState("1", 0L)
                        .deliveryClaimKind());
        assertEquals(
                Delivery.SendMode.NONE,
                SchedulerTestSupport.repository(registry)
                        .getRequestState("2", 0L)
                        .deliveryClaimKind());
        assertOccupancy(1, 2);
        for (RouteResult item : all ? List.of(first, second) : List.of(first)) {
            RequestContext requestContext = registry.repository.findActive(item.requestId());
            switch (expiry) {
                case "CANCEL" ->
                        registry.cancel(item.requestId(), 0L, CancelReason.CLIENT_CANCELLED);
                case "TIMER" ->
                        RequestProtocolTestSupport.expireInactiveRequest(
                                registry,
                                requestContext,
                                requestContext.snapshot().createdAtMs() + TIMEOUT_MS);
                case "LATE_DEADLINE" -> expireAtDeliveryValidation(requestContext);
                case "LATE_INACTIVITY" ->
                        ReflectionTestUtils.setField(
                                requestContext,
                                "lastWorkerAtMs",
                                System.currentTimeMillis() - TIMEOUT_MS - 1L);
                default -> throw new AssertionError(expiry);
            }
        }
        releaseExecutor.countDown();
        awaitDispatchTasks();
        assertFalse(first.ctx().data().getFuture().get(5, TimeUnit.SECONDS).isSuccess());
        assertEquals(
                Delivery.SendMode.NONE,
                SchedulerTestSupport.repository(registry)
                        .getRequestState("1", 0L)
                        .deliveryClaimKind());
        int survivors = all ? 0 : 1;
        if (all) {
            assertFalse(second.ctx().data().getFuture().get(5, TimeUnit.SECONDS).isSuccess());
        }
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry));
        assertOccupancy(survivors, survivors);
        assertEquals(survivors, decode.routingView().engineCapacityUsed());
        assertEquals(survivors, decode.routingView().inflightHardKv());
        assertEquals(
                2L * survivors, EndpointTestSupport.expectedReservedKv(decode.resourceSnapshot()));
        assertEquals(survivors, sent.size());
        if (all) {
            verifyNoInteractions(grpc);
        } else {
            assertEquals(
                    List.of(2L),
                    sent.getFirst().getDpSlotsList().stream()
                            .flatMap(dp -> dp.getRequestsList().stream())
                            .map(input -> input.getInput().getRequestId())
                            .toList());
            assertEquals(
                    Delivery.SendMode.BATCH_ENQUEUE,
                    SchedulerTestSupport.repository(registry)
                            .getRequestState("2", 0L)
                            .deliveryClaimKind());
            reply.complete(ack(2L));
            assertTrue(second.ctx().data().getFuture().get(5, TimeUnit.SECONDS).isSuccess());
            ledger.finish(201L, second)
                    .forEach(
                            requestStatus ->
                                    RequestProtocolTestSupport.applyPrefillStatus(
                                            registry, prefill, RoleType.PREFILL, requestStatus));
            DeliverySettlementTestSupport.decodeStatus(decode, 2L, true);
        }
        // Repeated terminal events cannot subtract the other member or leak a batch permit.
        registry.cancel("1", 0L, CancelReason.CLIENT_CANCELLED);
        assertOccupancy(0, 0);
        assertEquals(0, decode.routingView().engineCapacityUsed());
        assertEquals(0, decode.routingView().inflightHardKv());
        assertEquals(0, EndpointTestSupport.expectedReservedKv(decode.resourceSnapshot()));
        var admission =
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry)
                        .deliveries()
                        .tryReserveDelivery();
        assertTrue(admission.accepted(), "admission permit must be available after RPC completion");
        admission.value().close();
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void lateHandoffStartsOneBoundedObservationWindowWithoutFabricatingWorkerActivity(boolean ack)
            throws Exception {
        RouteResult item = item(1L);
        RequestContext requestContext = registry.repository.findActive("1");
        long lastStatus = System.currentTimeMillis() - TIMEOUT_MS + 10_000L;
        ReflectionTestUtils.setField(requestContext, "lastWorkerAtMs", lastStatus);
        ReflectionTestUtils.invokeMethod(
                requestContext, "commitSnapshot", System.currentTimeMillis());
        submit(List.of(item));
        releaseExecutor.countDown();
        awaitDispatchTasks();
        assertEquals(1, sent.size());
        long handoff = requestContext.snapshot().delivery().startedAtMs();
        assertEquals(lastStatus, ReflectionTestUtils.getField(requestContext, "lastWorkerAtMs"));
        if (ack) {
            reply.complete(ack(1L));
            assertTrue(item.ctx().data().getFuture().get(5, TimeUnit.SECONDS).isSuccess());
        }
        Token originalTimer = requestContext.snapshot().deadlines().inactivity();
        assertNotNull(originalTimer);
        RequestProtocolTestSupport.expireInactivity(
                registry, requestContext, originalTimer, lastStatus + TIMEOUT_MS);
        assertEquals(
                handoff + TIMEOUT_MS, requestContext.snapshot().deadlines().inactivity().atMs());
        assertOccupancy(1, 1);
        RequestProtocolTestSupport.expireInactiveRequest(
                registry, requestContext, handoff + TIMEOUT_MS - 1);
        assertOccupancy(1, 1);
        var cleanupProof =
                new java.util.concurrent.CompletableFuture<EngineGrpcClient.CancelReply>();
        org.mockito.Mockito.when(
                        SchedulerTestSupport.engineClient(registry)
                                .cancelAsync(
                                        any(),
                                        anyInt(),
                                        any(),
                                        org.mockito.ArgumentMatchers.anyLong()))
                .thenReturn(cleanupProof);
        RequestProtocolTestSupport.expireInactiveRequest(
                registry, requestContext, handoff + TIMEOUT_MS);
        assertOccupancy(1, 1);
        cleanupProof.complete(EngineGrpcClient.CancelReply.REQUEST_CLEANED);
        if (!ack) {
            reply.complete(ack(1L));
        }
        requestContext
                .snapshot()
                .delivery()
                .identity()
                .completion()
                .toCompletableFuture()
                .get(2, TimeUnit.SECONDS);
        RequestProtocolTestSupport.awaitCondition(
                () ->
                        SchedulerTestSupport.repository(registry).findTerminal(item.requestId())
                                != null);
        assertOccupancy(0, 0);
        assertEquals(0, decode.routingView().engineCapacityUsed());
        assertEquals(ack, item.ctx().data().getFuture().get(5, TimeUnit.SECONDS).isSuccess());
    }

    @Test
    void blockedRpcDoesNotHoldContextOrQueueOwnershipLock() throws Exception {
        RouteResult item = item(1L);
        RequestContext context = item.ctx();
        CountDownLatch rpcEntered = new CountDownLatch(1);
        CountDownLatch releaseRpc = new CountDownLatch(1);
        var cleanupProof = new CompletableFuture<EngineGrpcClient.CancelReply>();
        when(SchedulerTestSupport.engineClient(registry)
                        .cancelAsync(any(), anyInt(), any(), anyLong()))
                .thenReturn(cleanupProof);
        when(grpc.batchEnqueueAsync(anyString(), anyInt(), any()))
                .thenAnswer(
                        call -> {
                            assertFalse(Thread.holdsLock(context));
                            org.junit.jupiter.api.Assertions.assertFalse(
                                    org.flexlb.balance.endpoint.EndpointTestSupport.stateLock(
                                                    ledger.prefill)
                                            .isHeldByCurrentThread());
                            rpcEntered.countDown();
                            await(releaseRpc);
                            return reply;
                        });
        try (var contender = java.util.concurrent.Executors.newSingleThreadExecutor()) {
            submit(List.of(item));
            releaseExecutor.countDown();
            try {
                await(rpcEntered);
                contender
                        .submit(
                                () -> {
                                    synchronized (context) {
                                        assertEquals(
                                                Delivery.SendMode.BATCH_ENQUEUE,
                                                context.snapshot().delivery().identity().kind());
                                    }
                                    runWithStateLock(
                                            org.flexlb.balance.endpoint.EndpointTestSupport
                                                    .stateLock(ledger.prefill),
                                            () -> {
                                                assertEquals(0, ledger.prefill.queueDepth());
                                            });
                                    registry.cancel(
                                            item.requestId(), 0L, CancelReason.CLIENT_CANCELLED);
                                })
                        .get(2, TimeUnit.SECONDS);
                cleanupProof.complete(EngineGrpcClient.CancelReply.REQUEST_CLEANED);
                SchedulerTestSupport.awaitRequestWork(
                        org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry));
                assertOccupancy(1, 1); // sender has not yet returned, even with remote proof
            } finally {
                releaseRpc.countDown();
            }
        }
        awaitDispatchTasks();
        reply.complete(ack(1L));
        RequestProtocolTestSupport.awaitCondition(
                () -> SchedulerTestSupport.repository(registry).findTerminal("1") != null);
        assertOccupancy(0, 0);
    }

    @Test
    void submissionFailureBeforeExecutorAcceptanceSettlesCommittedRowsAndReturnsPermits()
            throws Exception {
        var route = item(9801L);
        var attempt = new java.util.concurrent.atomic.AtomicReference<Delivery>();
        org.mockito.Mockito.doAnswer(
                        call -> {
                            attempt.set(call.getArgument(0));
                            throw new IllegalStateException("submission failed before acceptance");
                        })
                .when(org.flexlb.balance.scheduler.DeliveryTestSupport.transport(dispatcher))
                .dispatch(any());
        try {
            org.junit.jupiter.api.Assertions.assertDoesNotThrow(() -> submit(List.of(route)));
            releaseExecutor.countDown();
            awaitDispatchTasks();
            SchedulerTestSupport.awaitRequestWork(SchedulerTestSupport.runtime(registry));
            assertOccupancy(0, 0);
            assertEquals(0, decode.resourceSnapshot().activeDispatchPermits());
            org.mockito.Mockito.verify(grpc, org.mockito.Mockito.never())
                    .batchEnqueueAsync(any(), anyInt(), any());
        } finally {
            // Keep a failed regression from blocking the fixture's infrastructure drain.
            if (attempt.get() != null) {
                attempt.get().releaseMembers();
                attempt.get().executionPermit.close();
            }
        }
    }

    @Test
    void committedMaterializationFailureBeforeHandoffClosesRealAdmission() throws Exception {
        RouteResult item = item(1L);
        var failure = new IllegalStateException("materialization failed before dispatcher entry");
        org.mockito.Mockito.doAnswer(
                        call -> {
                            var dequeued = ledger.commit(call.getArgument(0));
                            var capture =
                                    mock(
                                            org.flexlb.balance.projection.RequestTimeEstimation
                                                    .WorkSnapshot.class);
                            when(capture.calculate())
                                    .thenAnswer(
                                            ignored -> {
                                                org.junit.jupiter.api.Assertions.assertFalse(
                                                        org.flexlb.balance.endpoint
                                                                .EndpointTestSupport.stateLock(
                                                                        ledger.prefill)
                                                                .isHeldByCurrentThread());
                                                throw failure;
                                            });
                            return CapacityBoundary.Attempt.accepted(capture);
                        })
                .when(prefill)
                .tryStartQueuedDelivery(any());
        org.junit.jupiter.api.Assertions.assertDoesNotThrow(() -> submit(List.of(item)));
        verify(registry)
                .recordPreparationFailure(
                        argThat(
                                e ->
                                        e instanceof ContextResult.PreparationFailure failed
                                                && failed.route() == item
                                                && failed.detail().equals(failure.getMessage())));
        assertFalse(item.ctx().data().getFuture().get(5, TimeUnit.SECONDS).isSuccess());
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry));
        assertOccupancy(0, 0);
        assertEquals(0, decode.routingView().engineCapacityUsed());
        assertEquals(0, decode.routingView().inflightHardKv());
        assertEquals(0, EndpointTestSupport.expectedReservedKv(decode.resourceSnapshot()));
        verifyNoInteractions(grpc);
        var admission =
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry)
                        .deliveries()
                        .tryReserveDelivery();
        assertTrue(admission.accepted(), "admission permit must be available after RPC completion");
        admission.value().close();
    }

    @Test
    void rejectedExecutorSubmissionClosesCommittedAdmissionExactlyOnce() throws Exception {
        RouteResult item = item(1L);
        var rejection = new java.util.concurrent.RejectedExecutionException("executor rejected");
        var original =
                (ThreadPoolExecutor)
                        ReflectionTestUtils.getField(
                                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry)
                                        .deliveries(),
                                "dispatchExecutor");
        ThreadPoolExecutor rejected = mock(ThreadPoolExecutor.class);
        doThrow(rejection).when(rejected).execute(any());
        ReflectionTestUtils.setField(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry).deliveries(),
                "dispatchExecutor",
                rejected);
        try {
            submit(List.of(item));
        } finally {
            ReflectionTestUtils.setField(
                    org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry)
                            .deliveries(),
                    "dispatchExecutor",
                    original);
        }
        assertFalse(item.ctx().data().getFuture().get(5, TimeUnit.SECONDS).isSuccess());
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry));
        verify(rejected, times(1)).execute(any());
        assertOccupancy(0, 0);
        assertEquals(0, decode.routingView().engineCapacityUsed());
        assertEquals(0, decode.routingView().inflightHardKv());
        assertEquals(0, EndpointTestSupport.expectedReservedKv(decode.resourceSnapshot()));
        verifyNoInteractions(grpc);
        var admission =
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry)
                        .deliveries()
                        .tryReserveDelivery();
        assertTrue(admission.accepted(), "admission permit must be available after RPC completion");
        admission.value().close();
    }

    @Test
    void payloadFailureReleasesTemporaryCapacityWithoutCommittingDelivery() throws Exception {
        RouteResult item =
                item(1L, com.google.protobuf.ByteString.copyFrom(new byte[] {(byte) 0xff}));
        submit(List.of(item));
        releaseExecutor.countDown();
        awaitDispatchTasks();
        assertFalse(item.ctx().data().getFuture().get(5, TimeUnit.SECONDS).isSuccess());
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry));
        assertNull(item.ctx().snapshot().delivery());
        verify(
                        org.flexlb.balance.scheduler.DeliveryTestSupport.transport(
                                registry.dispatcher),
                        org.mockito.Mockito.never())
                .commit(
                        any(),
                        eq(Delivery.SendMode.BATCH_ENQUEUE),
                        anyLong(),
                        org.mockito.ArgumentMatchers.isNull());
        assertOccupancy(0, 0);
        assertEquals(0, decode.routingView().engineCapacityUsed());
        assertEquals(0, decode.routingView().inflightHardKv());
        assertEquals(0, EndpointTestSupport.expectedReservedKv(decode.resourceSnapshot()));
        verifyNoInteractions(grpc);
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void cancellationOrTimeoutDuringPayloadPreparationNeverStartsDelivery(boolean timeout)
            throws Exception {
        var route = item(99005L);
        var context = route.ctx();
        var encoding = new CountDownLatch(1);
        var resumeEncoding = new CountDownLatch(1);
        var dispatchConfig = org.mockito.Mockito.spy(config.getDispatcher());
        org.mockito.Mockito.doReturn(dispatchConfig).when(config).getDispatcher();
        doAnswer(
                        call -> {
                            encoding.countDown();
                            assertTrue(resumeEncoding.await(2, TimeUnit.SECONDS));
                            return call.callRealMethod();
                        })
                .when(dispatchConfig)
                .getFetchAttachTimeoutMs();
        try {
            submit(List.of(route));
            releaseExecutor.countDown();
            assertTrue(encoding.await(2, TimeUnit.SECONDS));
            assertNull(context.snapshot().delivery());
            assertEquals(1, decode.resourceSnapshot().activeDispatchPermits());
            assertEquals(1, decode.resourceSnapshot().queuedCount());
            assertEquals(0, decode.routingView().engineLoad());
            assertEquals(
                    1,
                    decode.routingView().inflightHardKv(),
                    "selected Decode KV remains charged while payload preparation holds its permit");
            if (timeout) {
                var deadline = context.snapshot().deadlines().inactivity();
                assertNotNull(deadline);
                registry.onDeadline(deadline, deadline.atMs());
            } else {
                registry.cancel(context, 0L, CancelReason.CLIENT_CANCELLED);
            }
            assertFalse(context.data().getFuture().get(2, TimeUnit.SECONDS).isSuccess());
            assertNull(context.snapshot().delivery());
        } finally {
            resumeEncoding.countDown();
        }
        awaitDispatchTasks();
        SchedulerTestSupport.awaitRequestWork(SchedulerTestSupport.runtime(registry));
        assertNull(context.snapshot().delivery());
        assertTrue(sent.isEmpty());
        assertOccupancy(0, 0);
        assertEquals(0, decode.resourceSnapshot().activeDispatchPermits());
        assertEquals(0, decode.resourceSnapshot().reservedCount());
        assertEquals(0, decode.routingView().engineCapacityUsed());
        assertEquals(0, decode.routingView().inflightHardKv());
        verify(grpc, org.mockito.Mockito.never()).batchEnqueueAsync(any(), anyInt(), any());
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.MethodSource("invalidMemberSubsets")
    void everyInvalidMemberSubsetPreservesPayloadAndRealResourceLedgers(
            String cause, int invalidMask) throws Exception {
        var items = List.of(item(11L), item(12L), item(13L));
        submit(items);
        assertOccupancy(1, 3);
        var survivors = new java.util.ArrayList<RouteResult>();
        for (int index = 0; index < items.size(); index++) {
            var item = items.get(index);
            if ((invalidMask & (1 << index)) == 0) {
                survivors.add(item);
                continue;
            }
            RequestContext context = item.ctx();
            switch (cause) {
                case "CLIENT_CANCELLED" ->
                        registry.cancel(item.requestId(), 0L, CancelReason.CLIENT_CANCELLED);
                case "SHUTDOWN" -> registry.cancel(item.requestId(), 0L, CancelReason.SHUTDOWN);
                case "DEADLINE" -> expireAtDeliveryValidation(context);
                case "INACTIVITY" ->
                        ReflectionTestUtils.setField(
                                context,
                                "lastWorkerAtMs",
                                System.currentTimeMillis() - TIMEOUT_MS - 1L);
                default -> throw new AssertionError(cause);
            }
        }
        releaseExecutor.countDown();
        awaitDispatchTasks();
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry));
        for (int index = 0; index < items.size(); index++) {
            if ((invalidMask & (1 << index)) != 0) {
                assertFalse(
                        items.get(index)
                                .ctx()
                                .data()
                                .getFuture()
                                .get(5, TimeUnit.SECONDS)
                                .isSuccess());
            }
        }
        assertOccupancy(survivors.isEmpty() ? 0 : 1, survivors.size());
        assertEquals(survivors.size(), decode.routingView().engineCapacityUsed());
        assertEquals(survivors.size(), decode.routingView().inflightHardKv());
        assertEquals(
                2L * survivors.size(),
                EndpointTestSupport.expectedReservedKv(decode.resourceSnapshot()));
        if (survivors.isEmpty()) {
            verifyNoInteractions(grpc);
        } else {
            assertEquals(1, sent.size());
            assertEquals(
                    survivors.stream().map(route -> Long.parseLong(route.requestId())).toList(),
                    sent.getFirst().getDpSlotsList().stream()
                            .flatMap(requestContext -> requestContext.getRequestsList().stream())
                            .map(request -> request.getInput().getRequestId())
                            .toList());
            var ack = EngineRpcService.EnqueueBatchResponsePB.newBuilder().setBatchId(201L);
            survivors.forEach(
                    item ->
                            ack.addSuccesses(
                                    EngineRpcService.EnqueueBatchSuccessPB.newBuilder()
                                            .setRequestId(Long.parseLong(item.requestId()))));
            reply.complete(ack.build());
            for (RouteResult item : survivors) {
                assertTrue(item.ctx().data().getFuture().get(5, TimeUnit.SECONDS).isSuccess());
                ledger.finish(201L, item)
                        .forEach(
                                requestStatus ->
                                        RequestProtocolTestSupport.applyPrefillStatus(
                                                registry,
                                                prefill,
                                                RoleType.PREFILL,
                                                requestStatus));
                DeliverySettlementTestSupport.decodeStatus(
                        decode, Long.parseLong(item.requestId()), true);
            }
        }
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry));
        for (int index = 0; index < items.size(); index++) {
            if ((invalidMask & (1 << index)) != 0) {
                registry.cancel(items.get(index).requestId(), 0L, CancelReason.CLIENT_CANCELLED);
            }
        }
        assertOccupancy(0, 0);
        assertEquals(0, decode.routingView().engineCapacityUsed());
        assertEquals(0, decode.routingView().inflightHardKv());
        assertEquals(0, EndpointTestSupport.expectedReservedKv(decode.resourceSnapshot()));
        var admission =
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry)
                        .deliveries()
                        .tryReserveDelivery();
        assertTrue(admission.accepted(), "admission permit must be available after RPC completion");
        admission.value().close();
    }

    static java.util.stream.Stream<org.junit.jupiter.params.provider.Arguments>
            invalidMemberSubsets() {
        return java.util.stream.Stream.of("CLIENT_CANCELLED", "SHUTDOWN", "DEADLINE", "INACTIVITY")
                .flatMap(
                        cause ->
                                java.util.stream.IntStream.range(0, 8)
                                        .mapToObj(
                                                mask ->
                                                        org.junit.jupiter.params.provider.Arguments
                                                                .of(cause, mask)));
    }

    @Test
    void deliveryObserverFailureRetainsRuntimeCauseAfterRealSettlement() throws Exception {
        var first = item(981_011L);
        var second = item(981_012L);
        var observerFailure =
                new IllegalStateException("observer failed after real Scheduler evidence");
        var callbackExited = new CountDownLatch(1);
        org.mockito.Mockito.doAnswer(
                        call -> {
                            call.callRealMethod();
                            callbackExited.countDown();
                            throw observerFailure;
                        })
                .when(registry)
                .onDeliveryResult(
                        argThat(
                                e ->
                                        e instanceof ContextResult.TransportResult result
                                                && result.delivery().route() == first));
        submit(List.of(first, second));
        releaseExecutor.countDown();
        awaitDispatchTasks();
        var ack = EngineRpcService.EnqueueBatchResponsePB.newBuilder().setBatchId(201L);
        ack.addSuccesses(
                EngineRpcService.EnqueueBatchSuccessPB.newBuilder()
                        .setRequestId(Long.parseLong(first.requestId())));
        ack.addSuccesses(
                EngineRpcService.EnqueueBatchSuccessPB.newBuilder()
                        .setRequestId(Long.parseLong(second.requestId())));
        reply.complete(ack.build());
        assertTrue(callbackExited.await(2, TimeUnit.SECONDS));
        assertTrue(first.ctx().data().getFuture().get(2, TimeUnit.SECONDS).isSuccess());
        assertTrue(second.ctx().data().getFuture().get(2, TimeUnit.SECONDS).isSuccess());
        org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry)
                .deliveries()
                .awaitDeliveries();
        for (var route : List.of(first, second)) {
            assertEquals(
                    ContextResult.SendState.DELIVERED, route.ctx().snapshot().delivery().send());
            ledger.finish(201L, route)
                    .forEach(
                            status ->
                                    RequestProtocolTestSupport.applyPrefillStatus(
                                            registry, prefill, RoleType.PREFILL, status));
            DeliverySettlementTestSupport.decodeStatus(
                    decode, Long.parseLong(route.requestId()), true);
        }
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry));
        assertOccupancy(0, 0);
        assertEquals(0, decode.routingView().engineCapacityUsed());
        assertEquals(0, decode.resourceSnapshot().activeDispatchPermits());
        assertEquals(0, registry.repository.liveRequestCount());
        assertSame(
                observerFailure,
                org.flexlb.balance.scheduler.SchedulerTestServices.failure(
                        org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry)),
                "background evidence failures must survive into shutdown reporting");
    }

    @Test
    void temporaryCloseFailuresRemainReportedWithoutChangingAcceptedRpcResult() throws Exception {
        decode = org.mockito.Mockito.spy(decode);
        var first = item(981_021L);
        var second = item(981_022L);
        var permitFailure =
                new IllegalStateException("permit close observer failed after actual release");
        org.mockito.Mockito.doAnswer(
                        call -> {
                            var result = call.callRealMethod();
                            DecodeEndpoint.EngineDispatchPermit permit = call.getArgument(0);
                            if (permit.belongsTo(decode, first.decodeReservation()))
                                throw permitFailure;
                            return result;
                        })
                .when(decode)
                .dispatch(
                        any(),
                        org.mockito.ArgumentMatchers.eq(
                                org.flexlb.balance.endpoint.DecodeResources.DispatchOutcome
                                        .ABANDONED));
        submit(List.of(first, second));
        releaseExecutor.countDown();
        awaitDispatchTasks();
        var ack = EngineRpcService.EnqueueBatchResponsePB.newBuilder().setBatchId(201L);
        ack.addSuccesses(
                EngineRpcService.EnqueueBatchSuccessPB.newBuilder()
                        .setRequestId(Long.parseLong(first.requestId())));
        ack.addSuccesses(
                EngineRpcService.EnqueueBatchSuccessPB.newBuilder()
                        .setRequestId(Long.parseLong(second.requestId())));
        reply.complete(ack.build());
        assertTrue(first.ctx().data().getFuture().get(2, TimeUnit.SECONDS).isSuccess());
        assertTrue(second.ctx().data().getFuture().get(2, TimeUnit.SECONDS).isSuccess());
        org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry)
                .deliveries()
                .awaitDeliveries();
        for (var route : List.of(first, second)) {
            assertEquals(
                    ContextResult.SendState.DELIVERED, route.ctx().snapshot().delivery().send());
            ledger.finish(201L, route)
                    .forEach(
                            status ->
                                    RequestProtocolTestSupport.applyPrefillStatus(
                                            registry, prefill, RoleType.PREFILL, status));
            DeliverySettlementTestSupport.decodeStatus(
                    decode, Long.parseLong(route.requestId()), true);
        }
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry));
        assertOccupancy(0, 0);
        assertEquals(0, decode.routingView().engineCapacityUsed());
        assertEquals(0, decode.resourceSnapshot().activeDispatchPermits());
        assertEquals(0, registry.repository.liveRequestCount());
        assertSame(
                permitFailure,
                org.flexlb.balance.scheduler.SchedulerTestServices.failure(
                        org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry)));
    }

    @Test
    void customDispatcherCanRequireTheWholeSelectedGroupInsteadOfAnImplicitPrefix() {
        configureDelivery(true);
        var available = anotherDecode();
        var first = item(81001L, null, available, 50);
        var second = item(81002L);
        var blocker = blockDecode();
        try {
            var result =
                    dispatcher.submit(
                            List.of(first, second),
                            "exact_group",
                            prefill.predictor.evaluator(),
                            OptionalLong.of(1L),
                            Delivery.PrefixPolicy.ALL_SELECTED);
            assertNotNull(result);
            assertEquals(CapacityBoundary.Resource.DECODE_ENGINE, result.resource());
            assertEquals(List.of(first, second), ledger.prefill.captureQueue(10).items());
            assertEquals(0, available.resourceSnapshot().activeDispatchPermits());
            assertEquals(0, ledger.prefill.stats().batchCount());
            assertTrue(sent.isEmpty());
        } finally {
            decode.release(blocker, DecodeResources.ReleaseReason.REMOTE_CLEANUP);
        }
    }

    @Test
    void selectingALaterRequestCannotSkipTheAvailableHead() {
        configureDelivery(false);
        var head = item(82001L);
        var later = item(82002L, null, anotherDecode(), 40);
        worker.processSelection(
                List.of(later),
                "non_prefix_selection",
                prefill.predictor.evaluator(),
                OptionalLong.of(1L));
        verify(prefill).tryStartQueuedDelivery(any());
        assertEquals(List.of(head, later), ledger.prefill.captureQueue(10).items());
        assertEquals(0, later.decodeEndpoint().resourceSnapshot().activeDispatchPermits());
        assertEquals(0, ledger.prefill.stats().batchCount());
        assertTrue(sent.isEmpty());
    }

    @Test
    void fifoReofferAndArrivalDuringCommitPreserveSelectedPrefix() throws Exception {
        configureOrderedQueue(false, DecisionPolicyConfig.Type.FIXED_WINDOW, 2);
        var first = item(100L);
        var second = item(10L, null, anotherDecode(2), 50);
        var third = item(50L, null, anotherDecode(3), 50);
        org.flexlb.balance.endpoint.EndpointTestSupport.removeQueued(ledger.prefill, first);
        ledger.enqueue(first);
        assertEquals(List.of(second, third, first), ledger.prefill.captureQueue(10).items());
        var arrival = new CompletableFuture<RouteResult>();
        doAnswer(
                        call -> {
                            arrival.complete(
                                    CompletableFuture.supplyAsync(
                                                    () -> item(1L, null, anotherDecode(4), 90))
                                            .get(5, TimeUnit.SECONDS));
                            return ledger.commit(call.getArgument(0));
                        })
                .when(prefill)
                .tryStartQueuedDelivery(any());
        cycle();
        assertEquals(
                List.of(first, arrival.get(5, TimeUnit.SECONDS)),
                ledger.prefill.captureQueue(10).items());
        releaseExecutor.countDown();
        awaitDispatchTasks();
        assertEquals(1, sent.size());
        assertEquals(
                List.of(10L, 50L),
                sent.getFirst().getDpSlotsList().stream()
                        .flatMap(dp -> dp.getRequestsList().stream())
                        .map(request -> request.getInput().getRequestId())
                        .toList());
        assertEquals(0, first.decodeEndpoint().resourceSnapshot().activeDispatchPermits());
        assertEquals(0, arrival.get().decodeEndpoint().resourceSnapshot().activeDispatchPermits());
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void fixedWindowCanPublishRoutesWithoutSendingAnEnqueueBatch(boolean phaseConfigured)
            throws Exception {
        if (phaseConfigured) {
            var delivery = new org.flexlb.config.PhasePoliciesConfig.Delivery();
            delivery.setType(org.flexlb.config.PhasePoliciesConfig.DeliveryType.ROUTE);
            config.getPhases().getGeneration().setDelivery(delivery);
        } else {
            config.getDispatcher().setType(org.flexlb.config.DispatcherConfig.Type.NON_BATCH);
        }
        configureDelivery(false, DecisionPolicyConfig.Type.FIXED_WINDOW, 2);
        var first = item(83001L);
        var second = item(83002L, null, anotherDecode(), 50);

        cycle();

        for (var route : List.of(first, second)) {
            assertTrue(route.ctx().data().getFuture().get(2, TimeUnit.SECONDS).isSuccess());
            assertEquals(
                    Delivery.SendMode.ROUTE_DECISION,
                    route.ctx().snapshot().delivery().identity().kind());
        }
        assertTrue(sent.isEmpty());
        assertTrue(ledger.prefill.captureQueue(10).items().isEmpty());
    }

    private void expireAtDeliveryValidation(RequestContext context) {
        doAnswer(
                        call -> {
                            Delivery.Member member = call.getArgument(0);
                            assertNotNull(
                                    member.decode(),
                                    "expiry is observed after dispatch capacity was acquired");
                            synchronized (context) {
                                assertFalse(
                                        context.validateDelivery(
                                                member.route(),
                                                null,
                                                context.data().getRequestExpiresAtMs()));
                            }
                            return call.callRealMethod();
                        })
                .when(
                        org.flexlb.balance.scheduler.DeliveryTestSupport.transport(
                                registry.dispatcher))
                .commit(
                        org.mockito.ArgumentMatchers.argThat(
                                member -> member != null && member.owner() == context),
                        any(),
                        anyLong(),
                        org.mockito.ArgumentMatchers.isNull());
    }

    private void configureOrderedQueue(
            boolean priority, DecisionPolicyConfig.Type grouping, int maxRequests) {
        ledger = new DeliverySettlementTestSupport();
        org.flexlb.balance.endpoint.EndpointTestSupport.stubResources(prefill, ledger.prefill);
        var execution =
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry).deliveries();
        org.mockito.Mockito.doCallRealMethod().when(execution).nextBatchId();
        configureDelivery(priority, grouping, maxRequests);
    }

    private void configureDelivery(boolean priorityOrdering) {
        configureDelivery(priorityOrdering, DecisionPolicyConfig.Type.FIXED_WINDOW, 2);
    }

    private void configureDelivery(
            boolean priorityOrdering, DecisionPolicyConfig.Type grouping, int maxRequests) {
        DeliveryTestSupport.setField(
                dispatcher,
                "mode",
                config.deliveryType(org.flexlb.dao.route.RequestPhase.GENERATION));
        ReflectionTestUtils.setField(prefill, "predictor", new FormulaPredictor("1"));
        ReflectionTestUtils.setField(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry).deliveries(),
                "decodeDeliveryCapacity",
                new DecodeResources.AdmissionCapacity(1, 100L));
        worker =
                org.mockito.Mockito.spy(
                        org.flexlb.balance.endpoint.EndpointTestSupport.attach(
                                org.flexlb.balance.endpoint.EndpointTestSupport.worker(
                                        "prefill-delivery",
                                        prefill,
                                        new QueueExecutionSettings(
                                                config.deliveryType(
                                                        org.flexlb.dao.route.RequestPhase
                                                                .GENERATION),
                                                org.flexlb.balance.planner.BuiltinGroupPlanner
                                                        .valueOf(grouping.name()),
                                                maxRequests,
                                                0L,
                                                0L),
                                        dispatcher,
                                        ledger.prefill,
                                        SchedulerTestSupport.queueDependencies(
                                                org.flexlb.balance.scheduler.SchedulerTestSupport
                                                        .runtime(registry)))));
        org.flexlb.balance.endpoint.EndpointTestSupport.attachWork(prefill, worker);
    }

    private RouteResult item(long id) {
        return item(id, null);
    }

    private RouteResult item(long id, com.google.protobuf.ByteString payload) {
        return item(id, payload, decode, 50);
    }

    private RouteResult item(
            long id, com.google.protobuf.ByteString payload, DecodeEndpoint target, int priority) {
        var context = RequestProtocolTestSupport.context(config, id);
        context.data()
                .setSchedulingMetadata(
                        org.flexlb.dao.SchedulingMetadata.explicit(priority, Long.MAX_VALUE));
        context.data()
                .setGenerateInputPb(
                        payload != null
                                ? payload
                                : EngineRpcService.GenerateInputPB.newBuilder()
                                        .setRequestId(id)
                                        .setGenerateConfig(
                                                EngineRpcService.GenerateConfigPB.newBuilder())
                                        .build()
                                        .toByteString());
        var future = RequestProtocolTestSupport.submitForManualPlacement(registry, context);
        DecodeResources.RequestReservation reservation;
        {
            var selectedEndpoint = target;

            reservation = EndpointTestSupport.registerDecode(target, id, 1L, 2L, priority);
        }
        assertNotNull(reservation);
        DeliverySettlementTestSupport.queueDecode(target, reservation);
        context.data().setFuture(future);
        var item =
                RequestExecutionTestSupport.registerPrefillRoute(
                        context, prefill, target, reservation, 0L, System.currentTimeMillis());
        RequestProtocolTestSupport.bind(
                registry, new RequestProtocolTestSupport.Registered(item, future));
        ledger.enqueue(item);
        return item;
    }

    private DecodeEndpoint anotherDecode() {
        return anotherDecode(2);
    }

    private DecodeEndpoint anotherDecode(int address) {
        return EndpointTestSupport.decode(
                WorkerStatus.createDiscovered(
                        RoleType.DECODE, null, "127.0.0." + address, 8080, 8081, null),
                SchedulerTestSupport.repository(registry));
    }

    private DecodeResources.RequestReservation blockDecode() {
        var reservation = EndpointTestSupport.registerDecode(decode, 90_000L, 1L, 2L, 50);
        var acquired =
                decode.acquireDispatchPermit(reservation, SchedulingTestConfig.capacity(config));
        assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.ACQUIRED, acquired.status());
        assertEquals(
                DecodeResources.EngineDispatchPermitTransferStatus.TRANSFERRED,
                acquired.value().dispatch().status());
        return reservation;
    }

    private void cycle() {
        assertNotNull(RequestExecutionTestSupport.schedule(currentWork()));
    }

    private PrefillQueueScheduler currentWork() {
        var automatic = RequestExecutionTestSupport.schedulingState(worker);
        if (automatic != null) return automatic;
        if (manualScheduling == null || manualScheduling != worker) manualScheduling = worker;
        return manualScheduling;
    }

    private void awaitDispatchTasks() throws Exception {
        ThreadPoolExecutor executor =
                (ThreadPoolExecutor)
                        ReflectionTestUtils.getField(
                                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry)
                                        .deliveries(),
                                "dispatchExecutor");
        if (executor.isShutdown()) {
            assertTrue(
                    executor.awaitTermination(5, TimeUnit.SECONDS),
                    "a stopped dispatch executor must finish all accepted tasks");
        } else {
            executor.submit(() -> {}).get(5, TimeUnit.SECONDS);
        }
    }

    private void submit(List<RouteResult> items) {
        var evaluator = new FormulaPredictor("100 * batchSize");
        org.junit.jupiter.api.Assertions.assertNull(
                worker.processSelection(items, "queued-race", evaluator, OptionalLong.empty()));
    }

    private void assertOccupancy(int batches, int requests) {
        assertEquals(batches, ledger.prefill.stats().batchCount());
        assertEquals(requests, ledger.prefill.stats().locallyOwnedRequests());
    }

    private static EngineRpcService.EnqueueBatchResponsePB ack(long id) {
        return EngineRpcService.EnqueueBatchResponsePB.newBuilder()
                .setBatchId(201L)
                .addSuccesses(EngineRpcService.EnqueueBatchSuccessPB.newBuilder().setRequestId(id))
                .build();
    }
}
