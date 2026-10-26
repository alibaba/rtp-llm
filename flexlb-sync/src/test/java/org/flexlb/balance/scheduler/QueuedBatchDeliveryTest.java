package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.DeliverySettlementTestSupport;
import org.flexlb.balance.endpoint.EndpointTestSupport;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.prediction.FormulaPredictor;
import org.flexlb.balance.scheduler.ExpirationTimer.Token;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DecisionPolicyConfig;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.OptionalLong;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

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
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

/** Real executor, request context and endpoint ledgers; only the Engine transport is simulated. */
class QueuedBatchDeliveryTest {
    private static final long TIMEOUT_MS = TimeUnit.HOURS.toMillis(1);
    private final CountDownLatch releaseExecutor = new CountDownLatch(1);
    private final CompletableFuture<EngineRpcService.EnqueueBatchResponsePB> reply =
            new CompletableFuture<>();
    private DeliverySettlementTestSupport ledger = new DeliverySettlementTestSupport();
    private FlexlbConfig config;
    private AbstractRequestScheduler registry;
    private DeliveryDispatcher dispatcher;
    private EngineGrpcClient grpc;
    private PrefillEndpoint prefill;
    private DecodeEndpoint decode;
    private PrefillQueueRunner worker;
    private final java.util.concurrent.CopyOnWriteArrayList<EngineRpcService.EnqueueBatchRequestPB>
            sent = new java.util.concurrent.CopyOnWriteArrayList<>();

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
                        mock(RecentCacheKeyTraceReporter.class));
        grpc = mock(EngineGrpcClient.class);
        when(grpc.batchEnqueueAsync(anyString(), anyInt(), any()))
                .thenAnswer(
                        call -> {
                            sent.add(call.getArgument(2));
                            return reply;
                        });
        dispatcher = new DeliveryDispatcher(grpc, service, reporter);
        var execution = org.mockito.Mockito.spy(registry.runtime.deliveries());
        ReflectionTestUtils.setField(registry.runtime, "deliveries", execution);
        org.mockito.Mockito.doReturn(201L).when(execution).nextBatchId();
        ReflectionTestUtils.setField(registry.runtime, "dispatcher", dispatcher);
        prefill = mock(PrefillEndpoint.class);
        when(prefill.getIp()).thenReturn("127.0.0.1");
        when(prefill.getGrpcPort()).thenReturn(8090);
        when(prefill.reserveBatch(any(), anyLong(), anyInt()))
                .thenAnswer(
                        call ->
                                ledger.reserveBatch(
                                        call.getArgument(0),
                                        call.getArgument(1),
                                        call.getArgument(2)));
        org.mockito.Mockito.doAnswer(
                        call -> {
                            RequestRoute item = call.getArgument(1);
                            assertFalse(
                                    Thread.holdsLock(
                                            registry.requests.findActive(item.requestId())));
                            return org.flexlb.balance.endpoint.EndpointTestSupport.releaseRequest(
                                    ledger.prefill, item);
                        })
                .when(prefill)
                .release(isNull(), any());
        org.mockito.Mockito.doAnswer(
                        call -> {
                            var released = ledger.prefill.release(call.getArgument(0), null);
                            if (released.generationHandoff() != null) {
                                ((AutoCloseable) released.generationHandoff()).close();
                            }
                            return released.released();
                        })
                .when(prefill)
                .release(any(PrefillState.Reservation.class), isNull());
        when(prefill.commitLocked(any(PrefillState.CommitPlan.class)))
                .thenAnswer(call -> ledger.commit(call.getArgument(0)));
        decode =
                EndpointTestSupport.decode(
                        WorkerStatus.createDiscovered(
                                RoleType.DECODE, null, "127.0.0.1", 8080, 8081, null),
                        SchedulerTestSupport.repository(registry));
        worker =
                org.mockito.Mockito.spy(
                        new PrefillQueueRunner(
                                "queued-delivery",
                                prefill,
                                QueueExecutionSettings.capture(config),
                                dispatcher,
                                ledger.prefill));
        ReflectionTestUtils.setField(prefill, "queueRunner", worker);
        ReflectionTestUtils.setField(prefill, "ownership", ledger.prefill);
        if (config.getDispatcher().getType() == org.flexlb.config.DispatcherConfig.Type.NON_BATCH) {
            return;
        }
        CountDownLatch entered = new CountDownLatch(1);
        registry.runtime
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
        registry.runtime.deliveries().awaitDeliveries();
        if (RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(registry)) {
            registry.closeOutstandingAndTerminalize();
            SchedulerTestSupport.timer(SchedulerTestSupport.runtime(registry)).close();
            SchedulerTestSupport.closeRequestExecutors(SchedulerTestSupport.runtime(registry));
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
        RequestRoute first = item(1L);
        RequestRoute second = item(2L);
        submit(List.of(first, second));
        assertEquals(
                DeliveryClaimKind.NONE,
                SchedulerTestSupport.repository(registry)
                        .getRequestState(1L, 0L)
                        .deliveryClaimKind());
        assertEquals(
                DeliveryClaimKind.NONE,
                SchedulerTestSupport.repository(registry)
                        .getRequestState(2L, 0L)
                        .deliveryClaimKind());
        assertOccupancy(1, 2);
        for (RequestRoute item : all ? List.of(first, second) : List.of(first)) {
            RequestContext requestContext = registry.requests.findActive(item.requestId());
            switch (expiry) {
                case "CANCEL" ->
                        registry.cancel(item.requestId(), 0L, CancelReason.CLIENT_CANCELLED);
                case "TIMER" ->
                        RequestProtocolTestSupport.expireInactiveRequest(
                                registry,
                                requestContext,
                                requestContext.snapshot().createdAtMs() + TIMEOUT_MS);
                case "LATE_DEADLINE" -> expireSchedulingFact(requestContext);
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
        assertFalse(first.future().get(5, TimeUnit.SECONDS).isSuccess());
        assertEquals(
                DeliveryClaimKind.NONE,
                SchedulerTestSupport.repository(registry)
                        .getRequestState(1L, 0L)
                        .deliveryClaimKind());
        int survivors = all ? 0 : 1;
        if (all) {
            assertFalse(second.future().get(5, TimeUnit.SECONDS).isSuccess());
        }
        SchedulerTestSupport.awaitRequestWork(registry.runtime);
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
                    DeliveryClaimKind.BATCH_ENQUEUE,
                    SchedulerTestSupport.repository(registry)
                            .getRequestState(2L, 0L)
                            .deliveryClaimKind());
            reply.complete(ack(2L));
            assertTrue(second.future().get(5, TimeUnit.SECONDS).isSuccess());
            ledger.finish(201L, second)
                    .forEach(
                            requestStatus ->
                                    RequestProtocolTestSupport.applyPrefillStatus(
                                            registry, prefill, RoleType.PREFILL, requestStatus));
            DeliverySettlementTestSupport.decodeStatus(decode, 2L, true);
        }
        // Repeated terminal events cannot subtract the other member or leak a batch permit.
        registry.cancel(1L, 0L, CancelReason.CLIENT_CANCELLED);
        assertOccupancy(0, 0);
        assertEquals(0, decode.routingView().engineCapacityUsed());
        assertEquals(0, decode.routingView().inflightHardKv());
        assertEquals(0, EndpointTestSupport.expectedReservedKv(decode.resourceSnapshot()));
        var admission = registry.runtime.deliveries().tryReserveDelivery();
        assertTrue(admission.accepted(), "admission permit must be available after RPC completion");
        admission.value().close();
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void lateHandoffStartsOneBoundedObservationWindowWithoutFabricatingWorkerActivity(boolean ack)
            throws Exception {
        RequestRoute item = item(1L);
        RequestContext requestContext = registry.requests.findActive(1L);
        long lastStatus = System.currentTimeMillis() - TIMEOUT_MS + 10_000L;
        ReflectionTestUtils.setField(requestContext, "lastWorkerAtMs", lastStatus);
        ReflectionTestUtils.invokeMethod(requestContext, "refresh", System.currentTimeMillis());
        submit(List.of(item));
        releaseExecutor.countDown();
        awaitDispatchTasks();
        assertEquals(1, sent.size());
        long handoff = requestContext.snapshot().delivery().startedAtMs();
        assertEquals(lastStatus, ReflectionTestUtils.getField(requestContext, "lastWorkerAtMs"));
        if (ack) {
            reply.complete(ack(1L));
            assertTrue(item.future().get(5, TimeUnit.SECONDS).isSuccess());
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
        registry.runtime
                .signals()
                .delivery(requestContext.snapshot().delivery().identity())
                .toCompletableFuture()
                .get(2, TimeUnit.SECONDS);
        RequestProtocolTestSupport.awaitCondition(
                () ->
                        SchedulerTestSupport.repository(registry).findTerminal(item.requestId())
                                != null);
        assertOccupancy(0, 0);
        assertEquals(0, decode.routingView().engineCapacityUsed());
        assertEquals(ack, item.future().get(5, TimeUnit.SECONDS).isSuccess());
    }

    @Test
    void blockedRpcDoesNotHoldContextOrQueueOwnershipLock() throws Exception {
        RequestRoute item = item(1L);
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
                            assertFalse(ledger.prefill.queueGuard().isHeldByCurrentThread());
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
                                                DeliveryClaimKind.BATCH_ENQUEUE,
                                                context.snapshot().delivery().identity().kind());
                                    }
                                    ledger.prefill.queueGuard().lock();
                                    try {
                                        assertEquals(0, ledger.prefill.progress().queueDepth());
                                    } finally {
                                        ledger.prefill.queueGuard().unlock();
                                    }
                                    registry.cancel(
                                            item.requestId(), 0L, CancelReason.CLIENT_CANCELLED);
                                })
                        .get(2, TimeUnit.SECONDS);
                cleanupProof.complete(EngineGrpcClient.CancelReply.REQUEST_CLEANED);
                SchedulerTestSupport.awaitRequestWork(registry.runtime);
                assertOccupancy(1, 1); // sender has not yet returned, even with remote proof
            } finally {
                releaseRpc.countDown();
            }
        }
        awaitDispatchTasks();
        reply.complete(ack(1L));
        RequestProtocolTestSupport.awaitCondition(
                () -> SchedulerTestSupport.repository(registry).findTerminal(1L) != null);
        assertOccupancy(0, 0);
    }

    @Test
    void committedMaterializationFailureBeforeHandoffClosesRealAdmission() throws Exception {
        RequestRoute item = item(1L);
        var failure = new IllegalStateException("materialization failed before dispatcher entry");
        org.mockito.Mockito.doAnswer(
                        call -> {
                            PrefillState.CommittedHandoff actual =
                                    (PrefillState.CommittedHandoff) call.callRealMethod();
                            var committed = org.mockito.Mockito.spy(actual);
                            var capture = mock(PrefillState.WorkCapture.class);
                            when(capture.materialize())
                                    .thenAnswer(
                                            ignored -> {
                                                assertFalse(
                                                        ledger.prefill
                                                                .queueGuard()
                                                                .isHeldByCurrentThread());
                                                throw failure;
                                            });
                            org.mockito.Mockito.doReturn(capture).when(committed).precedingWork();
                            return committed;
                        })
                .when(worker)
                .commitSelection(any());
        org.junit.jupiter.api.Assertions.assertDoesNotThrow(() -> submit(List.of(item)));
        verify(registry)
                .onDeliveryEvidence(
                        org.mockito.ArgumentMatchers.isNull(),
                        argThat(
                                e ->
                                        e instanceof ContextResult.PreparationFailure failed
                                                && failed.route() == item
                                                && failed.detail().equals(failure.getMessage())));
        assertFalse(item.future().get(5, TimeUnit.SECONDS).isSuccess());
        SchedulerTestSupport.awaitRequestWork(registry.runtime);
        assertOccupancy(0, 0);
        assertEquals(0, decode.routingView().engineCapacityUsed());
        assertEquals(0, decode.routingView().inflightHardKv());
        assertEquals(0, EndpointTestSupport.expectedReservedKv(decode.resourceSnapshot()));
        verifyNoInteractions(grpc);
        var admission = registry.runtime.deliveries().tryReserveDelivery();
        assertTrue(admission.accepted(), "admission permit must be available after RPC completion");
        admission.value().close();
    }

    @Test
    void rejectedExecutorSubmissionClosesCommittedAdmissionExactlyOnce() throws Exception {
        RequestRoute item = item(1L);
        var rejection = new java.util.concurrent.RejectedExecutionException("executor rejected");
        var original =
                (ThreadPoolExecutor)
                        ReflectionTestUtils.getField(
                                registry.runtime.deliveries(), "dispatchExecutor");
        ThreadPoolExecutor rejected = mock(ThreadPoolExecutor.class);
        doThrow(rejection).when(rejected).execute(any());
        ReflectionTestUtils.setField(registry.runtime.deliveries(), "dispatchExecutor", rejected);
        try {
            submit(List.of(item));
        } finally {
            ReflectionTestUtils.setField(
                    registry.runtime.deliveries(), "dispatchExecutor", original);
        }
        assertFalse(item.future().get(5, TimeUnit.SECONDS).isSuccess());
        SchedulerTestSupport.awaitRequestWork(registry.runtime);
        verify(rejected, times(1)).execute(any());
        assertOccupancy(0, 0);
        assertEquals(0, decode.routingView().engineCapacityUsed());
        assertEquals(0, decode.routingView().inflightHardKv());
        assertEquals(0, EndpointTestSupport.expectedReservedKv(decode.resourceSnapshot()));
        verifyNoInteractions(grpc);
        var admission = registry.runtime.deliveries().tryReserveDelivery();
        assertTrue(admission.accepted(), "admission permit must be available after RPC completion");
        admission.value().close();
    }

    @Test
    void payloadFailureAfterClaimUsesExistingNotSentSettlement() throws Exception {
        RequestRoute item =
                item(1L, com.google.protobuf.ByteString.copyFrom(new byte[] {(byte) 0xff}));
        submit(List.of(item));
        releaseExecutor.countDown();
        awaitDispatchTasks();
        assertFalse(item.future().get(5, TimeUnit.SECONDS).isSuccess());
        SchedulerTestSupport.awaitRequestWork(registry.runtime);
        assertOccupancy(0, 0);
        assertEquals(0, decode.routingView().engineCapacityUsed());
        assertEquals(0, decode.routingView().inflightHardKv());
        assertEquals(0, EndpointTestSupport.expectedReservedKv(decode.resourceSnapshot()));
        verifyNoInteractions(grpc);
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.MethodSource("invalidMemberSubsets")
    void everyInvalidMemberSubsetPreservesPayloadAndRealResourceLedgers(
            String cause, int invalidMask) throws Exception {
        var items = List.of(item(11L), item(12L), item(13L));
        submit(items);
        assertOccupancy(1, 3);
        var survivors = new java.util.ArrayList<RequestRoute>();
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
                case "DEADLINE" -> expireSchedulingFact(context);
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
        SchedulerTestSupport.awaitRequestWork(registry.runtime);
        for (int index = 0; index < items.size(); index++) {
            if ((invalidMask & (1 << index)) != 0) {
                assertFalse(items.get(index).future().get(5, TimeUnit.SECONDS).isSuccess());
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
                    survivors.stream().map(RequestRoute::requestId).toList(),
                    sent.getFirst().getDpSlotsList().stream()
                            .flatMap(requestContext -> requestContext.getRequestsList().stream())
                            .map(request -> request.getInput().getRequestId())
                            .toList());
            var ack = EngineRpcService.EnqueueBatchResponsePB.newBuilder().setBatchId(201L);
            survivors.forEach(
                    item ->
                            ack.addSuccesses(
                                    EngineRpcService.EnqueueBatchSuccessPB.newBuilder()
                                            .setRequestId(item.requestId())));
            reply.complete(ack.build());
            for (RequestRoute item : survivors) {
                assertTrue(item.future().get(5, TimeUnit.SECONDS).isSuccess());
                ledger.finish(201L, item)
                        .forEach(
                                requestStatus ->
                                        RequestProtocolTestSupport.applyPrefillStatus(
                                                registry,
                                                prefill,
                                                RoleType.PREFILL,
                                                requestStatus));
                DeliverySettlementTestSupport.decodeStatus(decode, item.requestId(), true);
            }
        }
        SchedulerTestSupport.awaitRequestWork(registry.runtime);
        for (int index = 0; index < items.size(); index++) {
            if ((invalidMask & (1 << index)) != 0) {
                registry.cancel(items.get(index).requestId(), 0L, CancelReason.CLIENT_CANCELLED);
            }
        }
        assertOccupancy(0, 0);
        assertEquals(0, decode.routingView().engineCapacityUsed());
        assertEquals(0, decode.routingView().inflightHardKv());
        assertEquals(0, EndpointTestSupport.expectedReservedKv(decode.resourceSnapshot()));
        var admission = registry.runtime.deliveries().tryReserveDelivery();
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
                .onDeliveryEvidence(
                        any(),
                        argThat(
                                e ->
                                        e instanceof ContextResult.TransportResult result
                                                && result.delivery().item() == first));
        submit(List.of(first, second));
        releaseExecutor.countDown();
        awaitDispatchTasks();
        var ack = EngineRpcService.EnqueueBatchResponsePB.newBuilder().setBatchId(201L);
        ack.addSuccesses(
                EngineRpcService.EnqueueBatchSuccessPB.newBuilder()
                        .setRequestId(first.requestId()));
        ack.addSuccesses(
                EngineRpcService.EnqueueBatchSuccessPB.newBuilder()
                        .setRequestId(second.requestId()));
        reply.complete(ack.build());
        assertTrue(callbackExited.await(2, TimeUnit.SECONDS));
        assertTrue(first.future().get(2, TimeUnit.SECONDS).isSuccess());
        assertTrue(second.future().get(2, TimeUnit.SECONDS).isSuccess());
        registry.runtime.deliveries().awaitDeliveries();
        for (var route : List.of(first, second)) {
            assertEquals(
                    ContextResult.SendState.DELIVERED, route.ctx().snapshot().delivery().send());
            ledger.finish(201L, route)
                    .forEach(
                            status ->
                                    RequestProtocolTestSupport.applyPrefillStatus(
                                            registry, prefill, RoleType.PREFILL, status));
            DeliverySettlementTestSupport.decodeStatus(decode, route.requestId(), true);
        }
        SchedulerTestSupport.awaitRequestWork(registry.runtime);
        assertOccupancy(0, 0);
        assertEquals(0, decode.routingView().engineCapacityUsed());
        assertEquals(0, decode.resourceSnapshot().activeDispatchPermits());
        assertEquals(0, registry.requests.liveRequestCount());
        assertSame(
                observerFailure,
                ReflectionTestUtils.getField(registry.runtime, "failure"),
                "background evidence failures must survive into shutdown reporting");
    }

    @Test
    void temporaryCloseFailuresRemainReportedWithoutChangingAcceptedRpcResult() throws Exception {
        decode = org.mockito.Mockito.spy(decode);
        var first = item(981_021L);
        var second = item(981_022L);
        var permitFailure =
                new IllegalStateException("permit close observer failed after actual release");
        var handoffFailure =
                new IllegalStateException("handoff close observer failed after actual release");
        org.mockito.Mockito.doAnswer(
                        call -> {
                            org.flexlb.balance.endpoint.DecodeResources.Acquisition<
                                            org.flexlb.balance.endpoint.DecodeResources
                                                    .EngineDispatchPermitAcquireStatus,
                                            DecodeEndpoint.EngineDispatchPermit>
                                    acquired =
                                            (org.flexlb.balance.endpoint.DecodeResources
                                                                    .Acquisition<
                                                            org.flexlb.balance.endpoint
                                                                    .DecodeResources
                                                                    .EngineDispatchPermitAcquireStatus,
                                                            DecodeEndpoint.EngineDispatchPermit>)
                                                    call.callRealMethod();
                            var permit = org.mockito.Mockito.spy(acquired.handle());
                            org.mockito.Mockito.doAnswer(
                                            release -> {
                                                release.callRealMethod();
                                                throw permitFailure;
                                            })
                                    .when(permit)
                                    .release();
                            return new org.flexlb.balance.endpoint.DecodeResources.Acquisition<>(
                                    acquired.status(), permit);
                        })
                .when(decode)
                .acquireDispatchPermit(
                        org.mockito.ArgumentMatchers.eq(first.decodeReservation()), any());
        org.mockito.Mockito.doAnswer(
                        call -> {
                            PrefillState.CommittedHandoff actual =
                                    (PrefillState.CommittedHandoff) call.callRealMethod();
                            var committed = org.mockito.Mockito.spy(actual);
                            org.mockito.Mockito.doAnswer(
                                            close -> {
                                                close.callRealMethod();
                                                throw handoffFailure;
                                            })
                                    .when(committed)
                                    .close();
                            return committed;
                        })
                .when(worker)
                .commitSelection(any());
        submit(List.of(first, second));
        releaseExecutor.countDown();
        awaitDispatchTasks();
        var ack = EngineRpcService.EnqueueBatchResponsePB.newBuilder().setBatchId(201L);
        ack.addSuccesses(
                EngineRpcService.EnqueueBatchSuccessPB.newBuilder()
                        .setRequestId(first.requestId()));
        ack.addSuccesses(
                EngineRpcService.EnqueueBatchSuccessPB.newBuilder()
                        .setRequestId(second.requestId()));
        reply.complete(ack.build());
        assertTrue(first.future().get(2, TimeUnit.SECONDS).isSuccess());
        assertTrue(second.future().get(2, TimeUnit.SECONDS).isSuccess());
        registry.runtime.deliveries().awaitDeliveries();
        for (var route : List.of(first, second)) {
            assertEquals(
                    ContextResult.SendState.DELIVERED, route.ctx().snapshot().delivery().send());
            ledger.finish(201L, route)
                    .forEach(
                            status ->
                                    RequestProtocolTestSupport.applyPrefillStatus(
                                            registry, prefill, RoleType.PREFILL, status));
            DeliverySettlementTestSupport.decodeStatus(decode, route.requestId(), true);
        }
        SchedulerTestSupport.awaitRequestWork(registry.runtime);
        assertOccupancy(0, 0);
        assertEquals(0, decode.routingView().engineCapacityUsed());
        assertEquals(0, decode.resourceSnapshot().activeDispatchPermits());
        assertEquals(0, registry.requests.liveRequestCount());
        assertSame(permitFailure, ReflectionTestUtils.getField(registry.runtime, "failure"));
        assertTrue(java.util.Arrays.asList(permitFailure.getSuppressed()).contains(handoffFailure));
    }

    private void expireSchedulingFact(RequestContext context) {
        long past = System.currentTimeMillis() - 1L;
        ReflectionTestUtils.setField(
                context.data(),
                "schedulingMetadata",
                org.flexlb.dao.SchedulingMetadata.explicit(context.data().getPriority(), past));
        ReflectionTestUtils.setField(context, "absoluteDeadline", past);
        ReflectionTestUtils.invokeMethod(context, "refresh", System.currentTimeMillis());
    }

    @ParameterizedTest
    @CsvSource({
        "true,50,false,true",
        "false,50,false,false",
        "true,40,false,false",
        "true,50,true,false"
    })
    void onlySamePriorityRequestsOnAnotherDecodeMayBypass(
            boolean priorityOrdering, int candidatePriority, boolean sameDecode, boolean allowed) {
        enableBypass(priorityOrdering);
        var head = item(1L);
        var other = sameDecode ? decode : anotherDecode();
        var candidate = item(2L, null, other, candidatePriority);
        var blocker = blockDecode();
        try {
            assertEquals(allowed, worker.canBypass(head, candidate));
            cycle();
            assertTrue(ledger.prefill.captureQueue(10).items().contains(head));
            assertEquals(!allowed, ledger.prefill.captureQueue(10).items().contains(candidate));
            assertFalse(worker.canBypass(head, candidate));
        } finally {
            decode.release(blocker, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        }
    }

    @Test
    void wakeupsAndNewArrivalsCannotRenewTheSingleBypassAllowance() {
        enableBypass(true);
        var head = item(1L);
        var other = anotherDecode();
        var first = item(2L, null, other, 50);
        var blocker = blockDecode();
        try {
            cycle();
            var next = item(3L, null, other, 50);
            worker.signal(false);
            cycle();
            cycle();
            assertEquals(List.of(head, next), ledger.prefill.captureQueue(10).items());
            assertFalse(worker.canBypass(head, next));
            assertEquals(1, other.resourceSnapshot().activeDispatchPermits());
            assertEquals(0, decode.resourceSnapshot().activeDispatchPermits());
            assertFalse(ledger.prefill.captureQueue(10).items().contains(first));
        } finally {
            decode.release(blocker, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        }
    }

    @Test
    void failedBypassCommitReleasesTemporaryResourcesAndRetainsAllowance() {
        enableBypass(true);
        var head = item(1L);
        var other = anotherDecode();
        var candidate = item(2L, null, other, 50);
        var blocker = blockDecode();
        doReturn(null).doCallRealMethod().when(worker).commitSelection(any());
        try {
            cycle();
            assertTrue(worker.canBypass(head, candidate));
            assertEquals(0, other.resourceSnapshot().activeDispatchPermits());
            assertEquals(0, ledger.prefill.stats().batchCount());
            cycle();
            assertFalse(worker.canBypass(head, candidate));
            assertEquals(List.of(head), ledger.prefill.captureQueue(10).items());
        } finally {
            decode.release(blocker, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void cancellationBeforeBypassCommitReleasesEveryTemporaryPermit(boolean cancelHead) {
        enableBypass(true);
        var head = item(1L);
        var other = anotherDecode();
        var candidate = item(2L, null, other, 50);
        var blocker = blockDecode();
        doAnswer(
                        call -> {
                            registry.cancel(
                                    cancelHead ? head.requestId() : candidate.requestId(),
                                    0L,
                                    CancelReason.CLIENT_CANCELLED);
                            SchedulerTestSupport.awaitRequestWork(registry.runtime);
                            return call.callRealMethod();
                        })
                .when(worker)
                .commitSelection(any());
        try {
            cycle();
            assertEquals(
                    List.of(cancelHead ? candidate : head),
                    ledger.prefill.captureQueue(10).items());
            assertEquals(0, other.resourceSnapshot().activeDispatchPermits());
            assertEquals(0, ledger.prefill.stats().batchCount());
            var permit = registry.runtime.deliveries().tryReserveDelivery();
            assertTrue(permit.accepted());
            permit.value().close();
            if (!cancelHead) {
                assertTrue(worker.canBypass(head, item(3L, null, other, 50)));
            }
        } finally {
            decode.release(blocker, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        }
    }

    @Test
    void recoveringHeadAbortsBypassBeforeCommit() {
        enableBypass(true);
        var head = item(1L);
        var other = anotherDecode();
        var candidate = item(2L, null, other, 50);
        var blocker = blockDecode();
        doAnswer(
                        call -> {
                            decode.release(blocker, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
                            return call.callRealMethod();
                        })
                .when(worker)
                .commitSelection(any());
        cycle();
        assertEquals(List.of(head, candidate), ledger.prefill.captureQueue(10).items());
        assertEquals(0, other.resourceSnapshot().activeDispatchPermits());
        assertEquals(0, ledger.prefill.stats().batchCount());
        assertTrue(worker.canBypass(head, candidate));
        cycle();
        assertFalse(ledger.prefill.captureQueue(10).items().contains(head));
    }

    @Test
    void secondBlockedRequestStopsBypassInsteadOfSkippingAnotherOwner() {
        enableBypass(true);
        var head = item(1L);
        var second = item(2L);
        item(3L, null, anotherDecode(), 50);
        var blocker = blockDecode();
        try {
            cycle();
            assertEquals(3, ledger.prefill.captureQueue(10).items().size());
            assertFalse(worker.canBypass(head, second));
            assertEquals(0, ledger.prefill.stats().batchCount());
        } finally {
            decode.release(blocker, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        }
    }

    @Test
    void bypassWaitNoticesArrivalsUntilItsAllowanceIsUsed() {
        enableBypass(true);
        var head = item(1L);
        var guard = ledger.prefill.queueGuard();
        long version = ledger.prefill.captureQueue(10).queueVersion();
        var candidate = item(2L, null, anotherDecode(), 50);
        guard.lock();
        try {
            assertFalse(
                    ledger.prefill.queueWaitCurrentLocked(
                            head, version, 0L, true, System.currentTimeMillis()));
        } finally {
            guard.unlock();
        }
        var blocker = blockDecode();
        try {
            cycle();
            item(3L, null, anotherDecode(), 50);
            guard.lock();
            try {
                assertTrue(
                        ledger.prefill.queueWaitCurrentLocked(
                                head, version, 0L, true, System.currentTimeMillis()));
            } finally {
                guard.unlock();
            }
            assertFalse(worker.canBypass(head, candidate));
        } finally {
            decode.release(blocker, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        }
    }

    @Test
    void arrivalWakesTheRealDecodeWaitAndUsesItsAllowanceOnce() throws InterruptedException {
        enableBypass(true);
        // The thread callback must observe the actual instance, rather than a Mockito spy copy.
        worker =
                new PrefillQueueRunner(
                        "bypass-wakeup",
                        prefill,
                        (QueueExecutionSettings) ReflectionTestUtils.getField(worker, "settings"),
                        dispatcher,
                        ledger.prefill);
        ReflectionTestUtils.setField(prefill, "queueRunner", worker);
        var head = item(1L);
        var blocker = blockDecode();
        worker.start();
        try {
            RequestProtocolTestSupport.awaitCondition(
                    () ->
                            worker.getLatestQueueWaitSnapshot()
                                    .containsValue(
                                            org.flexlb.balance.delivery.CapacityBoundary.Resource
                                                    .DECODE_ENGINE
                                                    .getDescription()));
            var other = anotherDecode();
            var candidate = item(2L, null, other, 50);
            worker.signal(false);
            RequestProtocolTestSupport.awaitCondition(
                    () -> !ledger.prefill.captureQueue(10).items().contains(candidate));
            assertFalse(worker.canBypass(head, candidate));
            assertEquals(List.of(head), ledger.prefill.captureQueue(10).items());
        } finally {
            assertNull(worker.stopAndAwait());
            decode.release(blocker, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        }
    }

    private void enableBypass(boolean priorityOrdering) {
        enableBypass(priorityOrdering, DecisionPolicyConfig.Type.FIXED_WINDOW, 2);
    }

    private void enableBypass(
            boolean priorityOrdering, DecisionPolicyConfig.Type grouping, int maxRequests) {
        ReflectionTestUtils.setField(prefill, "predictor", new FormulaPredictor("1"));
        ReflectionTestUtils.setField(
                registry.runtime.deliveries(),
                "decodeDeliveryCapacity",
                new DecodeResources.AdmissionCapacity(1, 100L));
        worker =
                org.mockito.Mockito.spy(
                        new PrefillQueueRunner(
                                "bypass",
                                prefill,
                                new QueueExecutionSettings(
                                        priorityOrdering,
                                        config.getDispatcher().getType(),
                                        grouping,
                                        maxRequests,
                                        0L,
                                        0L),
                                dispatcher,
                                ledger.prefill));
        ReflectionTestUtils.setField(prefill, "queueRunner", worker);
    }

    @ParameterizedTest
    @CsvSource({
        "false,SINGLE,1", "true,SINGLE,1",
        "false,FIXED_WINDOW,1", "true,FIXED_WINDOW,1",
        "false,FIXED_WINDOW,2", "true,FIXED_WINDOW,2",
        "false,FIXED_WINDOW,8", "true,FIXED_WINDOW,8"
    })
    void batchQueueBoundsOvertaking(
            boolean priority, DecisionPolicyConfig.Type grouping, int maxRequests)
            throws Exception {
        assertBoundedOvertaking(priority, grouping, maxRequests);
    }

    @ParameterizedTest
    @org.junit.jupiter.api.Tag("non-batch")
    @CsvSource({
        "false,SINGLE,1", "true,SINGLE,1",
        "false,FIXED_WINDOW,1", "true,FIXED_WINDOW,1",
        "false,FIXED_WINDOW,2", "true,FIXED_WINDOW,2",
        "false,FIXED_WINDOW,8", "true,FIXED_WINDOW,8"
    })
    void nonBatchQueueBoundsOvertaking(
            boolean priority, DecisionPolicyConfig.Type grouping, int maxRequests)
            throws Exception {
        assertBoundedOvertaking(priority, grouping, maxRequests);
    }

    private void assertBoundedOvertaking(
            boolean priority, DecisionPolicyConfig.Type grouping, int maxRequests)
            throws Exception {
        configureOrderedQueue(priority, grouping, maxRequests);
        var head = item(100L);
        var candidates = new ArrayList<RequestRoute>();
        for (int i = 0; i < maxRequests + 1; i++) {
            candidates.add(item(90L - i, null, anotherDecode(i + 2), 50));
        }
        var blocker = blockDecode();
        int overtaken = priority ? Math.max(1, maxRequests - 1) : 0;
        boolean batch = config.getDispatcher().getType() == DispatcherConfig.Type.BATCH;
        try {
            cycle();
            var remaining = new ArrayList<>(List.of(head));
            remaining.addAll(candidates.subList(overtaken, candidates.size()));
            assertEquals(remaining, ledger.prefill.captureQueue(100).items());
            assertEquals(0, decode.resourceSnapshot().activeDispatchPermits());
            for (int i = 0; i < candidates.size(); i++) {
                if (!batch) {
                    if (i < overtaken) {
                        var response = candidates.get(i).future().get(2, TimeUnit.SECONDS);
                        assertTrue(response.isSuccess(), response.toString());
                    } else {
                        assertFalse(candidates.get(i).future().isDone());
                    }
                }
                var resources = candidates.get(i).decodeEp().resourceSnapshot();
                assertEquals(i < overtaken ? 1 : 0, resources.engineCapacityUsed());
                assertEquals(batch && i < overtaken ? 1 : 0, resources.activeDispatchPermits());
            }
            assertEquals(batch && priority ? 1 : 0, ledger.prefill.stats().batchCount());
            releaseExecutor.countDown();
            awaitDispatchTasks();
            if (batch && priority) {
                assertEquals(1, sent.size());
                assertEquals(
                        candidates.subList(0, overtaken).stream()
                                .map(RequestRoute::requestId)
                                .toList(),
                        sent.getFirst().getDpSlotsList().stream()
                                .flatMap(dp -> dp.getRequestsList().stream())
                                .map(request -> request.getInput().getRequestId())
                                .toList());
            } else {
                assertTrue(sent.isEmpty());
            }
            var later = item(1L, null, anotherDecode(20), 50);
            remaining.add(later);
            for (int i = 0; i < 3; i++) {
                worker.signal(false);
                cycle();
            }
            assertEquals(remaining, ledger.prefill.captureQueue(100).items());
            assertFalse(worker.canBypass(head, later));
            assertEquals(0, later.decodeEp().resourceSnapshot().activeDispatchPermits());
            assertFalse(head.future().isDone());
            if (!batch) {
                verify(registry.runtime.deliveries(), org.mockito.Mockito.never())
                        .tryReserveDelivery();
                verify(prefill, org.mockito.Mockito.never())
                        .reserveBatch(any(), anyLong(), anyInt());
                verifyNoInteractions(grpc);
            }
            if (!priority) {
                decode.release(blocker, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
                blocker = null;
                cycle();
                assertEquals(
                        remaining.subList(maxRequests, remaining.size()),
                        ledger.prefill.captureQueue(100).items());
                assertEquals(1, decode.resourceSnapshot().engineCapacityUsed());
            }
        } finally {
            if (blocker != null) {
                decode.release(blocker, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
            }
        }
    }

    @Test
    void fifoReofferAndArrivalDuringCommitPreserveSelectedPrefix() throws Exception {
        configureOrderedQueue(false, DecisionPolicyConfig.Type.FIXED_WINDOW, 2);
        var first = item(100L);
        var second = item(10L, null, anotherDecode(2), 50);
        var third = item(50L, null, anotherDecode(3), 50);
        ledger.prefill.updateQueue(
                PrefillState.QueueChange.REMOVE, first, System.currentTimeMillis());
        ledger.enqueue(first);
        assertEquals(List.of(first, second, third), ledger.prefill.captureQueue(10).items());
        var arrival = new CompletableFuture<RequestRoute>();
        doAnswer(
                        call -> {
                            arrival.complete(
                                    CompletableFuture.supplyAsync(
                                                    () -> item(1L, null, anotherDecode(4), 90))
                                            .get(5, TimeUnit.SECONDS));
                            return call.callRealMethod();
                        })
                .when(worker)
                .commitSelection(any());
        cycle();
        assertEquals(
                List.of(third, arrival.get(5, TimeUnit.SECONDS)),
                ledger.prefill.captureQueue(10).items());
        releaseExecutor.countDown();
        awaitDispatchTasks();
        assertEquals(1, sent.size());
        assertEquals(
                List.of(100L, 10L),
                sent.getFirst().getDpSlotsList().stream()
                        .flatMap(dp -> dp.getRequestsList().stream())
                        .map(request -> request.getInput().getRequestId())
                        .toList());
        assertEquals(0, third.decodeEp().resourceSnapshot().activeDispatchPermits());
        assertEquals(0, arrival.get().decodeEp().resourceSnapshot().activeDispatchPermits());
    }

    private void configureOrderedQueue(
            boolean priority, DecisionPolicyConfig.Type grouping, int maxRequests) {
        ledger =
                new DeliverySettlementTestSupport(
                        priority
                                ? PrefillQueueRunner.PRIORITY_QUEUE_ORDER
                                : PrefillQueueRunner.FIFO_QUEUE_ORDER);
        ReflectionTestUtils.setField(prefill, "ownership", ledger.prefill);
        var execution = registry.runtime.deliveries();
        org.mockito.Mockito.doCallRealMethod().when(execution).nextBatchId();
        enableBypass(priority, grouping, maxRequests);
    }

    @Test
    void invalidCandidatePredictionDoesNotTerminateTheBlockedHead() throws Exception {
        enableBypass(true);
        var head = item(1L);
        var other = anotherDecode();
        var candidate = item(2L, null, other, 50);
        var blocker = blockDecode();
        ReflectionTestUtils.setField(prefill, "predictor", new FormulaPredictor("-1"));
        try {
            cycle();
            assertEquals(List.of(head), ledger.prefill.captureQueue(10).items());
            assertFalse(head.future().isDone());
            assertFalse(candidate.future().get(2, TimeUnit.SECONDS).isSuccess());
            assertEquals(0, other.resourceSnapshot().activeDispatchPermits());
            assertEquals(0, ledger.prefill.stats().batchCount());
            var next = item(3L, null, other, 50);
            assertTrue(worker.canBypass(head, next));
        } finally {
            decode.release(blocker, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        }
    }

    @Test
    @org.junit.jupiter.api.Tag("non-batch")
    void singleNonBatchDecisionCanBypassOnlyOnce() {
        enableBypass(true);
        worker =
                new PrefillQueueRunner(
                        "single-bypass",
                        prefill,
                        new QueueExecutionSettings(
                                true,
                                config.getDispatcher().getType(),
                                org.flexlb.config.DecisionPolicyConfig.Type.SINGLE,
                                1,
                                0L,
                                0L),
                        dispatcher,
                        ledger.prefill);
        ReflectionTestUtils.setField(prefill, "queueRunner", worker);
        var head = item(1L);
        var other = anotherDecode();
        item(2L, null, other, 50);
        var blocker = blockDecode();
        try {
            cycle();
            var next = item(3L, null, other, 50);
            cycle();
            assertEquals(List.of(head, next), ledger.prefill.captureQueue(10).items());
            assertFalse(worker.canBypass(head, next));
            assertEquals(0, ledger.prefill.stats().batchCount());
            verify(registry.runtime.deliveries(), org.mockito.Mockito.never()).tryReserveDelivery();
            verify(prefill, org.mockito.Mockito.never()).reserveBatch(any(), anyLong(), anyInt());
            verifyNoInteractions(grpc);
        } finally {
            decode.release(blocker, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void bypassReusesPredictionOnlyWhenTheEntireSelectedGroupCommits(boolean truncated) {
        enableBypass(true);
        var evaluator = org.mockito.Mockito.spy(DeliveryTestSupport.EVALUATOR);
        var predictor = mock(org.flexlb.balance.prediction.PrefillTimePredictor.class);
        when(predictor.evaluator()).thenReturn(evaluator);
        ReflectionTestUtils.setField(prefill, "predictor", predictor);
        worker =
                new PrefillQueueRunner(
                        "predicted-bypass",
                        prefill,
                        new QueueExecutionSettings(
                                true,
                                config.getDispatcher().getType(),
                                org.flexlb.config.DecisionPolicyConfig.Type.FIXED_WINDOW,
                                3,
                                0L,
                                1_000L),
                        dispatcher,
                        ledger.prefill);
        var head = item(1L);
        var other = anotherDecode();
        var first = item(2L, null, other, 50);
        var second = item(3L, null, truncated ? other : anotherDecode(), 50);
        var committed = new java.util.concurrent.atomic.AtomicReference<PrefillState.CommitPlan>();
        when(prefill.commitLocked(any()))
                .thenAnswer(
                        call -> {
                            var plan = call.getArgument(0, PrefillState.CommitPlan.class);
                            committed.set(plan);
                            return ledger.commit(plan);
                        });
        var blocker = blockDecode();
        try {
            cycle();
            assertEquals(
                    truncated ? List.of(first) : List.of(first, second), committed.get().items());
            assertEquals(
                    first.seqLen() + (truncated ? 0L : second.seqLen()),
                    committed.get().predictions()[0]);
            verify(evaluator, times(truncated ? 6 : 5)).predictBatchMs(any());
            assertEquals(
                    truncated ? List.of(head, second) : List.of(head),
                    ledger.prefill.captureQueue(10).items());
        } finally {
            decode.release(blocker, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        }
    }

    @Test
    void bypassCannotExceedThePrefillKvCapacity() {
        enableBypass(true);
        var head = item(1L);
        var other = anotherDecode();
        var candidate = item(2L, null, other, 50);
        ReflectionTestUtils.setField(candidate.ctx().data().getRequest(), "seqLen", 32L);
        var status = mock(WorkerStatus.class);
        when(status.committedEngineObservation())
                .thenReturn(
                        new WorkerStatus.EngineObservation(
                                RoleType.PREFILL,
                                null,
                                16L,
                                16L,
                                java.util.Map.of(),
                                0.0,
                                0L,
                                0L,
                                0L,
                                0L,
                                0L,
                                0L,
                                0L,
                                0L));
        when(prefill.getStatus()).thenReturn(status);
        var blocker = blockDecode();
        try {
            cycle();
            assertEquals(List.of(head, candidate), ledger.prefill.captureQueue(10).items());
            assertTrue(worker.canBypass(head, candidate));
            assertEquals(0, other.resourceSnapshot().activeDispatchPermits());
            assertEquals(0, ledger.prefill.stats().batchCount());
        } finally {
            decode.release(blocker, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        }
    }

    @Test
    void newHeadBeforeCommitRejectsBypassWithoutUsingAllowance() {
        enableBypass(true);
        var head = item(10L);
        var other = anotherDecode();
        var candidate = item(20L, null, other, 50);
        var blocker = blockDecode();
        doAnswer(
                        call -> {
                            item(1L, null, other, 90);
                            return call.callRealMethod();
                        })
                .when(worker)
                .commitSelection(any());
        try {
            cycle();
            assertEquals(3, ledger.prefill.captureQueue(10).items().size());
            assertEquals(0, other.resourceSnapshot().activeDispatchPermits());
            assertEquals(0, ledger.prefill.stats().batchCount());
            var first = ledger.prefill.captureQueue(10).head();
            ledger.prefill.updateQueue(
                    PrefillState.QueueChange.REMOVE, first, System.currentTimeMillis());
            assertTrue(worker.canBypass(head, candidate));
        } finally {
            decode.release(blocker, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        }
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

    private DecodeResources.ReservationHandle blockDecode() {
        try (var pin = decode.tryPinGeneration()) {
            return EndpointTestSupport.reserveUnqueuedDecode(decode, pin, 90_000L, 1L, 2L, 50);
        }
    }

    private void cycle() {
        assertNotNull(ReflectionTestUtils.invokeMethod(worker, "processQueue"));
    }

    private RequestRoute item(long id) {
        return item(id, null);
    }

    private RequestRoute item(long id, com.google.protobuf.ByteString payload) {
        return item(id, payload, decode, 50);
    }

    private RequestRoute item(
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
        var future = RequestProtocolTestSupport.register(registry, context);
        DecodeResources.ReservationHandle reservation;
        try (var pin = target.tryPinGeneration()) {
            reservation =
                    EndpointTestSupport.reserveUnqueuedDecode(target, pin, id, 1L, 2L, priority);
        }
        assertNotNull(reservation);
        DeliverySettlementTestSupport.queueDecode(target, reservation);
        ServerStatus status = new ServerStatus();
        status.setRole(RoleType.PREFILL);
        status.setServerIp("127.0.0.1");
        status.setHttpPort(8080);
        status.setGrpcPort(8090);
        context.data().setFuture(future);
        var item =
                SchedulingTestConfig.createRoute(
                        context,
                        new Response(),
                        status,
                        null,
                        prefill,
                        target,
                        reservation,
                        System.currentTimeMillis());
        RequestProtocolTestSupport.bind(
                registry, new RequestProtocolTestSupport.Registered(item, future));
        ledger.enqueue(item);
        return item;
    }

    private void submit(List<RequestRoute> items) {
        var evaluator = new FormulaPredictor("100 * batchSize");
        org.junit.jupiter.api.Assertions.assertNull(
                worker.processSelection(
                        items, "queued-race", evaluator, OptionalLong.empty(), null));
    }

    private void awaitDispatchTasks() throws Exception {
        ThreadPoolExecutor executor =
                (ThreadPoolExecutor)
                        ReflectionTestUtils.getField(
                                registry.runtime.deliveries(), "dispatchExecutor");
        if (executor.isShutdown()) {
            assertTrue(
                    executor.awaitTermination(5, TimeUnit.SECONDS),
                    "a stopped dispatch executor must finish all accepted tasks");
        } else {
            executor.submit(() -> {}).get(5, TimeUnit.SECONDS);
        }
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
