package org.flexlb.balance.scheduler;

import static org.flexlb.balance.scheduler.DeliveryStrategy.failUnsentDelivery;

import org.flexlb.balance.endpoint.EndpointTestSupport;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.PrefillState;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DeliverySettlementTestSupport;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.prediction.FormulaPredictor;
import org.flexlb.balance.scheduler.ExpirationTimer.InactivityDeadline;
import org.flexlb.config.ConfigService;
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

import java.util.List;
import java.util.OptionalLong;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import static org.flexlb.balance.scheduler.RequestProtocolTestSupport.await;
import static org.flexlb.balance.scheduler.SchedulingTestConfig.freezeInputs;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.doAnswer;
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
    private final CompletableFuture<EngineRpcService.EnqueueBatchResponsePB> reply = new CompletableFuture<>();
    private final DeliverySettlementTestSupport ledger = new DeliverySettlementTestSupport();
    private FlexlbConfig config;
    private AbstractRequestScheduler registry;
    private DefaultBatchDispatcher dispatcher;
    private EngineGrpcClient grpc;
    private PrefillEndpoint prefill;
    private DecodeEndpoint decode;
    private BatchDeliveryStrategy strategy;
    private final java.util.concurrent.CopyOnWriteArrayList<EngineRpcService.EnqueueBatchRequestPB> sent =
            new java.util.concurrent.CopyOnWriteArrayList<>();

    @BeforeEach
    void setUp() {
        config = SchedulingTestConfig.batchConfig();
        config.getRequestLifecycle().getRequest().setTimeoutMs(TIMEOUT_MS);
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        DeliveryMetricsReporter reporter = mock(DeliveryMetricsReporter.class);
        registry = org.flexlb.balance.scheduler.SchedulerTestSupport.create(service, reporter, mock(RequestSchedulerReporter.class),
                mock(RecentCacheKeyTraceReporter.class));
        grpc = mock(EngineGrpcClient.class);
        when(grpc.batchEnqueueAsync(anyString(), anyInt(), any())).thenAnswer(call -> {
            sent.add(call.getArgument(2));
            return reply;
        });
        dispatcher = SchedulerTestSupport.createDispatcher(grpc, service, 1, 1);
        strategy = new BatchDeliveryStrategy(dispatcher::tryPrepareSubmission, () -> 201L, reporter);
        prefill = mock(PrefillEndpoint.class);
        when(prefill.getIp()).thenReturn("127.0.0.1");
        when(prefill.getGrpcPort()).thenReturn(8090);
        when(prefill.reserveBatch(any(), anyLong(), anyInt())).thenAnswer(call ->
                ledger.reserveBatch(call.getArgument(0), call.getArgument(1), call.getArgument(2)));
        when(prefill.releaseRequest(any())).thenAnswer(call -> {
            RequestRoute item = call.getArgument(0);
            assertFalse(Thread.holdsLock(registry.findRequestContext(item.requestId())));
            return org.flexlb.balance.endpoint.EndpointTestSupport.releaseRequest(ledger.prefill, item);
        });
        decode = EndpointTestSupport.decode(WorkerStatus.createDiscovered(RoleType.DECODE, null,
                "127.0.0.1", 8080, 8081, null), org.flexlb.balance.scheduler.SchedulerTestSupport.repository(registry));
        CountDownLatch entered = new CountDownLatch(1);
        dispatcher.tryPrepareSubmission().value().submit(sender -> {
            entered.countDown();
            await(releaseExecutor);
        });
        await(entered);
    }

    @AfterEach
    void close() throws Exception {
        releaseExecutor.countDown();
        awaitDispatchTasks();
        reply.complete(EngineRpcService.EnqueueBatchResponsePB.newBuilder().setBatchId(201L).build());
        dispatcher.shutdown();
        if (RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(registry)) {
            registry.closeOutstandingAndTerminalize();
            org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry).timer().close();
            org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry).closeRequestExecutors();
        }
    }

    @ParameterizedTest
    @CsvSource({"CANCEL, false", "CANCEL, true", "TIMER, false", "TIMER, true",
            "LATE_DEADLINE, false", "LATE_DEADLINE, true", "LATE_INACTIVITY, false", "LATE_INACTIVITY, true"})
    void invalidQueuedMembersNeverCrossHandoffAndReleaseExactResources(String expiry, boolean all)
            throws Exception {
        RequestRoute first = item(1L);
        RequestRoute second = item(2L);
        submit(List.of(first, second));
        assertEquals(DeliveryClaimKind.NONE, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(registry).getRequestState(1L, 0L).deliveryClaimKind());
        assertEquals(DeliveryClaimKind.NONE, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(registry).getRequestState(2L, 0L).deliveryClaimKind());
        assertOccupancy(1, 2);
        for (RequestRoute item : all ? List.of(first, second) : List.of(first)) {
            RequestContext requestContext = registry.findRequestContext(item.requestId());
            switch (expiry) {
                case "CANCEL" -> registry.cancel(item.requestId(), 0L, CancelReason.CLIENT_CANCELLED);
                case "TIMER" -> RequestProtocolTestSupport.expireInactiveRequest(registry, requestContext, requestContext.createdAtMs() + TIMEOUT_MS);
                case "LATE_DEADLINE" -> ReflectionTestUtils.setField(requestContext, "schedulingMetadata",
                        org.flexlb.dao.SchedulingMetadata.explicit(requestContext.getPriority(), System.currentTimeMillis() - 1L));
                case "LATE_INACTIVITY" -> ReflectionTestUtils.setField(requestContext, "lastWorkerStatusAtMs",
                        System.currentTimeMillis() - TIMEOUT_MS - 1L);
                default -> throw new AssertionError(expiry);
            }
        }
        releaseExecutor.countDown();
        awaitDispatchTasks();
        assertFalse(first.future().get(5, TimeUnit.SECONDS).isSuccess());
        assertEquals(DeliveryClaimKind.NONE, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(registry).getRequestState(1L, 0L).deliveryClaimKind());
        int survivors = all ? 0 : 1;
        if (all) { assertFalse(second.future().get(5, TimeUnit.SECONDS).isSuccess()); }
        assertOccupancy(survivors, survivors);
        assertEquals(survivors, decode.routingView().engineCapacityUsed());
        assertEquals(survivors, decode.routingView().inflightHardKv());
        assertEquals(2L * survivors, EndpointTestSupport.expectedReservedKv(decode.resourceSnapshot()));
        assertEquals(survivors, sent.size());
        if (all) {
            verifyNoInteractions(grpc);
        } else {
            assertEquals(List.of(2L), sent.getFirst().getDpSlotsList().stream()
                    .flatMap(dp -> dp.getRequestsList().stream()).map(input -> input.getInput().getRequestId()).toList());
            assertEquals(DeliveryClaimKind.BATCH_ENQUEUE, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(registry).getRequestState(2L, 0L).deliveryClaimKind());
            reply.complete(ack(2L));
            assertTrue(second.future().get(5, TimeUnit.SECONDS).isSuccess());
            ledger.finish(201L, second).forEach(requestStatus -> RequestProtocolTestSupport.applyPrefillStatus(registry, prefill, RoleType.PREFILL, requestStatus));
            DeliverySettlementTestSupport.decodeStatus(decode, 2L, true);
        }
        // Repeated terminal events cannot subtract the other member or leak a batch permit.
        registry.cancel(1L, 0L, CancelReason.CLIENT_CANCELLED);
        assertOccupancy(0, 0);
        assertEquals(0, decode.routingView().engineCapacityUsed());
        assertEquals(0, decode.routingView().inflightHardKv());
        assertEquals(0, EndpointTestSupport.expectedReservedKv(decode.resourceSnapshot()));
        dispatcher.tryPrepareSubmission().value().close();
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void lateHandoffStartsOneBoundedObservationWindowWithoutFabricatingWorkerActivity(boolean ack)
            throws Exception {
        RequestRoute item = item(1L);
        RequestContext requestContext = registry.findRequestContext(1L);
        long lastStatus = System.currentTimeMillis() - TIMEOUT_MS + 10_000L;
        ReflectionTestUtils.setField(requestContext, "lastWorkerStatusAtMs", lastStatus);
        submit(List.of(item));
        releaseExecutor.countDown();
        awaitDispatchTasks();
        assertEquals(1, sent.size());
        long handoff = (long) ReflectionTestUtils.getField(requestContext, "batchEnqueueStartedAtMs");
        assertEquals(lastStatus, ReflectionTestUtils.getField(requestContext, "lastWorkerStatusAtMs"));
        if (ack) {
            reply.complete(ack(1L));
            assertTrue(item.future().get(5, TimeUnit.SECONDS).isSuccess());
        }
        InactivityDeadline originalTimer = (InactivityDeadline) ReflectionTestUtils.getField(requestContext, "inactivityDeadline");
        assertNotNull(originalTimer);
        RequestProtocolTestSupport.expireInactivity(registry, requestContext, originalTimer, lastStatus + TIMEOUT_MS);
        assertEquals(handoff + TIMEOUT_MS, requestContext.inactivityDeadlineAtMs().orElseThrow());
        assertOccupancy(1, 1);
        RequestProtocolTestSupport.expireInactiveRequest(registry, requestContext, handoff + TIMEOUT_MS - 1);
        assertOccupancy(1, 1);
        var cleanupProof = new java.util.concurrent.CompletableFuture<org.flexlb.balance.eviction.EngineCancelChannel.CancelAck>();
        org.mockito.Mockito.when(SchedulerTestSupport.cancelChannel(registry).cancel(any(), org.mockito.ArgumentMatchers.anyLong(), any(), org.mockito.ArgumentMatchers.anyLong())).thenReturn(cleanupProof);
        RequestProtocolTestSupport.expireInactiveRequest(registry, requestContext, handoff + TIMEOUT_MS);
        assertOccupancy(1, 1);
        cleanupProof.complete(org.flexlb.balance.eviction.EngineCancelChannel.CancelAck.REQUEST_CLEANED);
        if (!ack) { reply.complete(ack(1L)); }
        requestContext.delivery().settlement().toCompletableFuture().get(2, TimeUnit.SECONDS);
        RequestProtocolTestSupport.awaitCondition(() ->
                SchedulerTestSupport.repository(registry).findTerminal(item.requestId()) != null);
        assertOccupancy(0, 0);
        assertEquals(0, decode.routingView().engineCapacityUsed());
        assertEquals(ack, item.future().get(5, TimeUnit.SECONDS).isSuccess());
    }

    @Test
    void blockedRpcDoesNotHoldContextOrTransactionMonitor() throws Exception {
        RequestRoute item = item(1L);
        RequestContext requestContext = registry.findRequestContext(1L);
        CountDownLatch rpcEntered = new CountDownLatch(1);
        CountDownLatch releaseRpc = new CountDownLatch(1);
        when(grpc.batchEnqueueAsync(anyString(), anyInt(), any())).thenAnswer(call -> {
            assertFalse(Thread.holdsLock(requestContext));
            rpcEntered.countDown();
            await(releaseRpc);
            return reply;
        });
        var evaluator = new FormulaPredictor("100");
        try (var transaction = strategy.prepare(List.of(item), evaluator, OptionalLong.empty());
             var contender = java.util.concurrent.Executors.newSingleThreadExecutor()) {
            var preceding = commit(transaction).materialize();
            strategy.deliver(transaction, "blocked-rpc", 0, preceding, evaluator);
            releaseExecutor.countDown();
            try {
                await(rpcEntered);
                contender.submit(() -> {
                    failUnsentDelivery(transaction, new IllegalStateException("scheduler cleanup"), false);
                    transaction.close();
                    synchronized (requestContext) {
                        assertEquals(DeliveryClaimKind.BATCH_ENQUEUE, requestContext.snapshot().deliveryClaimKind());
                    }
                }).get(2, TimeUnit.SECONDS);
                assertOccupancy(1, 1);
            } finally {
                releaseRpc.countDown();
            }
        }
    }

    @Test
    void unresolvedCommittedDeliveryIsAbortedWithoutAnOriginalFailure() throws Exception {
        RequestRoute item = item(1L);
        try (var transaction = strategy.prepare(List.of(item), new FormulaPredictor("100"), OptionalLong.empty())) {
            commit(transaction);
            failUnsentDelivery(transaction, null, false);
            failUnsentDelivery(transaction, null, false);
        }
        var response = item.future().get(5, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        assertTrue(response.getErrorMessage().contains("delivery returned without resolving owner"));
        assertOccupancy(0, 0);
        assertEquals(0, decode.routingView().engineCapacityUsed());
        assertEquals(0, decode.routingView().inflightHardKv());
        verifyNoInteractions(grpc);
    }

    @Test
    void rejectedExecutorSubmissionClosesCommittedAdmissionExactlyOnce() throws Exception {
        RequestRoute item = item(1L);
        var rejection = new java.util.concurrent.RejectedExecutionException("executor rejected");
        DefaultBatchDispatcher.PreparedSubmission submission = mock(DefaultBatchDispatcher.PreparedSubmission.class);
        doThrow(rejection).when(submission).submit(any());
        var rejectingStrategy = new BatchDeliveryStrategy(() -> org.flexlb.balance.delivery.CapacityBoundary.Attempt.accepted(submission), () -> 201L, mock(DeliveryMetricsReporter.class));
        var evaluator = new FormulaPredictor("100");
        try (var transaction = rejectingStrategy.prepare(List.of(item), evaluator, OptionalLong.empty())) {
            var preceding = commit(transaction).materialize();
            assertSame(rejection, assertThrows(java.util.concurrent.RejectedExecutionException.class,
                    () -> rejectingStrategy.deliver(transaction, "rejected", 0, preceding, evaluator)));
            failUnsentDelivery(transaction, rejection, false);
        }
        assertFalse(item.future().get(5, TimeUnit.SECONDS).isSuccess());
        verify(submission, times(1)).close();
        assertOccupancy(0, 0);
        assertEquals(0, decode.routingView().engineCapacityUsed());
        assertEquals(0, decode.routingView().inflightHardKv());
        assertEquals(0, EndpointTestSupport.expectedReservedKv(decode.resourceSnapshot()));
        verifyNoInteractions(grpc);
    }

    @Test
    void payloadFailureAfterClaimUsesExistingNotSentSettlement() throws Exception {
        RequestRoute item = item(1L);
        item.ctx().setGenerateInputPb(com.google.protobuf.ByteString.EMPTY);
        submit(List.of(item));
        releaseExecutor.countDown();
        awaitDispatchTasks();
        assertFalse(item.future().get(5, TimeUnit.SECONDS).isSuccess());
        registry.runtime.continuations().awaitIdle();
        assertOccupancy(0, 0);
        assertEquals(0, decode.routingView().engineCapacityUsed());
        assertEquals(0, decode.routingView().inflightHardKv());
        assertEquals(0, EndpointTestSupport.expectedReservedKv(decode.resourceSnapshot()));
        verifyNoInteractions(grpc);
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.MethodSource("invalidMemberSubsets")
    void everyInvalidMemberSubsetPreservesPayloadAndRealResourceLedgers(String cause, int invalidMask) throws Exception {
        var items = List.of(item(11L), item(12L), item(13L));
        submit(items);
        assertOccupancy(1, 3);
        var survivors = new java.util.ArrayList<RequestRoute>();
        for (int index = 0; index < items.size(); index++) {
            var item = items.get(index);
            if ((invalidMask & (1 << index)) == 0) { survivors.add(item); continue; }
            RequestContext context = item.ctx();
            switch (cause) {
                case "CLIENT_CANCELLED" -> registry.cancel(item.requestId(), 0L, CancelReason.CLIENT_CANCELLED);
                case "SHUTDOWN" -> registry.cancel(item.requestId(), 0L, CancelReason.SHUTDOWN);
                case "DEADLINE" -> ReflectionTestUtils.setField(context, "schedulingMetadata",
                        org.flexlb.dao.SchedulingMetadata.explicit(context.getPriority(), System.currentTimeMillis() - 1L));
                case "INACTIVITY" -> ReflectionTestUtils.setField(context, "lastWorkerStatusAtMs",
                        System.currentTimeMillis() - TIMEOUT_MS - 1L);
                default -> throw new AssertionError(cause);
            }
        }
        releaseExecutor.countDown();
        awaitDispatchTasks();
        registry.runtime.continuations().awaitIdle();
        for (int index = 0; index < items.size(); index++) {
            if ((invalidMask & (1 << index)) != 0) {
                assertFalse(items.get(index).future().get(5, TimeUnit.SECONDS).isSuccess());
            }
        }
        assertOccupancy(survivors.isEmpty() ? 0 : 1, survivors.size());
        assertEquals(survivors.size(), decode.routingView().engineCapacityUsed());
        assertEquals(survivors.size(), decode.routingView().inflightHardKv());
        assertEquals(2L * survivors.size(), EndpointTestSupport.expectedReservedKv(decode.resourceSnapshot()));
        if (survivors.isEmpty()) {
            verifyNoInteractions(grpc);
        } else {
            assertEquals(1, sent.size());
            assertEquals(survivors.stream().map(RequestRoute::requestId).toList(), sent.getFirst().getDpSlotsList().stream()
                    .flatMap(requestContext -> requestContext.getRequestsList().stream()).map(request -> request.getInput().getRequestId()).toList());
            var ack = EngineRpcService.EnqueueBatchResponsePB.newBuilder().setBatchId(201L);
            survivors.forEach(item -> ack.addSuccesses(EngineRpcService.EnqueueBatchSuccessPB.newBuilder().setRequestId(item.requestId())));
            reply.complete(ack.build());
            for (RequestRoute item : survivors) {
                assertTrue(item.future().get(5, TimeUnit.SECONDS).isSuccess());
                ledger.finish(201L, item).forEach(requestStatus -> RequestProtocolTestSupport.applyPrefillStatus(registry, prefill, RoleType.PREFILL, requestStatus));
                DeliverySettlementTestSupport.decodeStatus(decode, item.requestId(), true);
            }
        }
        registry.runtime.continuations().awaitIdle();
        for (int index = 0; index < items.size(); index++) {
            if ((invalidMask & (1 << index)) != 0) { registry.cancel(items.get(index).requestId(), 0L, CancelReason.CLIENT_CANCELLED); }
        }
        assertOccupancy(0, 0);
        assertEquals(0, decode.routingView().engineCapacityUsed());
        assertEquals(0, decode.routingView().inflightHardKv());
        assertEquals(0, EndpointTestSupport.expectedReservedKv(decode.resourceSnapshot()));
        dispatcher.tryPrepareSubmission().value().close();
    }

    static java.util.stream.Stream<org.junit.jupiter.params.provider.Arguments> invalidMemberSubsets() {
        return java.util.stream.Stream.of("CLIENT_CANCELLED", "SHUTDOWN", "DEADLINE", "INACTIVITY").flatMap(cause ->
                java.util.stream.IntStream.range(0, 8).mapToObj(mask -> org.junit.jupiter.params.provider.Arguments.of(cause, mask)));
    }

    private RequestRoute item(long id) {
        var context = RequestProtocolTestSupport.context(config, id);
        context.setGenerateInputPb(EngineRpcService.GenerateInputPB.newBuilder().setRequestId(id)
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder()).build().toByteString());
        var future = RequestProtocolTestSupport.register(registry, context);
        DecodeResources.ReservationHandle reservation;
        try (var pin = decode.tryPinGeneration()) {
            reservation = EndpointTestSupport.reserveUnqueuedDecode(decode, pin, id, 1L, 2L, 50);
        }
        assertNotNull(reservation);
        DeliverySettlementTestSupport.queueDecode(decode, reservation);
        ServerStatus status = new ServerStatus();
        status.setRole(RoleType.PREFILL);
        status.setServerIp("127.0.0.1");
        status.setHttpPort(8080);
        status.setGrpcPort(8090);
        context.setFuture(future);
        var item = org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(freezeInputs(context), new Response(), status, null,
                prefill, decode, reservation, System.currentTimeMillis());
        RequestProtocolTestSupport.bind(registry, new RequestProtocolTestSupport.Registered(item, future));
        ledger.enqueue(item);
        return item;
    }

    private PrefillState.WorkCapture commit(DeliveryTransaction transaction) {
        var lock = ledger.prefill.ownershipLock();
        lock.lock();
        try {
            return transaction.commitSelectionLocked(System.currentTimeMillis()).precedingWork();
        } finally {
            lock.unlock();
        }
    }

    private void submit(List<RequestRoute> items) {
        var evaluator = new FormulaPredictor("100 * batchSize");
        try (var transaction = strategy.prepare(items, evaluator, OptionalLong.empty())) {
            assertEquals(items, transaction.items());
            var preceding = commit(transaction).materialize();
            strategy.deliver(transaction, "queued-race", 0, preceding, evaluator);
        }
    }

    private void awaitDispatchTasks() throws Exception {
        ThreadPoolExecutor executor = (ThreadPoolExecutor) ReflectionTestUtils.getField(dispatcher, "dispatchExecutor");
        executor.submit(() -> { }).get(5, TimeUnit.SECONDS);
    }

    private void assertOccupancy(int batches, int requests) {
        assertEquals(batches, ledger.prefill.stats().batchCount());
        assertEquals(requests, ledger.prefill.stats().locallyOwnedRequests());
    }

    private static EngineRpcService.EnqueueBatchResponsePB ack(long id) {
        return EngineRpcService.EnqueueBatchResponsePB.newBuilder().setBatchId(201L)
                .addSuccesses(EngineRpcService.EnqueueBatchSuccessPB.newBuilder().setRequestId(id)).build();
    }
}
