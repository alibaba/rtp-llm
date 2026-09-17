package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryMetrics;
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
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.client.EngineGrpcClient;
import org.flexlb.service.monitor.BatchSchedulerReporter;
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

import static org.flexlb.balance.scheduler.RequestLifecycleTestSupport.await;
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

/** Real executor, Slot and endpoint ledgers; only the Engine transport is simulated. */
class QueuedBatchDeliveryTest {
    private static final long TIMEOUT_MS = TimeUnit.HOURS.toMillis(1);
    private final CountDownLatch releaseExecutor = new CountDownLatch(1);
    private final CompletableFuture<EngineRpcService.EnqueueBatchResponsePB> reply = new CompletableFuture<>();
    private final DeliverySettlementTestSupport ledger = new DeliverySettlementTestSupport();
    private FlexlbConfig config;
    private RequestRegistry registry;
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
        BatchSchedulerReporter reporter = mock(BatchSchedulerReporter.class);
        registry = new RequestRegistry(service, reporter, mock(RequestSchedulerReporter.class));
        grpc = mock(EngineGrpcClient.class);
        when(grpc.batchEnqueueAsync(anyString(), anyInt(), any())).thenAnswer(call -> {
            sent.add(call.getArgument(2));
            return reply;
        });
        dispatcher = new DefaultBatchDispatcher(grpc, service, null, 1, 1);
        strategy = new BatchDeliveryStrategy(dispatcher::tryPrepareSubmission, () -> 201L,
                registry, new DeliveryMetrics(reporter));
        prefill = mock(PrefillEndpoint.class);
        when(prefill.getIp()).thenReturn("127.0.0.1");
        when(prefill.getGrpcPort()).thenReturn(8090);
        when(prefill.reserveBatch(any(), anyLong(), anyInt())).thenAnswer(call ->
                ledger.reserveBatch(call.getArgument(0), call.getArgument(1), call.getArgument(2)));
        when(prefill.releaseCommittedItem(any())).thenAnswer(call -> {
            ScheduledRequest item = call.getArgument(0);
            assertFalse(Thread.holdsLock(registry.requestSlot(item.requestId())));
            return ledger.prefill.terminalizeCommittedItem(item);
        });
        when(prefill.expireCommittedItem(any())).thenAnswer(call ->
                ledger.prefill.terminalizeCommittedItem(call.getArgument(0)));
        doAnswer(call -> { prefill.releaseCommittedItem(call.getArgument(0)); return null; })
                .when(prefill).settleFailedRequest(any());
        decode = new DecodeEndpoint(WorkerStatus.createDiscovered(RoleType.DECODE, null,
                "127.0.0.1", 8080, 8081, null), new EndpointEventProjector(registry));
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
        if (registry.closeAdmissionAndAwaitMutations()) {
            registry.closeOutstandingAndTerminalize();
            registry.closeExpiration();
            registry.closePublisher();
        }
    }

    @ParameterizedTest
    @CsvSource({"CANCEL, false", "CANCEL, true", "TIMER, false", "TIMER, true",
            "LATE_DEADLINE, false", "LATE_DEADLINE, true", "LATE_INACTIVITY, false", "LATE_INACTIVITY, true"})
    void invalidQueuedMembersNeverCrossHandoffAndReleaseExactResources(String expiry, boolean all)
            throws Exception {
        ScheduledRequest first = item(1L);
        ScheduledRequest second = item(2L);
        submit(List.of(first, second));
        assertEquals(DeliveryClaimKind.NONE, registry.getRequestState(1L, 0L).deliveryClaimKind());
        assertEquals(DeliveryClaimKind.NONE, registry.getRequestState(2L, 0L).deliveryClaimKind());
        assertOccupancy(1, 2);
        for (ScheduledRequest item : all ? List.of(first, second) : List.of(first)) {
            RequestSlot slot = registry.requestSlot(item.requestId());
            switch (expiry) {
                case "CANCEL" -> registry.cancelRequest(item.requestId(), 0L, CancelReason.CLIENT_CANCELLED);
                case "TIMER" -> registry.expireInactiveRequest(slot, slot.createdAtMs() + TIMEOUT_MS);
                case "LATE_DEADLINE" -> ReflectionTestUtils.setField(item, "expiresAtMs", System.currentTimeMillis() - 1L);
                case "LATE_INACTIVITY" -> ReflectionTestUtils.setField(slot, "lastWorkerStatusAtMs",
                        System.currentTimeMillis() - TIMEOUT_MS - 1L);
                default -> throw new AssertionError(expiry);
            }
        }
        releaseExecutor.countDown();
        awaitDispatchTasks();
        assertFalse(first.future().get(5, TimeUnit.SECONDS).isSuccess());
        assertEquals(DeliveryClaimKind.NONE, registry.getRequestState(1L, 0L).deliveryClaimKind());
        int survivors = all ? 0 : 1;
        assertOccupancy(survivors, survivors);
        assertEquals(survivors, decode.routingView().engineCapacityUsed());
        assertEquals(survivors, decode.routingView().inflightHardKv());
        assertEquals(2L * survivors, decode.routingView().inflightExpectedKv());
        assertEquals(survivors, sent.size());
        if (all) {
            assertFalse(second.future().get(5, TimeUnit.SECONDS).isSuccess());
            verifyNoInteractions(grpc);
        } else {
            assertEquals(List.of(2L), sent.getFirst().getDpSlotsList().stream()
                    .flatMap(dp -> dp.getRequestsList().stream()).map(input -> input.getInput().getRequestId()).toList());
            assertEquals(DeliveryClaimKind.BATCH_ENQUEUE, registry.getRequestState(2L, 0L).deliveryClaimKind());
            reply.complete(ack(2L));
            assertTrue(second.future().get(5, TimeUnit.SECONDS).isSuccess());
            ledger.finish(201L, second).forEach(fact -> registry.processPrefillStatus(prefill, RoleType.PREFILL, fact));
            DeliverySettlementTestSupport.decodeStatus(decode, 2L, true);
        }
        // Repeated terminal events cannot subtract the other member or leak a batch permit.
        registry.cancelRequest(1L, 0L, CancelReason.CLIENT_CANCELLED);
        assertOccupancy(0, 0);
        assertEquals(0, decode.routingView().engineCapacityUsed());
        assertEquals(0, decode.routingView().inflightHardKv());
        assertEquals(0, decode.routingView().inflightExpectedKv());
        dispatcher.tryPrepareSubmission().value().close();
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void lateHandoffStartsOneBoundedObservationWindowWithoutFabricatingWorkerActivity(boolean ack)
            throws Exception {
        ScheduledRequest item = item(1L);
        RequestSlot slot = registry.requestSlot(1L);
        long lastStatus = System.currentTimeMillis() - TIMEOUT_MS + 10_000L;
        ReflectionTestUtils.setField(slot, "lastWorkerStatusAtMs", lastStatus);
        submit(List.of(item));
        releaseExecutor.countDown();
        awaitDispatchTasks();
        assertEquals(1, sent.size());
        long handoff = (long) ReflectionTestUtils.getField(slot, "batchEnqueueStartedAtMs");
        assertEquals(lastStatus, ReflectionTestUtils.getField(slot, "lastWorkerStatusAtMs"));
        if (ack) {
            reply.complete(ack(1L));
            assertTrue(item.future().get(5, TimeUnit.SECONDS).isSuccess());
        }
        InactivityDeadline originalTimer = (InactivityDeadline) ReflectionTestUtils.getField(slot, "inactivityDeadline");
        slot.onInactivityDeadline(originalTimer, lastStatus + TIMEOUT_MS);
        assertEquals(handoff + TIMEOUT_MS, slot.inactivityDeadlineAtMs().orElseThrow());
        assertOccupancy(1, 1);
        registry.expireInactiveRequest(slot, handoff + TIMEOUT_MS - 1);
        assertOccupancy(1, 1);
        registry.expireInactiveRequest(slot, handoff + TIMEOUT_MS);
        assertOccupancy(0, 0);
        assertEquals(0, decode.routingView().engineCapacityUsed());
        assertEquals(ack, item.future().get(5, TimeUnit.SECONDS).isSuccess());
    }

    @Test
    void blockedRpcDoesNotHoldSlotOrTransactionMonitor() throws Exception {
        ScheduledRequest item = item(1L);
        RequestSlot slot = registry.requestSlot(1L);
        CountDownLatch rpcEntered = new CountDownLatch(1);
        CountDownLatch releaseRpc = new CountDownLatch(1);
        when(grpc.batchEnqueueAsync(anyString(), anyInt(), any())).thenAnswer(call -> {
            assertFalse(Thread.holdsLock(slot));
            rpcEntered.countDown();
            await(releaseRpc);
            return reply;
        });
        try (var transaction = strategy.prepare(List.of(item), new FormulaPredictor("100"), OptionalLong.empty());
             var contender = java.util.concurrent.Executors.newSingleThreadExecutor()) {
            var preceding = transaction.commitUnderLock();
            transaction.handoff("blocked-rpc", 0, preceding);
            releaseExecutor.countDown();
            try {
                await(rpcEntered);
                contender.submit(() -> {
                    transaction.abort(new IllegalStateException("scheduler cleanup"));
                    transaction.close();
                    synchronized (slot) {
                        assertEquals(DeliveryClaimKind.BATCH_ENQUEUE, slot.snapshot().deliveryClaimKind());
                    }
                }).get(2, TimeUnit.SECONDS);
                assertOccupancy(1, 1);
            } finally {
                releaseRpc.countDown();
            }
        }
    }

    @Test
    void rejectedExecutorSubmissionClosesCommittedAdmissionExactlyOnce() throws Exception {
        ScheduledRequest item = item(1L);
        var rejection = new java.util.concurrent.RejectedExecutionException("executor rejected");
        BatchDeliveryStrategy.PreparedSubmission submission = mock(BatchDeliveryStrategy.PreparedSubmission.class);
        doThrow(rejection).when(submission).submit(any());
        var rejectingStrategy = new BatchDeliveryStrategy(
                () -> org.flexlb.balance.delivery.CapacityBoundary.Attempt.accepted(submission),
                () -> 201L, registry, new DeliveryMetrics(mock(BatchSchedulerReporter.class)));
        try (var transaction = rejectingStrategy.prepare(List.of(item), new FormulaPredictor("100"), OptionalLong.empty())) {
            var preceding = transaction.commitUnderLock();
            assertSame(rejection, assertThrows(java.util.concurrent.RejectedExecutionException.class,
                    () -> transaction.handoff("rejected", 0, preceding)));
            transaction.abort(rejection);
        }
        assertFalse(item.future().get(5, TimeUnit.SECONDS).isSuccess());
        verify(submission, times(1)).close();
        assertOccupancy(0, 0);
        assertEquals(0, decode.routingView().engineCapacityUsed());
        assertEquals(0, decode.routingView().inflightHardKv());
        assertEquals(0, decode.routingView().inflightExpectedKv());
        verifyNoInteractions(grpc);
    }

    @Test
    void payloadFailureAfterClaimUsesExistingNotSentSettlement() throws Exception {
        ScheduledRequest item = item(1L);
        item.ctx().setGenerateInputPb(com.google.protobuf.ByteString.EMPTY);
        submit(List.of(item));
        releaseExecutor.countDown();
        awaitDispatchTasks();
        assertFalse(item.future().get(5, TimeUnit.SECONDS).isSuccess());
        assertOccupancy(0, 0);
        assertEquals(0, decode.routingView().engineCapacityUsed());
        assertEquals(0, decode.routingView().inflightHardKv());
        assertEquals(0, decode.routingView().inflightExpectedKv());
        verifyNoInteractions(grpc);
    }

    private ScheduledRequest item(long id) {
        var context = RequestLifecycleTestSupport.context(config, id);
        context.setGenerateInputPb(EngineRpcService.GenerateInputPB.newBuilder().setRequestId(id)
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder()).build().toByteString());
        var future = registry.register(context);
        DecodeEndpoint.ReservationHandle reservation;
        try (var pin = decode.tryPinGeneration()) {
            reservation = decode.reserveUnqueued(pin, id, 1L, 2L, 50);
        }
        assertNotNull(reservation);
        DeliverySettlementTestSupport.queueDecode(decode, reservation);
        ServerStatus status = new ServerStatus();
        status.setRole(RoleType.PREFILL);
        status.setServerIp("127.0.0.1");
        status.setHttpPort(8080);
        status.setGrpcPort(8090);
        var item = new ScheduledRequest(context, future, new Response(), status, null,
                prefill, decode, reservation, System.currentTimeMillis());
        RequestLifecycleTestSupport.bind(registry, new RequestLifecycleTestSupport.Registered(item, future));
        ledger.enqueue(item);
        return item;
    }

    private void submit(List<ScheduledRequest> items) {
        try (var transaction = strategy.prepare(items, new FormulaPredictor("100 * batchSize"), OptionalLong.empty())) {
            assertEquals(items, transaction.items());
            var preceding = transaction.commitUnderLock();
            transaction.handoff("queued-race", 0, preceding);
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
