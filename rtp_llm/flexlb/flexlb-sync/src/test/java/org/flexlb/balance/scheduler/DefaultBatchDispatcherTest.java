package org.flexlb.balance.scheduler;

import com.google.protobuf.ByteString;
import com.google.protobuf.Int64Value;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class DefaultBatchDispatcherTest {

    private ConfigService configService;
    private EngineGrpcClient grpcClient;
    private BatchSchedulerReporter reporter;
    private FlexlbConfig config;
    private DefaultBatchDispatcher dispatcher;
    private TestCallback callback;

    @BeforeEach
    void setUp() {
        configService = mock(ConfigService.class);
        grpcClient = mock(EngineGrpcClient.class);
        reporter = mock(BatchSchedulerReporter.class);
        config = new FlexlbConfig();
        config.setFlexlbBatchDispatchPoolSize(2);
        config.setFlexlbBatchDispatchQueueSize(10);
        config.setFlexlbBatchEnqueueDeadlineMs(5000);
        when(configService.loadBalanceConfig()).thenReturn(config);

        dispatcher = new DefaultBatchDispatcher(grpcClient, configService, null);
        callback = new TestCallback();
    }

    @Test
    void dispatchSendsItemsToGrpcAndReceivesAck() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        BatchItem item = createBatchItem(1L, 500, 200, prefillEp);

        EngineRpcService.EnqueueBatchResponsePB response = ackResponse(1L, List.of(1L));
        when(grpcClient.batchEnqueue(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenReturn(response);

        dispatcher.dispatch(List.of(item), prefillEp, 1L, 100, "test_reason", callback);

        assertTrue(callback.successLatch.await(5, TimeUnit.SECONDS), "onSuccess should be called");
        assertEquals(1, callback.successCount.get());
        assertEquals(0, callback.failureCount.get());
    }

    @Test
    void dispatchHandlesGrpcError() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        BatchItem item = createBatchItem(1L, 500, 200, prefillEp);

        when(grpcClient.batchEnqueue(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenThrow(new RuntimeException("gRPC connection refused"));

        dispatcher.dispatch(List.of(item), prefillEp, 1L, 100, "test_reason", callback);

        assertTrue(callback.failureLatch.await(5, TimeUnit.SECONDS), "onFailure should be called");
        assertEquals(1, callback.failureCount.get());
        assertEquals(0, callback.successCount.get());
    }

    @Test
    void dispatchHandlesNullGrpcResponse() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        BatchItem item = createBatchItem(1L, 500, 200, prefillEp);

        when(grpcClient.batchEnqueue(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenReturn(null);

        dispatcher.dispatch(List.of(item), prefillEp, 1L, 100, "test_reason", callback);

        assertTrue(callback.failureLatch.await(5, TimeUnit.SECONDS));
        assertEquals(1, callback.failureCount.get());
    }

    @Test
    void dispatchRejectsAckWithDifferentBatchId() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        BatchItem item = createBatchItem(8L, 500, 200, prefillEp);
        when(grpcClient.batchEnqueue(anyString(), anyInt(), any(), anyLong())).thenReturn(
                EngineRpcService.EnqueueBatchResponsePB.newBuilder()
                        .setBatchId(87L)
                        .addSuccesses(EngineRpcService.EnqueueBatchSuccessPB.newBuilder().setRequestId(8L))
                        .build());

        dispatcher.dispatch(List.of(item), prefillEp, 88L,
                100, "batch_id_mismatch", callback);

        assertTrue(callback.failureLatch.await(5, TimeUnit.SECONDS));
        assertEquals(0, callback.successCount.get());
        assertEquals(1, callback.failureCount.get());
    }

    @Test
    void dispatchKeepsCommittedItemsEvenIfCancellationRacesWithSend() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        BatchItem active = createBatchItem(1L, 500, 200, prefillEp);
        BatchItem cancelled = createBatchItem(2L, 300, 100, prefillEp);
        cancelled.ctx().cancel(); // mark as cancelled

        AtomicReference<EngineRpcService.EnqueueBatchRequestPB> captured = new AtomicReference<>();
        when(grpcClient.batchEnqueue(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenAnswer(inv -> {
                    captured.set(inv.getArgument(2));
                    return ackResponse(1L, List.of(1L, 2L));
                });

        dispatcher.dispatch(List.of(active, cancelled), prefillEp, 1L, 100, "test", callback);

        assertTrue(callback.successLatch.await(5, TimeUnit.SECONDS));
        EngineRpcService.EnqueueBatchRequestPB sent = captured.get();
        assertNotNull(sent);
        // Scheduler committed both requests before handing them to the dispatcher.
        // Dropping one here could let Cancel arrive before Enqueue and be lost.
        long sentCount = sent.getDpSlotsList().stream()
                .mapToLong(slot -> slot.getRequestsCount())
                .sum();
        assertEquals(2, sentCount);
    }

    @Test
    void dispatchHandlesRejectedExecutionAfterShutdown() {
        dispatcher.shutdown();

        PrefillEndpoint prefillEp = createPrefillEndpoint();
        BatchItem item = createBatchItem(1L, 500, 200, prefillEp);

        dispatcher.dispatch(List.of(item), prefillEp, 1L, 100, "test", callback);

        // Should fail synchronously when executor is shut down
        assertEquals(1, callback.failureCount.get());
    }

    @Test
    void dispatchHandlesResponseWithErrors() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        BatchItem item = createBatchItem(1L, 500, 200, prefillEp);

        EngineRpcService.EnqueueBatchResponsePB response =
                EngineRpcService.EnqueueBatchResponsePB.newBuilder()
                        .setBatchId(1L)
                        .addErrors(EngineRpcService.EnqueueBatchErrorPB.newBuilder()
                                .setRequestId(1L)
                                .setErrorInfo(EngineRpcService.ErrorDetailsPB.newBuilder()
                                        .setErrorCode(500)
                                        .setErrorMessage("engine busy")
                                        .build())
                                .build())
                        .build();
        when(grpcClient.batchEnqueue(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenReturn(response);

        dispatcher.dispatch(List.of(item), prefillEp, 1L, 100, "test", callback);

        assertTrue(callback.failureLatch.await(5, TimeUnit.SECONDS));
        assertEquals(1, callback.failureCount.get());
    }

    @Test
    void dispatchHandlesMissingAck() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        BatchItem item = createBatchItem(1L, 500, 200, prefillEp);

        EngineRpcService.EnqueueBatchResponsePB response =
                EngineRpcService.EnqueueBatchResponsePB.newBuilder()
                        .setBatchId(1L)
                        .build(); // no success, no error
        when(grpcClient.batchEnqueue(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenReturn(response);

        dispatcher.dispatch(List.of(item), prefillEp, 1L, 100, "test", callback);

        assertTrue(callback.failureLatch.await(5, TimeUnit.SECONDS));
        assertEquals(1, callback.failureCount.get());
    }

    @Test
    void shutdownDrainsExecutor() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();

        // Submit tasks so executor has work in flight
        CountDownLatch started = new CountDownLatch(1);
        when(grpcClient.batchEnqueue(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenAnswer(inv -> {
                    started.countDown();
                    return ackResponse(1L, List.of(1L));
                });

        BatchItem item = createBatchItem(1L, 500, 200, prefillEp);
        dispatcher.dispatch(List.of(item), prefillEp, 1L, 100, "test", callback);

        // Wait for at least one task to start, then shutdown
        assertTrue(started.await(5, TimeUnit.SECONDS));
        dispatcher.shutdown();

        // Post-shutdown dispatch should be rejected immediately
        int failuresBefore = callback.failureCount.get();
        BatchItem extra = createBatchItem(99L, 500, 200, prefillEp);
        dispatcher.dispatch(List.of(extra), prefillEp, 99L, 100, "test", callback);
        assertEquals(failuresBefore + 1, callback.failureCount.get(), "Post-shutdown dispatch should add exactly 1 failure");
    }

    // ---- helpers ----

    private PrefillEndpoint createPrefillEndpoint() {
        WorkerStatus status = new WorkerStatus();
        status.setIp("127.0.0.1");
        status.setPort(8080);
        status.setGrpcPort(8090);
        status.setRole(RoleType.PREFILL);
        FlexlbConfig epConfig = new FlexlbConfig();
        epConfig.setFlexlbBatchQueueMaxSize(100);
        epConfig.setFlexlbBatchFixedWaitMs(300);
        return new PrefillEndpoint(status, epConfig, noopHandler(), reporter);
    }

    private static BatchDecisionHandler noopHandler() {
        return new BatchDecisionHandler() {
            @Override public void onExpired(BatchItem head) {}
            @Override public void onUrgent(BatchItem head, DispatchMeta meta) {}
            @Override public void onBatchReady(List<BatchItem> items, DispatchMeta meta) {}
            @Override public void onOfferFailure(BatchItem item, Throwable error) {}
        };
    }

    private BatchItem createBatchItem(long requestId, long seqLen, long hitCacheLen, PrefillEndpoint prefillEp) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);

        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);

        // Provide a valid GenerateInputPB bytes (minimum: requestId + empty config)
        EngineRpcService.GenerateInputPB input = EngineRpcService.GenerateInputPB.newBuilder()
                .setRequestId(requestId)
                .setGroupId(Int64Value.of(1L))
                .setGroupSize(1)
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder().build())
                .build();
        ctx.setGenerateInputPbBytes(input.toByteArray());

        ServerStatus prefill = new ServerStatus();
        prefill.setRole(RoleType.PREFILL);
        prefill.setServerIp("127.0.0.1");
        prefill.setHttpPort(8080);
        prefill.setGrpcPort(8090);
        prefill.setDpRank(0L);
        DebugInfo debugInfo = new DebugInfo();
        debugInfo.setHitCacheLen(hitCacheLen);
        prefill.setDebugInfo(debugInfo);

        return new BatchItem(ctx, new CompletableFuture<>(), null, prefill, null, prefillEp, null, 0, System.currentTimeMillis());
    }

    private EngineRpcService.EnqueueBatchResponsePB ackResponse(long batchId, List<Long> successIds) {
        EngineRpcService.EnqueueBatchResponsePB.Builder builder =
                EngineRpcService.EnqueueBatchResponsePB.newBuilder().setBatchId(batchId);
        for (long id : successIds) {
            builder.addSuccesses(EngineRpcService.EnqueueBatchSuccessPB.newBuilder()
                    .setRequestId(id)
                    .build());
        }
        return builder.build();
    }

    // ---- Test callback ----

    private static class TestCallback implements DispatchCallback {
        final AtomicInteger successCount = new AtomicInteger(0);
        final AtomicInteger failureCount = new AtomicInteger(0);
        final CountDownLatch successLatch = new CountDownLatch(1);
        final CountDownLatch failureLatch = new CountDownLatch(1);

        @Override
        public void onSuccess(BatchItem item, long batchId) {
            successCount.incrementAndGet();
            successLatch.countDown();
        }

        @Override
        public void onFailure(BatchItem item, Throwable error) {
            failureCount.incrementAndGet();
            failureLatch.countDown();
        }
    }
}
