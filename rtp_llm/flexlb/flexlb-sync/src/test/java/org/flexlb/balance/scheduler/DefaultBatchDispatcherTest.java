package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
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

    @AfterEach
    void tearDown() {
        dispatcher.shutdown();
    }

    @Test
    void dispatchSendsItemsToGrpcAndReceivesAck() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        BatchItem item = createBatchItem(1L, 500, 200, prefillEp);

        EngineRpcService.EnqueueBatchResponsePB response = ackResponse(1L, List.of(1L));
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenReturn(CompletableFuture.completedFuture(response));

        dispatcher.dispatch(List.of(item), prefillEp, 1L, 100, "test_reason", callback);

        assertTrue(callback.successLatch.await(5, TimeUnit.SECONDS), "onSuccess should be called");
        assertEquals(1, callback.successCount.get());
        assertEquals(0, callback.failureCount.get());
    }

    @Test
    void dispatchHandlesGrpcError() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        BatchItem item = createBatchItem(1L, 500, 200, prefillEp);

        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenReturn(CompletableFuture.failedFuture(new RuntimeException("gRPC connection refused")));

        dispatcher.dispatch(List.of(item), prefillEp, 1L, 100, "test_reason", callback);

        assertTrue(callback.failureLatch.await(5, TimeUnit.SECONDS), "onFailure should be called");
        assertEquals(1, callback.failureCount.get());
        assertEquals(0, callback.successCount.get());
    }

    @Test
    void dispatchHandlesNullGrpcResponse() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        BatchItem item = createBatchItem(1L, 500, 200, prefillEp);

        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenReturn(CompletableFuture.completedFuture(null));

        dispatcher.dispatch(List.of(item), prefillEp, 1L, 100, "test_reason", callback);

        assertTrue(callback.failureLatch.await(5, TimeUnit.SECONDS));
        assertEquals(1, callback.failureCount.get());
    }

    @Test
    void dispatchRejectsAckWithDifferentBatchId() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        BatchItem item = createBatchItem(8L, 500, 200, prefillEp);
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(), anyLong())).thenReturn(
                CompletableFuture.completedFuture(EngineRpcService.EnqueueBatchResponsePB.newBuilder()
                        .setBatchId(87L)
                        .addSuccesses(EngineRpcService.EnqueueBatchSuccessPB.newBuilder().setRequestId(8L))
                        .build()));

        dispatcher.dispatch(List.of(item), prefillEp, 88L,
                100, "batch_id_mismatch", callback);

        assertTrue(callback.failureLatch.await(5, TimeUnit.SECONDS));
        assertEquals(0, callback.successCount.get());
        assertEquals(1, callback.failureCount.get());
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
    void rpcCompletionFallsBackInlineWhenExecutorQueueIsFull() throws Exception {
        recreateDispatcher(1, 1);
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        CompletableFuture<EngineRpcService.EnqueueBatchResponsePB> firstRpc =
                new CompletableFuture<>();
        CountDownLatch firstStarted = new CountDownLatch(1);
        CountDownLatch secondStarted = new CountDownLatch(1);
        CountDownLatch releaseSecond = new CountDownLatch(1);
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(), anyLong()))
                .thenAnswer(invocation -> {
                    EngineRpcService.EnqueueBatchRequestPB request = invocation.getArgument(2);
                    if (request.getBatchId() == 101L) {
                        firstStarted.countDown();
                        return firstRpc;
                    }
                    if (request.getBatchId() == 102L) {
                        secondStarted.countDown();
                        if (!releaseSecond.await(3, TimeUnit.SECONDS)) {
                            throw new IllegalStateException("second dispatch was not released");
                        }
                    }
                    return CompletableFuture.completedFuture(
                            ackResponse(request.getBatchId(), requestIds(request)));
                });

        TestCallback firstCallback = new TestCallback();
        TestCallback secondCallback = new TestCallback();
        TestCallback thirdCallback = new TestCallback();
        BatchItem first = createBatchItem(101L, 500, 0, prefillEp);
        BatchItem second = createBatchItem(102L, 500, 0, prefillEp);
        BatchItem third = createBatchItem(103L, 500, 0, prefillEp);

        try {
            dispatcher.dispatch(List.of(first), prefillEp, 101L, 1, "first", firstCallback);
            assertTrue(firstStarted.await(2, TimeUnit.SECONDS));
            dispatcher.dispatch(List.of(second), prefillEp, 102L, 1, "second", secondCallback);
            assertTrue(secondStarted.await(2, TimeUnit.SECONDS));
            dispatcher.dispatch(List.of(third), prefillEp, 103L, 1, "third", thirdCallback);

            firstRpc.complete(ackResponse(101L, List.of(101L)));

            assertTrue(firstCallback.terminalLatch.await(2, TimeUnit.SECONDS),
                    "completion must not be dropped when the executor queue is full");
            assertEquals(1, firstCallback.successCount.get());
            assertEquals(0, firstCallback.failureCount.get());
        } finally {
            releaseSecond.countDown();
        }
        assertTrue(secondCallback.terminalLatch.await(2, TimeUnit.SECONDS));
        assertTrue(thirdCallback.terminalLatch.await(2, TimeUnit.SECONDS));
    }

    @Test
    void rpcCompletionAfterShutdownTerminatesCallbackExactlyOnce() throws Exception {
        recreateDispatcher(1, 1);
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        CompletableFuture<EngineRpcService.EnqueueBatchResponsePB> rpc =
                new CompletableFuture<>();
        CountDownLatch started = new CountDownLatch(1);
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(), anyLong()))
                .thenAnswer(invocation -> {
                    started.countDown();
                    return rpc;
                });
        TestCallback completion = new TestCallback();
        BatchItem item = createBatchItem(111L, 500, 0, prefillEp);

        dispatcher.dispatch(List.of(item), prefillEp, 111L, 1, "shutdown", completion);
        assertTrue(started.await(2, TimeUnit.SECONDS));
        dispatcher.shutdown();
        rpc.complete(ackResponse(111L, List.of(111L)));

        assertTrue(completion.terminalLatch.await(2, TimeUnit.SECONDS));
        assertEquals(1, completion.successCount.get() + completion.failureCount.get());
        assertEquals(1, completion.successCount.get(),
                "an RPC already sent before shutdown must converge from its real completion");
        verify(prefillEp, times(0)).releaseBatch(111L);
    }

    @Test
    void shutdownFailsDispatchStillQueuedInExecutor() throws Exception {
        recreateDispatcher(1, 1);
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        CountDownLatch firstStarted = new CountDownLatch(1);
        CountDownLatch releaseFirst = new CountDownLatch(1);
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(), anyLong()))
                .thenAnswer(invocation -> {
                    firstStarted.countDown();
                    try {
                        releaseFirst.await(3, TimeUnit.SECONDS);
                    } catch (InterruptedException interrupted) {
                        Thread.currentThread().interrupt();
                        throw new CancellationException("dispatch interrupted by shutdown");
                    }
                    EngineRpcService.EnqueueBatchRequestPB request = invocation.getArgument(2);
                    return CompletableFuture.completedFuture(
                            ackResponse(request.getBatchId(), requestIds(request)));
                });
        TestCallback running = new TestCallback();
        TestCallback queued = new TestCallback();

        dispatcher.dispatch(List.of(createBatchItem(121L, 500, 0, prefillEp)),
                prefillEp, 121L, 1, "running", running);
        assertTrue(firstStarted.await(2, TimeUnit.SECONDS));
        dispatcher.dispatch(List.of(createBatchItem(122L, 500, 0, prefillEp)),
                prefillEp, 122L, 1, "queued", queued);

        CountDownLatch shutdownStarted = new CountDownLatch(1);
        CompletableFuture<Void> shutdown = CompletableFuture.runAsync(() -> {
            shutdownStarted.countDown();
            dispatcher.shutdown();
        });
        assertTrue(shutdownStarted.await(2, TimeUnit.SECONDS));
        releaseFirst.countDown();
        shutdown.get(2, TimeUnit.SECONDS);

        assertTrue(running.terminalLatch.await(2, TimeUnit.SECONDS));
        assertTrue(queued.terminalLatch.await(2, TimeUnit.SECONDS),
                "shutdown must fail work removed from the executor queue");
        assertEquals(1, running.successCount.get(),
                "shutdown must not fail an RPC after its send side effect started");
        assertEquals(1, queued.failureCount.get());
        assertEquals(1, running.successCount.get() + running.failureCount.get());
        assertEquals(1, queued.successCount.get() + queued.failureCount.get());
        verify(prefillEp, times(0)).releaseBatch(121L);
        verify(prefillEp, times(1)).releaseBatch(122L);
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
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenReturn(CompletableFuture.completedFuture(response));

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
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenReturn(CompletableFuture.completedFuture(response));

        dispatcher.dispatch(List.of(item), prefillEp, 1L, 100, "test", callback);

        assertTrue(callback.failureLatch.await(5, TimeUnit.SECONDS));
        assertEquals(1, callback.failureCount.get());
    }

    @Test
    void shutdownDrainsExecutor() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();

        // Submit tasks so executor has work in flight
        CountDownLatch started = new CountDownLatch(1);
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenAnswer(inv -> {
                    started.countDown();
                    return CompletableFuture.completedFuture(ackResponse(1L, List.of(1L)));
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
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        when(endpoint.getHttpPort()).thenReturn(8080);
        when(endpoint.getGrpcPort()).thenReturn(8090);
        return endpoint;
    }

    private void recreateDispatcher(int poolSize, int queueSize) {
        dispatcher.shutdown();
        config.setFlexlbBatchDispatchPoolSize(poolSize);
        config.setFlexlbBatchDispatchQueueSize(queueSize);
        dispatcher = new DefaultBatchDispatcher(grpcClient, configService, null);
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

        Response routeResponse = new Response();
        routeResponse.setSuccess(true);
        routeResponse.setServerStatus(List.of(prefill));
        return new BatchItem(ctx, new CompletableFuture<>(), routeResponse,
                hitCacheLen, prefillEp, System.currentTimeMillis());
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

    private static List<Long> requestIds(EngineRpcService.EnqueueBatchRequestPB request) {
        return request.getDpSlotsList().stream()
                .flatMap(slot -> slot.getRequestsList().stream())
                .map(EngineRpcService.EnqueueBatchExternalInputPB::getInput)
                .map(EngineRpcService.GenerateInputPB::getRequestId)
                .toList();
    }

    // ---- Test callback ----

    private static class TestCallback implements DispatchCallback {
        final AtomicInteger successCount = new AtomicInteger(0);
        final AtomicInteger failureCount = new AtomicInteger(0);
        final CountDownLatch successLatch = new CountDownLatch(1);
        final CountDownLatch failureLatch = new CountDownLatch(1);
        final CountDownLatch terminalLatch = new CountDownLatch(1);

        @Override
        public void onSuccess(BatchItem item, long batchId) {
            successCount.incrementAndGet();
            successLatch.countDown();
            terminalLatch.countDown();
        }

        @Override
        public void onFailure(BatchItem item, Throwable error) {
            failureCount.incrementAndGet();
            failureLatch.countDown();
            terminalLatch.countDown();
        }
    }
}
