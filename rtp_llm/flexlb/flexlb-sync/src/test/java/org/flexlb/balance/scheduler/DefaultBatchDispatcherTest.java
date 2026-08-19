package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.RoleTypeProtoConverter;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.doThrow;
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

        assertTrue(callback.uncertainLatch.await(5, TimeUnit.SECONDS),
                "post-send transport error must be reconciled");
        assertEquals(1, callback.uncertainCount.get());
        assertEquals(0, callback.failureCount.get());
        assertEquals(0, callback.successCount.get());
    }

    @Test
    void dispatchHandlesNullGrpcResponse() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        BatchItem item = createBatchItem(1L, 500, 200, prefillEp);

        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenReturn(CompletableFuture.completedFuture(null));

        dispatcher.dispatch(List.of(item), prefillEp, 1L, 100, "test_reason", callback);

        assertTrue(callback.uncertainLatch.await(5, TimeUnit.SECONDS));
        assertEquals(1, callback.uncertainCount.get());
    }

    @Test
    void dispatchHandlesNullGrpcFutureAsUncertain() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        BatchItem item = createBatchItem(1L, 500, 200, prefillEp);
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(), anyLong()))
                .thenReturn(null);

        dispatcher.dispatch(List.of(item), prefillEp, 1L, 100, "test", callback);

        assertTrue(callback.uncertainLatch.await(5, TimeUnit.SECONDS));
        assertEquals(1, callback.uncertainCount.get());
        assertEquals(0, callback.failureCount.get());
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

        assertTrue(callback.uncertainLatch.await(5, TimeUnit.SECONDS));
        assertEquals(0, callback.successCount.get());
        assertEquals(1, callback.uncertainCount.get());
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
    void dispatch_should_fail_before_send_when_endpoint_generation_is_retired() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        doThrow(new WorkerEndpoint.EndpointRetiredException("127.0.0.1:8080"))
                .when(prefillEp).initiateGenerationDispatch(any(), any());
        BatchItem item = createBatchItem(1L, 500, 200, prefillEp);

        dispatcher.dispatch(List.of(item), prefillEp, 1L, 100, "retired", callback);

        assertTrue(callback.failureLatch.await(5, TimeUnit.SECONDS));
        assertEquals(1, callback.failureCount.get());
        assertEquals(0, callback.uncertainCount.get());
        verify(prefillEp).releaseBatch(1L);
        verify(grpcClient, never()).batchEnqueueAsync(anyString(), anyInt(), any(), anyLong());
    }

    @Test
    void dispatch_should_fail_before_send_when_decode_generation_is_missing() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        BatchItem base = createBatchItem(1L, 500, 200, prefillEp);
        ServerStatus decode = new ServerStatus();
        decode.setRole(RoleType.DECODE);
        decode.setServerIp("127.0.0.2");
        decode.setHttpPort(8081);
        decode.setGrpcPort(8091);
        decode.setRequestId(1L);
        BatchItem missingDecodeGeneration = new BatchItem(
                base.ctx(), base.future(), base.routeResponse(), base.prefill(), decode,
                prefillEp, null, base.enqueuedAtMs());

        dispatcher.dispatch(List.of(missingDecodeGeneration), prefillEp,
                1L, 100, "missing_decode_generation", callback);

        assertTrue(callback.failureLatch.await(5, TimeUnit.SECONDS));
        assertEquals(1, callback.failureCount.get());
        assertEquals(0, callback.uncertainCount.get());
        verify(prefillEp).releaseBatch(1L);
        verify(grpcClient, never()).batchEnqueueAsync(anyString(), anyInt(), any(), anyLong());
    }

    @Test
    void queued_dispatch_should_not_invoke_grpc_after_endpoint_retirement() throws Exception {
        dispatcher.shutdown();
        config.setFlexlbBatchDispatchPoolSize(1);
        dispatcher = new DefaultBatchDispatcher(grpcClient, configService, null);

        PrefillEndpoint blocker = createPrefillEndpoint();
        PrefillEndpoint retiring = createRealPrefillEndpoint("127.0.0.2", 8081, 8091);
        TestCallback blockerCallback = new TestCallback();
        TestCallback retiringCallback = new TestCallback();
        CountDownLatch invocationEntered = new CountDownLatch(1);
        CountDownLatch releaseInvocation = new CountDownLatch(1);
        AtomicInteger rpcInvocations = new AtomicInteger();
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(), anyLong()))
                .thenAnswer(ignored -> {
                    rpcInvocations.incrementAndGet();
                    invocationEntered.countDown();
                    if (!releaseInvocation.await(5, TimeUnit.SECONDS)) {
                        throw new IllegalStateException("test dispatch release timed out");
                    }
                    return CompletableFuture.completedFuture(ackResponse(11L, List.of(1L)));
                });

        dispatcher.dispatch(List.of(createBatchItem(1L, 500, 200, blocker)),
                blocker, 11L, 100, "block_dispatch_executor", blockerCallback);
        assertTrue(invocationEntered.await(5, TimeUnit.SECONDS));

        dispatcher.dispatch(List.of(createBatchItem(2L, 500, 200, retiring)),
                retiring, 22L, 100, "queued_then_retired", retiringCallback);
        retiring.close();
        releaseInvocation.countDown();

        assertTrue(retiringCallback.failureLatch.await(5, TimeUnit.SECONDS));
        assertEquals(1, retiringCallback.failureCount.get());
        assertEquals(0, retiringCallback.uncertainCount.get());
        assertEquals(1, rpcInvocations.get(), "retired queued batch must never invoke gRPC");
    }

    @Test
    void executorRejectionIsolatesFailureCallbacksForEveryItem() {
        dispatcher.shutdown();
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        BatchItem first = createBatchItem(1L, 500, 200, prefillEp);
        BatchItem second = createBatchItem(2L, 500, 200, prefillEp);
        AtomicInteger attempts = new AtomicInteger();
        DispatchCallback throwingCallback = new DispatchCallback() {
            @Override
            public void onSuccess(BatchItem item, long batchId) {
            }

            @Override
            public void onFailure(BatchItem item, Throwable error) {
                if (attempts.incrementAndGet() == 1) {
                    throw new IllegalStateException("first callback failed");
                }
            }
        };

        dispatcher.dispatch(List.of(first, second), prefillEp,
                1L, 100, "executor_rejected", throwingCallback);

        assertEquals(2, attempts.get(), "one broken callback must not suppress later items");
        verify(prefillEp, times(1)).releaseBatch(1L);
    }

    @Test
    void unexpectedPreSendFailureIsDefiniteAndIsolatesCallbacks() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        BatchItem first = createBatchItem(1L, 500, 200, prefillEp);
        BatchItem second = createBatchItem(2L, 500, 200, prefillEp);
        when(configService.loadBalanceConfig())
                .thenThrow(new IllegalStateException("config unavailable before send"));
        CountDownLatch attempted = new CountDownLatch(2);
        AtomicInteger failures = new AtomicInteger();
        AtomicInteger uncertain = new AtomicInteger();
        DispatchCallback throwingCallback = new DispatchCallback() {
            @Override
            public void onSuccess(BatchItem item, long batchId) {
            }

            @Override
            public void onFailure(BatchItem item, Throwable error) {
                failures.incrementAndGet();
                attempted.countDown();
                if (item.requestId() == 1L) {
                    throw new IllegalStateException("first callback failed");
                }
            }

            @Override
            public void onDispatchUncertain(BatchItem item, long batchId, Throwable error) {
                uncertain.incrementAndGet();
            }
        };

        dispatcher.dispatch(List.of(first, second), prefillEp,
                2L, 100, "pre_send_failure", throwingCallback);

        assertTrue(attempted.await(5, TimeUnit.SECONDS));
        assertEquals(2, failures.get());
        assertEquals(0, uncertain.get());
        verify(prefillEp, times(1)).releaseBatch(2L);
        verify(grpcClient, never()).batchEnqueueAsync(anyString(), anyInt(), any(), anyLong());
    }

    @Test
    void synchronousRpcInvocationFailureIsUncertainAndIsolatesCallbacks() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        BatchItem first = createBatchItem(1L, 500, 200, prefillEp);
        BatchItem second = createBatchItem(2L, 500, 200, prefillEp);
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(), anyLong()))
                .thenThrow(new IllegalStateException("client threw after invocation began"));
        CountDownLatch attempted = new CountDownLatch(2);
        AtomicInteger failures = new AtomicInteger();
        AtomicInteger uncertain = new AtomicInteger();
        DispatchCallback throwingCallback = new DispatchCallback() {
            @Override
            public void onSuccess(BatchItem item, long batchId) {
            }

            @Override
            public void onFailure(BatchItem item, Throwable error) {
                failures.incrementAndGet();
            }

            @Override
            public void onDispatchUncertain(BatchItem item, long batchId, Throwable error) {
                uncertain.incrementAndGet();
                attempted.countDown();
                if (item.requestId() == 1L) {
                    throw new IllegalStateException("first callback failed");
                }
            }
        };

        dispatcher.dispatch(List.of(first, second), prefillEp,
                3L, 100, "post_boundary_throw", throwingCallback);

        assertTrue(attempted.await(5, TimeUnit.SECONDS));
        assertEquals(0, failures.get());
        assertEquals(2, uncertain.get());
        verify(prefillEp, never()).releaseBatch(3L);
    }

    @Test
    void completionExecutorRejectionAfterRpcInvocationIsUncertain() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        BatchItem item = createBatchItem(1L, 500, 200, prefillEp);
        CompletableFuture<EngineRpcService.EnqueueBatchResponsePB> rpcFuture = new CompletableFuture<>();
        CountDownLatch invoked = new CountDownLatch(1);
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(), anyLong()))
                .thenAnswer(invocation -> {
                    invoked.countDown();
                    return rpcFuture;
                });

        dispatcher.dispatch(List.of(item), prefillEp,
                4L, 100, "completion_rejected", callback);
        assertTrue(invoked.await(5, TimeUnit.SECONDS));
        dispatcher.shutdown();
        rpcFuture.complete(ackResponse(4L, List.of(1L)));

        assertTrue(callback.uncertainLatch.await(5, TimeUnit.SECONDS));
        assertEquals(0, callback.successCount.get());
        assertEquals(0, callback.failureCount.get());
        assertEquals(1, callback.uncertainCount.get());
        verify(prefillEp, never()).releaseBatch(4L);
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
        assertTrue(callback.lastError instanceof DefaultBatchDispatcher.EngineRejectedException);
        assertEquals(500,
                ((DefaultBatchDispatcher.EngineRejectedException) callback.lastError).errorCode());
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

        assertTrue(callback.uncertainLatch.await(5, TimeUnit.SECONDS));
        assertEquals(1, callback.uncertainCount.get());
    }

    @Test
    void responseCallbackFailureIsIsolatedAndNeverReclassifiesOtherItemsAsUncertain()
            throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        BatchItem succeeded = createBatchItem(11L, 500, 200, prefillEp);
        BatchItem rejected = createBatchItem(12L, 500, 200, prefillEp);
        EngineRpcService.EnqueueBatchResponsePB response =
                EngineRpcService.EnqueueBatchResponsePB.newBuilder()
                        .setBatchId(91L)
                        .addSuccesses(EngineRpcService.EnqueueBatchSuccessPB.newBuilder()
                                .setRequestId(11L).build())
                        .addErrors(EngineRpcService.EnqueueBatchErrorPB.newBuilder()
                                .setRequestId(12L)
                                .setErrorInfo(EngineRpcService.ErrorDetailsPB.newBuilder()
                                        .setErrorCode(500L)
                                        .setErrorMessage("rejected")
                                        .build())
                                .build())
                        .build();
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(), anyLong()))
                .thenReturn(CompletableFuture.completedFuture(response));

        CountDownLatch callbacksAttempted = new CountDownLatch(2);
        AtomicInteger successes = new AtomicInteger();
        AtomicInteger failures = new AtomicInteger();
        AtomicInteger uncertain = new AtomicInteger();
        DispatchCallback throwingCallback = new DispatchCallback() {
            @Override
            public void onSuccess(BatchItem item, long batchId) {
                successes.incrementAndGet();
                callbacksAttempted.countDown();
            }

            @Override
            public void onFailure(BatchItem item, Throwable error) {
                failures.incrementAndGet();
                callbacksAttempted.countDown();
                throw new IllegalStateException("callback failed after committing failure");
            }

            @Override
            public void onDispatchUncertain(BatchItem item, long batchId, Throwable error) {
                uncertain.incrementAndGet();
            }
        };

        dispatcher.dispatch(List.of(succeeded, rejected), prefillEp,
                91L, 100, "callback_isolation", throwingCallback);

        assertTrue(callbacksAttempted.await(5, TimeUnit.SECONDS));
        assertEquals(1, successes.get());
        assertEquals(1, failures.get());
        assertEquals(0, uncertain.get(),
                "a later callback exception must not reclassify any item");
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

    // ---- task40: priority passthrough to GenerateInputPB ----

    @Test
    void dispatchForwardsCarriedPriorityIntoGenerateInput() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        BatchItem item = createBatchItem(1L, 500, 200, prefillEp);
        item.ctx().getRequest().setPriority(60);

        List<EngineRpcService.EnqueueBatchRequestPB> sent = new CopyOnWriteArrayList<>();
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenAnswer(inv -> {
                    sent.add(inv.getArgument(2));
                    return CompletableFuture.completedFuture(ackResponse(1L, List.of(1L)));
                });

        dispatcher.dispatch(List.of(item), prefillEp, 1L, 100, "test", callback);

        assertTrue(callback.successLatch.await(5, TimeUnit.SECONDS));
        assertEquals(60, sentInput(sent.getFirst()).getPriority());
    }

    @Test
    void dispatchLeavesPriorityUnsetForNoPriorityRequests() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        BatchItem item = createBatchItem(1L, 500, 200, prefillEp);
        // default Request priority is the no-priority sentinel (0)

        List<EngineRpcService.EnqueueBatchRequestPB> sent = new CopyOnWriteArrayList<>();
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenAnswer(inv -> {
                    sent.add(inv.getArgument(2));
                    return CompletableFuture.completedFuture(ackResponse(1L, List.of(1L)));
                });

        dispatcher.dispatch(List.of(item), prefillEp, 1L, 100, "test", callback);

        assertTrue(callback.successLatch.await(5, TimeUnit.SECONDS));
        assertEquals(0, sentInput(sent.getFirst()).getPriority());
    }

    @Test
    void dispatchDualWritesCompatibleRoleAddress() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        BatchItem item = createBatchItem(1L, 500, 200, prefillEp);

        List<EngineRpcService.EnqueueBatchRequestPB> sent = new CopyOnWriteArrayList<>();
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(), anyLong()))
                .thenAnswer(inv -> {
                    sent.add(inv.getArgument(2));
                    return CompletableFuture.completedFuture(ackResponse(1L, List.of(1L)));
                });

        dispatcher.dispatch(List.of(item), prefillEp, 1L, 100, "role_compat", callback);

        assertTrue(callback.successLatch.await(5, TimeUnit.SECONDS));
        EngineRpcService.RoleAddrPB addr = sentInput(sent.getFirst())
                .getGenerateConfig().getRoleAddrs(0);
        assertEquals(EngineRpcService.RoleAddrPB.RoleType.PREFILL, addr.getRole());
        assertEquals("PREFILL", addr.getRoleStr());
        assertEquals(RoleType.PREFILL, RoleTypeProtoConverter.fromRoleAddr(addr));
    }

    private static EngineRpcService.GenerateInputPB sentInput(EngineRpcService.EnqueueBatchRequestPB request) {
        return request.getDpSlotsList().getFirst().getRequestsList().getFirst().getInput();
    }

    // ---- helpers ----

    private PrefillEndpoint createPrefillEndpoint() {
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        when(endpoint.getHttpPort()).thenReturn(8080);
        when(endpoint.getGrpcPort()).thenReturn(8090);
        when(endpoint.initiateGenerationDispatch(any(), any())).thenAnswer(invocation -> {
            Supplier<?> start = invocation.getArgument(1);
            return start.get();
        });
        return endpoint;
    }

    private PrefillEndpoint createRealPrefillEndpoint(String ip, int httpPort, int grpcPort) {
        WorkerStatus status = new WorkerStatus();
        status.setIp(ip);
        status.setPort(httpPort);
        status.setGrpcPort(grpcPort);
        status.setRole(RoleType.PREFILL);
        FlexlbConfig endpointConfig = new FlexlbConfig();
        endpointConfig.setCostFormula("10 + batchSize");
        endpointConfig.setFlexlbBatchQueueMaxSize(10);
        endpointConfig.setFlexlbBatchFixedWaitMs(1000);
        return new PrefillEndpoint(
                status, endpointConfig, mock(BatchDecisionHandler.class), reporter);
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

        return new BatchItem(ctx, new CompletableFuture<>(), null, prefill, null, prefillEp, null, System.currentTimeMillis());
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
        final AtomicInteger uncertainCount = new AtomicInteger(0);
        final CountDownLatch successLatch = new CountDownLatch(1);
        final CountDownLatch failureLatch = new CountDownLatch(1);
        final CountDownLatch uncertainLatch = new CountDownLatch(1);
        volatile Throwable lastError;

        @Override
        public void onSuccess(BatchItem item, long batchId) {
            successCount.incrementAndGet();
            successLatch.countDown();
        }

        @Override
        public void onFailure(BatchItem item, Throwable error) {
            lastError = error;
            failureCount.incrementAndGet();
            failureLatch.countDown();
        }

        @Override
        public void onDispatchUncertain(BatchItem item, long batchId, Throwable error) {
            lastError = error;
            uncertainCount.incrementAndGet();
            uncertainLatch.countDown();
        }
    }
}
