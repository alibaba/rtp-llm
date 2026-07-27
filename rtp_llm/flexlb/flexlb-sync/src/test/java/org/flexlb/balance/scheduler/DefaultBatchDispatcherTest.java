package org.flexlb.balance.scheduler;

import com.google.protobuf.Int64Value;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Method;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
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
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.timeout;
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

        dispatcher = new DefaultBatchDispatcher(grpcClient, configService, null, reporter);
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
    void dispatchUsesConfiguredDeadlineWhenAbsoluteDeadlineIsMissing() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        BatchItem item = createBatchItem(1L, 500, 200, prefillEp);
        AtomicReference<Long> capturedDeadlineMs = new AtomicReference<>();
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(), anyLong()))
                .thenAnswer(invocation -> {
                    capturedDeadlineMs.set(invocation.getArgument(3));
                    return CompletableFuture.completedFuture(ackResponse(1L, List.of(1L)));
                });

        dispatcher.dispatch(List.of(item), prefillEp, 1L, 100, "test", callback);

        assertTrue(callback.successLatch.await(5, TimeUnit.SECONDS));
        assertEquals(0L, item.absoluteDeadlineMs());
        assertEquals(5000L, capturedDeadlineMs.get());
    }

    @Test
    void dispatchClampsConfiguredDeadlineToAbsoluteDeadline() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        long absoluteDeadlineMs = System.currentTimeMillis() + 2000;
        BatchItem item = createBatchItem(1L, 500, 200, prefillEp, absoluteDeadlineMs);
        AtomicReference<Long> capturedDeadlineMs = new AtomicReference<>();
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(), anyLong()))
                .thenAnswer(invocation -> {
                    capturedDeadlineMs.set(invocation.getArgument(3));
                    return CompletableFuture.completedFuture(ackResponse(1L, List.of(1L)));
                });

        dispatcher.dispatch(List.of(item), prefillEp, 1L, 100, "test", callback);

        assertTrue(callback.successLatch.await(5, TimeUnit.SECONDS));
        assertTrue(capturedDeadlineMs.get() > 0);
        assertTrue(capturedDeadlineMs.get() <= 2000);
    }

    @Test
    void dispatchSkipsGrpcWhenAbsoluteDeadlineAlreadyExpired() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        BatchItem item = createBatchItem(
                1L, 500, 200, prefillEp, System.currentTimeMillis() - 1);

        dispatcher.dispatch(List.of(item), prefillEp, 1L, 100, "test", callback);

        assertTrue(callback.failureLatch.await(5, TimeUnit.SECONDS));
        assertEquals(1, callback.failureCount.get());
        // onTimeout fires before releaseBatch on the dispatch thread; use a
        // timed verify to avoid racing with the executor.
        verify(prefillEp, timeout(5000)).releaseBatch(1L);
        verify(grpcClient, never()).batchEnqueueAsync(
                anyString(), anyInt(), any(), anyLong());
    }

    @Test
    void completedItemWithExpiredDeadlineDoesNotBlockActiveItem() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        BatchItem completed = createBatchItem(
                1L, 500, 200, prefillEp, System.currentTimeMillis() - 1);
        completed.future().complete(null);
        BatchItem active = createBatchItem(2L, 300, 100, prefillEp);
        AtomicReference<Long> capturedDeadlineMs = new AtomicReference<>();
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(), anyLong()))
                .thenAnswer(invocation -> {
                    capturedDeadlineMs.set(invocation.getArgument(3));
                    return CompletableFuture.completedFuture(ackResponse(1L, List.of(2L)));
                });

        dispatcher.dispatch(
                List.of(completed, active), prefillEp, 1L, 100, "test", callback);

        assertTrue(callback.successLatch.await(5, TimeUnit.SECONDS));
        assertEquals(1, callback.successCount.get());
        assertEquals(1, callback.failureCount.get());
        assertEquals(5000L, capturedDeadlineMs.get());
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
    void dispatchKeepsCommittedItemsEvenIfCancellationRacesWithSend() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        BatchItem active = createBatchItem(1L, 500, 200, prefillEp);
        BatchItem cancelled = createBatchItem(2L, 300, 100, prefillEp);
        cancelled.ctx().cancel(); // mark as cancelled

        AtomicReference<EngineRpcService.EnqueueBatchRequestPB> captured = new AtomicReference<>();
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(), any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenAnswer(inv -> {
                    captured.set(inv.getArgument(2));
                    return CompletableFuture.completedFuture(ackResponse(1L, List.of(1L, 2L)));
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

    @Test
    void buildInput_rewrites_timeout_ms_when_absolute_deadline_set() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        long absoluteDeadlineMs = System.currentTimeMillis() + 2000;
        // Set a large initial timeout_ms; buildInput should overwrite it with the
        // dispatch-time remaining budget derived from absoluteDeadlineMs.
        BatchItem item = createBatchItem(1L, 500, 200, prefillEp, absoluteDeadlineMs, 100_000);
        AtomicReference<EngineRpcService.EnqueueBatchRequestPB> captured = new AtomicReference<>();
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(),
                any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenAnswer(inv -> {
                    captured.set(inv.getArgument(2));
                    return CompletableFuture.completedFuture(ackResponse(1L, List.of(1L)));
                });

        dispatcher.dispatch(List.of(item), prefillEp, 1L, 100, "test", callback);

        assertTrue(callback.successLatch.await(5, TimeUnit.SECONDS));
        EngineRpcService.EnqueueBatchRequestPB sent = captured.get();
        assertNotNull(sent);
        int timeoutMs = sent.getDpSlots(0).getRequests(0).getInput()
                .getGenerateConfig().getTimeoutMs();
        assertTrue(timeoutMs > 0, "timeout_ms should be rewritten to remaining budget");
        assertTrue(timeoutMs <= 2000,
                "timeout_ms should be clamped to remaining budget, got " + timeoutMs);
    }

    @Test
    void buildInput_preserves_timeout_ms_when_no_absolute_deadline() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        // absoluteDeadlineMs = 0 (legacy client); original timeout_ms must survive.
        BatchItem item = createBatchItem(1L, 500, 200, prefillEp, 0L, 100_000);
        AtomicReference<EngineRpcService.EnqueueBatchRequestPB> captured = new AtomicReference<>();
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(),
                any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenAnswer(inv -> {
                    captured.set(inv.getArgument(2));
                    return CompletableFuture.completedFuture(ackResponse(1L, List.of(1L)));
                });

        dispatcher.dispatch(List.of(item), prefillEp, 1L, 100, "test", callback);

        assertTrue(callback.successLatch.await(5, TimeUnit.SECONDS));
        EngineRpcService.EnqueueBatchRequestPB sent = captured.get();
        assertNotNull(sent);
        int timeoutMs = sent.getDpSlots(0).getRequests(0).getInput()
                .getGenerateConfig().getTimeoutMs();
        assertEquals(100_000, timeoutMs,
                "timeout_ms should be preserved when absoluteDeadlineMs is 0");
    }

    @Test
    void buildInput_rewrites_timeout_only_for_items_with_absolute_deadline() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        long absoluteDeadlineMs = System.currentTimeMillis() + 2000;
        // Item A carries an absolute deadline; its timeout_ms must be rewritten.
        BatchItem itemA = createBatchItem(1L, 500, 200, prefillEp, absoluteDeadlineMs, 100_000);
        // Item B has no deadline (legacy client); its timeout_ms must survive.
        BatchItem itemB = createBatchItem(2L, 300, 100, prefillEp, 0L, 50_000);
        AtomicReference<EngineRpcService.EnqueueBatchRequestPB> captured = new AtomicReference<>();
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(),
                any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenAnswer(inv -> {
                    captured.set(inv.getArgument(2));
                    return CompletableFuture.completedFuture(ackResponse(1L, List.of(1L, 2L)));
                });

        dispatcher.dispatch(List.of(itemA, itemB), prefillEp, 1L, 100, "test", callback);

        assertTrue(callback.successLatch.await(5, TimeUnit.SECONDS));
        EngineRpcService.EnqueueBatchRequestPB sent = captured.get();
        assertNotNull(sent);
        Integer timeoutA = null;
        Integer timeoutB = null;
        for (EngineRpcService.EnqueueBatchExternalInputPB request : sent.getDpSlots(0).getRequestsList()) {
            EngineRpcService.GenerateInputPB input = request.getInput();
            if (input.getRequestId() == 1L) {
                timeoutA = input.getGenerateConfig().getTimeoutMs();
            } else if (input.getRequestId() == 2L) {
                timeoutB = input.getGenerateConfig().getTimeoutMs();
            }
        }
        assertNotNull(timeoutA);
        assertNotNull(timeoutB);
        assertTrue(timeoutA > 0 && timeoutA <= 2000,
                "item A timeout_ms should be rewritten to remaining budget, got " + timeoutA);
        assertEquals(50_000, timeoutB.intValue(),
                "item B timeout_ms should be preserved when absoluteDeadlineMs is 0");
    }

    @Test
    void dispatch_drops_expired_items_and_sends_survivors() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        long now = System.currentTimeMillis();
        // Item A already expired; must be dropped before dispatch with onTimeout.
        BatchItem itemA = createBatchItem(1L, 500, 200, prefillEp, now - 1000, 100_000);
        // Item B still has budget; must be dispatched with rewritten timeout_ms.
        BatchItem itemB = createBatchItem(2L, 300, 100, prefillEp, now + 2000, 100_000);
        AtomicReference<EngineRpcService.EnqueueBatchRequestPB> captured = new AtomicReference<>();
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(),
                any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenAnswer(inv -> {
                    captured.set(inv.getArgument(2));
                    return CompletableFuture.completedFuture(ackResponse(1L, List.of(2L)));
                });

        dispatcher.dispatch(List.of(itemA, itemB), prefillEp, 1L, 100, "test", callback);

        assertTrue(callback.successLatch.await(5, TimeUnit.SECONDS));
        EngineRpcService.EnqueueBatchRequestPB sent = captured.get();
        assertNotNull(sent);
        long sentCount = sent.getDpSlotsList().stream()
                .mapToLong(slot -> slot.getRequestsCount())
                .sum();
        assertEquals(1, sentCount, "only the surviving item should be sent");
        EngineRpcService.GenerateInputPB input = sent.getDpSlots(0).getRequests(0).getInput();
        assertEquals(2L, input.getRequestId());
        int timeoutMs = input.getGenerateConfig().getTimeoutMs();
        assertTrue(timeoutMs > 0 && timeoutMs <= 2000,
                "survivor timeout_ms should be within remaining budget, got " + timeoutMs);
        assertEquals(1, callback.timeoutCount.get(), "expired item should get onTimeout");
        assertTrue(callback.timeoutRequestIds.contains(1L));
        verify(reporter).reportDispatchExpired("PREFILL", "127.0.0.1:8080", 1);
    }

    @Test
    void dispatch_skips_grpc_when_all_items_expired() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        long now = System.currentTimeMillis();
        BatchItem itemA = createBatchItem(1L, 500, 200, prefillEp, now - 1000, 100_000);
        BatchItem itemB = createBatchItem(2L, 300, 100, prefillEp, now - 500, 100_000);

        dispatcher.dispatch(List.of(itemA, itemB), prefillEp, 7L, 100, "test", callback);

        verify(prefillEp, timeout(5000)).releaseBatch(7L);
        assertEquals(2, callback.timeoutCount.get(), "every expired item should get onTimeout");
        assertTrue(callback.timeoutRequestIds.contains(1L));
        assertTrue(callback.timeoutRequestIds.contains(2L));
        verify(grpcClient, never()).batchEnqueueAsync(anyString(), anyInt(), any(), anyLong());
        verify(reporter).reportDispatchExpired("PREFILL", "127.0.0.1:8080", 2);
    }

    @Test
    void buildInput_floors_timeout_ms_to_1_when_deadline_already_passed() throws Exception {
        // Covers the race-leak path: an item whose deadline passes between the
        // per-item expiry check and buildInput must produce timeout_ms >= 1
        // (0 would be treated as "unset" downstream and fall back to defaults).
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        BatchItem item = createBatchItem(1L, 500, 200, prefillEp,
                System.currentTimeMillis() - 1000, 100_000);
        Method buildInput = DefaultBatchDispatcher.class.getDeclaredMethod(
                "buildInput", long.class, int.class, BatchItem.class);
        buildInput.setAccessible(true);

        EngineRpcService.GenerateInputPB input =
                (EngineRpcService.GenerateInputPB) buildInput.invoke(dispatcher, 1L, 1, item);

        int timeoutMs = input.getGenerateConfig().getTimeoutMs();
        assertTrue(timeoutMs >= 1, "timeout_ms must be floored to >= 1, got " + timeoutMs);
        assertEquals(1, timeoutMs, "expired remaining budget should floor to exactly 1ms");
    }

    @Test
    void dispatch_drops_expired_items_without_reporter() throws Exception {
        // reporter=null must be safe: expired items still get onTimeout, no gRPC, no exception.
        DefaultBatchDispatcher noReporterDispatcher =
                new DefaultBatchDispatcher(grpcClient, configService, null, null);
        try {
            PrefillEndpoint prefillEp = createPrefillEndpoint();
            BatchItem item = createBatchItem(
                    1L, 500, 200, prefillEp, System.currentTimeMillis() - 1000, 100_000);

            noReporterDispatcher.dispatch(List.of(item), prefillEp, 3L, 100, "test", callback);

            assertTrue(callback.failureLatch.await(5, TimeUnit.SECONDS));
            assertEquals(1, callback.timeoutCount.get());
            assertTrue(callback.timeoutRequestIds.contains(1L));
            verify(prefillEp, timeout(5000)).releaseBatch(3L);
            verify(grpcClient, never()).batchEnqueueAsync(anyString(), anyInt(), any(), anyLong());
        } finally {
            noReporterDispatcher.shutdown();
        }
    }

    @Test
    void dispatch_reports_fallback_when_deadline_expires_after_per_item_check() throws Exception {
        // Deterministically reproduce the race: the item is still valid at the
        // per-item expiry check, but its deadline passes before the EnqueueBatch
        // deadline computation. Uses an injected time source instead of sleeps.
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        long deadline = 1_000_000L;
        BatchItem item = createBatchItem(1L, 500, 200, prefillEp, deadline, 100_000);
        AtomicInteger nowCalls = new AtomicInteger();
        DefaultBatchDispatcher racingDispatcher =
                new DefaultBatchDispatcher(grpcClient, configService, null, reporter) {
                    @Override
                    long nowMs() {
                        // First call (per-item check): 1ms before the deadline;
                        // subsequent calls (deadline computation + log): past it.
                        return nowCalls.getAndIncrement() == 0 ? deadline - 1 : deadline + 1;
                    }
                };
        try {
            racingDispatcher.dispatch(List.of(item), prefillEp, 5L, 100, "test", callback);

            assertTrue(callback.failureLatch.await(5, TimeUnit.SECONDS));
            assertEquals(1, callback.timeoutCount.get(), "valid item should go through onTimeout");
            assertTrue(callback.timeoutRequestIds.contains(1L));
            verify(prefillEp, timeout(5000)).releaseBatch(5L);
            verify(grpcClient, never()).batchEnqueueAsync(anyString(), anyInt(), any(), anyLong());
            verify(reporter).reportDispatchExpired("PREFILL", "127.0.0.1:8080", 1);
        } finally {
            racingDispatcher.shutdown();
        }
    }

    @Test
    void dispatch_drops_only_deadline_items_and_preserves_legacy_items() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        // Item A carries an already-passed deadline; must be dropped.
        BatchItem itemA = createBatchItem(1L, 500, 200, prefillEp,
                System.currentTimeMillis() - 1000, 100_000);
        // Item B is a legacy client without deadline; must survive with its original timeout_ms.
        BatchItem itemB = createBatchItem(2L, 300, 100, prefillEp, 0L, 50_000);
        AtomicReference<EngineRpcService.EnqueueBatchRequestPB> captured = new AtomicReference<>();
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(),
                any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenAnswer(inv -> {
                    captured.set(inv.getArgument(2));
                    return CompletableFuture.completedFuture(ackResponse(1L, List.of(2L)));
                });

        dispatcher.dispatch(List.of(itemA, itemB), prefillEp, 1L, 100, "test", callback);

        assertTrue(callback.successLatch.await(5, TimeUnit.SECONDS));
        EngineRpcService.EnqueueBatchRequestPB sent = captured.get();
        assertNotNull(sent);
        long sentCount = sent.getDpSlotsList().stream()
                .mapToLong(slot -> slot.getRequestsCount())
                .sum();
        assertEquals(1, sentCount, "only the legacy item should be sent");
        EngineRpcService.GenerateInputPB input = sent.getDpSlots(0).getRequests(0).getInput();
        assertEquals(2L, input.getRequestId());
        assertEquals(50_000, input.getGenerateConfig().getTimeoutMs(),
                "legacy item timeout_ms must be preserved");
        assertEquals(1, callback.timeoutCount.get());
        assertTrue(callback.timeoutRequestIds.contains(1L));
        verify(reporter).reportDispatchExpired("PREFILL", "127.0.0.1:8080", 1);
    }

    @Test
    void dispatch_survives_onTimeout_exception_and_sends_valid_items() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        long now = System.currentTimeMillis();
        BatchItem itemA = createBatchItem(1L, 500, 200, prefillEp, now - 1000, 100_000);
        BatchItem itemB = createBatchItem(2L, 300, 100, prefillEp, now + 2000, 100_000);
        ThrowingOnTimeoutCallback throwingCallback = new ThrowingOnTimeoutCallback(1L);
        AtomicReference<EngineRpcService.EnqueueBatchRequestPB> captured = new AtomicReference<>();
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(),
                any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenAnswer(inv -> {
                    captured.set(inv.getArgument(2));
                    return CompletableFuture.completedFuture(ackResponse(1L, List.of(2L)));
                });

        dispatcher.dispatch(List.of(itemA, itemB), prefillEp, 1L, 100, "test", throwingCallback);

        assertTrue(throwingCallback.successLatch.await(5, TimeUnit.SECONDS),
                "valid item must still be dispatched despite the onTimeout exception");
        assertEquals(1, throwingCallback.successCount.get());
        EngineRpcService.EnqueueBatchRequestPB sent = captured.get();
        assertNotNull(sent);
        assertEquals(2L, sent.getDpSlots(0).getRequests(0).getInput().getRequestId());
        // The throwing item is routed to onFailure as best-effort fallback.
        assertEquals(1, throwingCallback.failureCount.get());
        // Batch proceeds with the survivor, so it must NOT be released here.
        verify(prefillEp, never()).releaseBatch(anyLong());
    }

    @Test
    void dispatch_survives_onTimeout_exception_when_all_items_expired() throws Exception {
        PrefillEndpoint prefillEp = createPrefillEndpoint();
        long now = System.currentTimeMillis();
        BatchItem itemA = createBatchItem(1L, 500, 200, prefillEp, now - 1000, 100_000);
        BatchItem itemB = createBatchItem(2L, 300, 100, prefillEp, now - 500, 100_000);
        ThrowingOnTimeoutCallback throwingCallback = new ThrowingOnTimeoutCallback(1L);

        dispatcher.dispatch(List.of(itemA, itemB), prefillEp, 9L, 100, "test", throwingCallback);

        // No batch leak: the batch is still released after the callback exception.
        verify(prefillEp, timeout(5000)).releaseBatch(9L);
        assertEquals(1, throwingCallback.timeoutCount.get(),
                "second expired item must still receive onTimeout");
        assertTrue(throwingCallback.timeoutRequestIds.contains(2L));
        // The throwing item is routed to onFailure; the second one delegates normally.
        assertEquals(2, throwingCallback.failureCount.get());
        verify(grpcClient, never()).batchEnqueueAsync(anyString(), anyInt(), any(), anyLong());
        verify(reporter).reportDispatchExpired("PREFILL", "127.0.0.1:8080", 2);
    }

    // ---- helpers ----

    private PrefillEndpoint createPrefillEndpoint() {
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getIp()).thenReturn("127.0.0.1");
        when(endpoint.getHttpPort()).thenReturn(8080);
        when(endpoint.getGrpcPort()).thenReturn(8090);
        when(endpoint.ipPort()).thenReturn("127.0.0.1:8080");
        return endpoint;
    }

    private BatchItem createBatchItem(long requestId, long seqLen, long hitCacheLen, PrefillEndpoint prefillEp) {
        return createBatchItem(requestId, seqLen, hitCacheLen, prefillEp, 0L);
    }

    private BatchItem createBatchItem(long requestId, long seqLen, long hitCacheLen,
                                      PrefillEndpoint prefillEp, long absoluteDeadlineMs) {
        return createBatchItem(requestId, seqLen, hitCacheLen, prefillEp, absoluteDeadlineMs, 0);
    }

    private BatchItem createBatchItem(long requestId, long seqLen, long hitCacheLen,
                                      PrefillEndpoint prefillEp, long absoluteDeadlineMs,
                                      int timeoutMs) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);

        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);

        // Provide a valid GenerateInputPB bytes (minimum: requestId + config with timeout_ms)
        EngineRpcService.GenerateInputPB input = EngineRpcService.GenerateInputPB.newBuilder()
                .setRequestId(requestId)
                .setGroupId(Int64Value.of(1L))
                .setGroupSize(1)
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder()
                        .setTimeoutMs(timeoutMs)
                        .build())
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

        if (absoluteDeadlineMs == 0) {
            return new BatchItem(ctx, new CompletableFuture<>(), null, prefill, null,
                    prefillEp, null, System.currentTimeMillis());
        }
        return new BatchItem(ctx, new CompletableFuture<>(), null, prefill, null,
                prefillEp, null, System.currentTimeMillis(), absoluteDeadlineMs);
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
        final AtomicInteger timeoutCount = new AtomicInteger(0);
        final List<Long> timeoutRequestIds = new CopyOnWriteArrayList<>();
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

        @Override
        public void onTimeout(BatchItem item, Throwable error) {
            timeoutCount.incrementAndGet();
            timeoutRequestIds.add(item.requestId());
            // Keep the default delegation semantics so existing failure assertions hold.
            onFailure(item, error);
        }
    }

    /**
     * Callback whose onTimeout throws for a specific requestId, to verify the
     * dispatcher's exception guard does not break the remaining items.
     */
    private static class ThrowingOnTimeoutCallback extends TestCallback {
        final long throwForRequestId;

        ThrowingOnTimeoutCallback(long throwForRequestId) {
            this.throwForRequestId = throwForRequestId;
        }

        @Override
        public void onTimeout(BatchItem item, Throwable error) {
            if (item.requestId() == throwForRequestId) {
                throw new RuntimeException("boom from onTimeout");
            }
            super.onTimeout(item, error);
        }
    }
}
