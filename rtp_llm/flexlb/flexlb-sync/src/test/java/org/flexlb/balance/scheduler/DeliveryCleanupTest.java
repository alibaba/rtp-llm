package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;

import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class DeliveryCleanupTest {
    @Test void alreadyPhysicalSettledDoesNotSubmitToClosedExecutorOrLeaveDedupEntry() {
        var runtime = runtime();
        try {
            var claim = claim();
            runtime.signals().completeDelivery(claim);
            stopWorkers(runtime).shutdown();
            when(claim.owner().snapshot().delivery().send()).thenReturn(ContextResult.SendState.NOT_SENT);
            var cleanup = new DeliveryCleanup();
            assertDoesNotThrow(() -> cleanup.start(claim, mock(EngineGrpcClient.class), runtime).join());
            assertDoesNotThrow(() -> cleanup.start(claim, mock(EngineGrpcClient.class), runtime).join());
        } finally { runtime.shutdown(); }
    }

    @Test void physicalSettlementWinningExecutorRejectionDoesNotRecordFalseFailure() {
        var runtime = runtime();
        var original = stopWorkers(runtime);
        try {
            var claim = claim();
            var rejecting = mock(java.util.concurrent.ExecutorService.class);
            doAnswer(call -> {
                runtime.signals().completeDelivery(claim);
                throw new java.util.concurrent.RejectedExecutionException("closed after physical settlement");
            }).when(rejecting).execute(any(Runnable.class));
            org.springframework.test.util.ReflectionTestUtils.setField(runtime, "stopWorkers", rejecting);
            assertDoesNotThrow(() -> new DeliveryCleanup().start(claim, mock(EngineGrpcClient.class), runtime).join());
            assertTrue(runtime.signals().delivery(claim).toCompletableFuture().isDone());
        } finally {
            org.springframework.test.util.ReflectionTestUtils.setField(runtime, "stopWorkers", original);
            // Real shutdown rethrows any recorded facility failure, so this checks the entire lifecycle.
            assertDoesNotThrow(runtime::shutdown);
        }
    }

    @Test void exactIdentityDedupEndsAtPhysicalSignalRatherThanSessionStart() throws Exception {
        var runtime = runtime();
        try {
            var channel = mock(EngineGrpcClient.class);
            var claim = claim();
            var equalValue = new DeliveryClaim(claim.owner(), claim.sequence(), claim.item(), claim.kind(), claim.batchId());
            var cleanup = new DeliveryCleanup();
            var first = cleanup.start(claim, channel, runtime);
            assertSame(first, cleanup.start(claim, channel, runtime));
            var separate = cleanup.start(equalValue, channel, runtime);
            assertNotSame(first, separate, "same record value does not identify the same operation");
            assertFalse(runtime.signals().delivery(claim).toCompletableFuture().isDone());
            runtime.signals().completeDelivery(claim);
            first.get(2, TimeUnit.SECONDS);
            assertFalse(runtime.signals().delivery(equalValue).toCompletableFuture().isDone(),
                    "physical settlement belongs only to the exact claim");
            runtime.signals().completeDelivery(equalValue);
            separate.get(2, TimeUnit.SECONDS);
            verifyNoInteractions(channel);
        } finally { runtime.shutdown(); }
    }

    @Test void rejectedStartupUsesAcceptedWorkAndDoesNotInventPhysicalSettlement() throws Exception {
        var runtime = runtime();
        var claim = claim();
        try {
            stopWorkers(runtime).shutdown();
            when(claim.owner().snapshot().delivery().send()).thenReturn(ContextResult.SendState.NOT_SENT);
            var cleanup = new DeliveryCleanup();
            var first = cleanup.start(claim, mock(EngineGrpcClient.class), runtime);
            assertEquals(DeliveryCleanup.Result.NOT_SENT, first.get(2, TimeUnit.SECONDS));
            assertSame(first, cleanup.start(claim, mock(EngineGrpcClient.class), runtime));
            assertFalse(runtime.signals().delivery(claim).toCompletableFuture().isDone());
            runtime.signals().completeDelivery(claim);
            var after = cleanup.start(claim, mock(EngineGrpcClient.class), runtime);
            assertNotSame(first, after);
            assertDoesNotThrow(after::join);
        } finally { assertDoesNotThrow(runtime::shutdown); }
    }

    @Test void settlementBeforeQueuedPreemptionDoesNotLeaveItsCallerWaiting() throws Exception {
        var runtime = runtime();
        var original = stopWorkers(runtime);
        var queued = new java.util.concurrent.ConcurrentLinkedQueue<Runnable>();
        var paused = mock(java.util.concurrent.ExecutorService.class);
        doAnswer(call -> { queued.add(call.getArgument(0)); return null; }).when(paused).execute(any(Runnable.class));
        org.springframework.test.util.ReflectionTestUtils.setField(runtime, "stopWorkers", paused);
        try {
            var claim = claim();
            var client = mock(EngineGrpcClient.class);
            var cleanup = new DeliveryCleanup();
            cleanup.start(claim, client, runtime);
            runtime.signals().completeDelivery(claim);
            var registration = new PreemptionRegistration(claim.owner(), 1L,
                    new org.flexlb.balance.endpoint.DecodeResources.ReservationHandle(1L, 41L, 1L),
                    new org.flexlb.balance.preemption.CancelTarget("127.0.0.1", 9000));
            var answer = cleanup.preempt(registration, client, runtime, 500L);
            Runnable task;
            while ((task = queued.poll()) != null) { task.run(); }
            assertEquals(EngineGrpcClient.CancelReply.FAILED, answer.get(1, TimeUnit.SECONDS));
            verifyNoInteractions(client);
        } finally {
            org.springframework.test.util.ReflectionTestUtils.setField(runtime, "stopWorkers", original);
            runtime.shutdown();
        }
    }

    private static java.util.concurrent.ExecutorService stopWorkers(SchedulerRuntime runtime) {
        return (java.util.concurrent.ExecutorService) org.springframework.test.util.ReflectionTestUtils.getField(runtime, "stopWorkers");
    }

    private static DeliveryClaim claim() {
        var context = mock(RequestContext.class, RETURNS_DEEP_STUBS);
        var claim = new DeliveryClaim(context, 1, mock(RequestRoute.class), DeliveryClaimKind.BATCH_ENQUEUE, 17);
        when(context.snapshot().delivery().identity()).thenReturn(claim);
        when(context.snapshot().delivery().send()).thenReturn(ContextResult.SendState.NOT_STARTED);
        return claim;
    }

    private static SchedulerRuntime runtime() {
        var config = DeliveryTestSupport.service(SchedulingTestConfig.batchConfig());
        return new SchedulerRuntime(new RequestRepository(), mock(EndpointRegistry.class),
                mock(DeliveryMetricsReporter.class), mock(RequestSchedulerReporter.class),
                new DeliveryDispatcher(null, config, new DeliveryMetricsReporter(null)), config, mock(EngineGrpcClient.class));
    }
}
