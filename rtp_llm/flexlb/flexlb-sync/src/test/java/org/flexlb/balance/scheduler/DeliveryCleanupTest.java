package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.eviction.EngineCancelChannel;
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
            runtime.deliveryCleanupExecutor.shutdown();
            var cleanup = new DeliveryCleanup();
            assertDoesNotThrow(() -> cleanup.start(claim, mock(EngineCancelChannel.class), runtime).join());
            assertDoesNotThrow(() -> cleanup.start(claim, mock(EngineCancelChannel.class), runtime).join());
        } finally { runtime.shutdown(); }
    }

    @Test void physicalSettlementWinningExecutorRejectionDoesNotRecordFalseFailure() {
        var runtime = runtime();
        var original = runtime.deliveryCleanupExecutor;
        try {
            var claim = claim();
            var rejecting = mock(java.util.concurrent.ExecutorService.class);
            doAnswer(call -> {
                runtime.signals().completeDelivery(claim);
                throw new java.util.concurrent.RejectedExecutionException("closed after physical settlement");
            }).when(rejecting).execute(any(Runnable.class));
            org.springframework.test.util.ReflectionTestUtils.setField(runtime, "deliveryCleanupExecutor", rejecting);
            assertDoesNotThrow(() -> new DeliveryCleanup().start(claim, mock(EngineCancelChannel.class), runtime).join());
            assertTrue(runtime.signals().delivery(claim).toCompletableFuture().isDone());
        } finally {
            org.springframework.test.util.ReflectionTestUtils.setField(runtime, "deliveryCleanupExecutor", original);
            // Real shutdown rethrows any recorded facility failure, so this checks the entire lifecycle.
            assertDoesNotThrow(runtime::shutdown);
        }
    }

    @Test void exactIdentityDedupEndsAtPhysicalSignalRatherThanSessionStart() throws Exception {
        var runtime = runtime();
        try {
            var channel = mock(EngineCancelChannel.class);
            var claim = claim();
            var equalValue = new DeliveryClaim(claim.owner(), claim.sequence(), claim.item(), claim.kind(), claim.batchId());
            var cleanup = new DeliveryCleanup();
            var first = cleanup.start(claim, channel, runtime);
            assertSame(first, cleanup.start(claim, channel, runtime));
            var separate = cleanup.start(equalValue, channel, runtime);
            assertNotSame(first, separate, "same record value does not identify the same operation");
            assertFalse(first.isDone());
            runtime.signals().completeDelivery(claim);
            first.get(2, TimeUnit.SECONDS);
            assertFalse(separate.isDone(), "physical settlement belongs only to the exact claim");
            runtime.signals().completeDelivery(equalValue);
            separate.get(2, TimeUnit.SECONDS);
            verifyNoInteractions(channel);
        } finally { runtime.shutdown(); }
    }

    @Test void failedStartupKeepsDedupAndDoesNotInventPhysicalSettlement() {
        var runtime = runtime();
        var claim = claim();
        try {
            runtime.deliveryCleanupExecutor.shutdown();
            var cleanup = new DeliveryCleanup();
            var first = cleanup.start(claim, mock(EngineCancelChannel.class), runtime);
            assertTrue(first.isCompletedExceptionally());
            assertSame(first, cleanup.start(claim, mock(EngineCancelChannel.class), runtime));
            assertFalse(runtime.signals().delivery(claim).toCompletableFuture().isDone());
            runtime.signals().completeDelivery(claim);
            var after = cleanup.start(claim, mock(EngineCancelChannel.class), runtime);
            assertNotSame(first, after);
            assertDoesNotThrow(after::join);
        } finally { assertThrows(RuntimeException.class, runtime::shutdown); }
    }

    private static DeliveryClaim claim() {
        return new DeliveryClaim(new RequestContext(SchedulingTestConfig.batchConfig()), 1,
                mock(RequestRoute.class), DeliveryClaimKind.BATCH_ENQUEUE, 17);
    }

    private static SchedulerRuntime runtime() {
        var config = DeliveryTestSupport.service(SchedulingTestConfig.batchConfig());
        return new SchedulerRuntime(new RequestRepository(), mock(EndpointRegistry.class),
                mock(DeliveryMetricsReporter.class), mock(RequestSchedulerReporter.class),
                new DeliveryDispatcher(null, config, null, new DeliveryMetricsReporter(null), null), config, mock(EngineCancelChannel.class));
    }
}
