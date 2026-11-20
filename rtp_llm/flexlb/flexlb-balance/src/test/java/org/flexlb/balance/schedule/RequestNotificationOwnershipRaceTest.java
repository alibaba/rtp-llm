package org.flexlb.balance.schedule;

import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.metrics.RequestSchedulerReporter;
import org.flexlb.balance.requestlog.TheoryCacheHitReporter;
import org.flexlb.config.ConfigService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

import static org.flexlb.balance.schedule.RequestContext.Phase;
import static org.flexlb.balance.schedule.RequestContext.ResponseKind;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * A frontend cancel owns its publication; cleanup retains its independently claimed notification.
 */
@Timeout(15)
class RequestNotificationOwnershipRaceTest {
    @Test
    void externalCancellationPreservesCleanupUntilItCompletes() throws Exception {
        var config = SchedulingTestConfig.newConfig();
        SchedulingTestConfig.useFifoQueue(config);
        var service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var owner =
                (QueuedRequestScheduler)
                        SchedulerTestServices.scheduler(
                                SchedulerTestSupport.create(
                                        service,
                                        mock(DeliveryMetricsReporter.class),
                                        mock(RequestSchedulerReporter.class),
                                        mock(TheoryCacheHitReporter.class)));
        var runtime =
                org.flexlb.balance.schedule.SchedulerTestSupport.runtime(
                        SchedulerTestServices.lifecycle(owner));
        var context = RequestProtocolTestSupport.context(config, 9401L);
        var future = RequestProtocolTestSupport.submitForManualPlacement(owner, context);
        var originalTimer = runtime.deadlines();
        var timer = spy(originalTimer);
        var notificationStarted = new CountDownLatch(1);
        var finishNotification = new CountDownLatch(1);
        var blockFirstNotification = new AtomicBoolean(true);
        var notificationOwner = new AtomicReference<RequestState.State>();
        var originalCaller = Thread.currentThread();
        doAnswer(
                        call -> {
                            if (blockFirstNotification.getAndSet(false)) {
                                assertNotSame(
                                        originalCaller,
                                        Thread.currentThread(),
                                        "the cancelling thread owns cleanup");
                                var state = context.snapshot();
                                assertEquals(Phase.ENDING, state.phase());
                                notificationOwner.set(state.endState());
                                assertNotNull(notificationOwner.get());
                                assertFalse(state.notificationsDone());
                                notificationStarted.countDown();
                                assertTrue(finishNotification.await(5, TimeUnit.SECONDS));
                            }
                            return call.callRealMethod();
                        })
                .when(timer)
                .update(eq(context), any());
        org.flexlb.balance.schedule.SchedulerTestServices.setDependency(runtime, "timer", timer);
        try {
            assertNull(context.snapshot().route());
            var cancellation =
                    java.util.concurrent.CompletableFuture.supplyAsync(() -> future.cancel(false));
            assertTrue(notificationStarted.await(5, TimeUnit.SECONDS));
            assertEquals(
                    ResponseKind.CANCELLATION,
                    context.snapshot().responseForFuture().responseKind());
            assertEquals(CancelReason.CLIENT_CANCELLED, context.snapshot().cancellationReason());
            assertEquals(
                    Phase.ENDING,
                    context.snapshot().phase(),
                    "an external publisher cannot complete a running close owner");
            assertSame(notificationOwner.get(), context.snapshot().endState());
            assertFalse(context.snapshot().notificationsDone());
            assertEquals(1, runtime.requests().liveRequestCount());
            assertSame(context, runtime.requests().findActive("9401"));

            verify(timer, times(1)).update(eq(context), any());
            finishNotification.countDown();
            assertTrue(cancellation.get(5, TimeUnit.SECONDS));
            RequestProtocolTestSupport.awaitCondition(
                    () ->
                            context.snapshot().phase() == Phase.CLOSED
                                    && runtime.requests().liveRequestCount() == 0);
            SchedulerTestSupport.awaitRequestWork(runtime);
            assertSame(notificationOwner.get(), context.snapshot().endState());
            assertTrue(context.snapshot().notificationsDone());
            assertTrue(future.isCancelled());
            assertEquals(CancelReason.CLIENT_CANCELLED, context.snapshot().cancellationReason());
            assertTrue(publications(runtime) == 0);
            assertNull(SchedulerTestSupport.failure(owner));
        } finally {
            finishNotification.countDown();
            org.flexlb.balance.schedule.SchedulerTestServices.setDependency(
                    runtime, "timer", originalTimer);
            SchedulerTestSupport.shutdown(runtime);
        }
    }

    private static int publications(SchedulerTestServices runtime) {
        var executorOwner = SchedulerTestSupport.responseCompletions(runtime);
        var lock = ReflectionTestUtils.getField(executorOwner, "completionLock");
        synchronized (lock) {
            return (int) ReflectionTestUtils.getField(executorOwner, "pendingCompletions");
        }
    }
}
