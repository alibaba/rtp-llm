package org.flexlb.balance.scheduler;

import org.flexlb.config.ConfigService;
import org.flexlb.service.TheoryCacheHitReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

import static org.flexlb.balance.scheduler.ContextResult.EndToken;
import static org.flexlb.balance.scheduler.ContextResult.Phase;
import static org.flexlb.balance.scheduler.ContextResult.ResponseKind;
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
import static org.mockito.Mockito.doCallRealMethod;
import static org.mockito.Mockito.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * A frontend cancel owns its publication; a queued close retains its independently claimed notification.
 */
@Timeout(15)
class RequestNotificationOwnershipRaceTest {
    @Test
    void externalCancellationCannotFinishAnotherRunningTerminalNotificationOwner()
            throws Exception {
        var config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.useFifoQueue(config);
        var service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var owner =
                (QueuedRequestScheduler)
                        SchedulerTestSupport.create(
                                service,
                                mock(DeliveryMetricsReporter.class),
                                mock(RequestSchedulerReporter.class),
                                mock(TheoryCacheHitReporter.class));
        var runtime = owner.runtime;
        var context = RequestProtocolTestSupport.context(config, "9401");
        var future = RequestProtocolTestSupport.register(owner, context);
        var originalTimer = runtime.deadlines();
        var timer = spy(originalTimer);
        var notificationStarted = new CountDownLatch(1);
        var finishNotification = new CountDownLatch(1);
        var blockFirstNotification = new AtomicBoolean(true);
        var notificationOwner = new AtomicReference<EndToken>();
        var originalCaller = Thread.currentThread();
        doAnswer(
                        call -> {
                            if (blockFirstNotification.getAndSet(false)) {
                                assertNotSame(
                                        originalCaller,
                                        Thread.currentThread(),
                                        "the queue decision thread owns this notification");
                                var state = context.snapshot();
                                assertEquals(Phase.ENDING, state.phase());
                                notificationOwner.set(state.close().endOperation());
                                assertNotNull(notificationOwner.get());
                                assertFalse(state.close().notificationsAttempted());
                                notificationStarted.countDown();
                                assertTrue(finishNotification.await(5, TimeUnit.SECONDS));
                            }
                            return call.callRealMethod();
                        })
                .when(timer)
                .update(eq(context), any());
        ReflectionTestUtils.setField(runtime, "timer", timer);
        // Restore the real asynchronous postEvent path, replacing the helper's synchronous control
        // stub.
        doAnswer(
                        call -> {
                            call.callRealMethod();
                            assertTrue(notificationStarted.await(5, TimeUnit.SECONDS));
                            return null;
                        })
                .when(owner)
                .queueControl(eq(context), org.mockito.ArgumentMatchers.isNull());
        owner.start();
        try {
            assertNull(context.snapshot().route());
            assertTrue(
                    future.cancel(false),
                    "the frontend cancellation keeps its synchronous Future contract");
            assertTrue(future.isCancelled());
            assertEquals(ResponseKind.CANCELLATION, context.snapshot().selectedResponse().kind());
            assertEquals(CancelReason.CLIENT_CANCELLED, context.snapshot().cancellation().reason());
            assertEquals(
                    Phase.ENDING,
                    context.snapshot().phase(),
                    "an external publisher cannot complete a running close owner");
            assertSame(notificationOwner.get(), context.snapshot().close().endOperation());
            assertFalse(context.snapshot().close().notificationsAttempted());
            assertFalse(context.snapshot().close().notificationsDone());
            assertFalse(context.snapshot().close().timersDone());
            assertEquals(1, runtime.requests().liveRequestCount());
            assertSame(context, runtime.requests().findActive("9401"));
            assertTrue(
                    publications(runtime).isEmpty(),
                    "the frontend publisher releases its own registration only");
            verify(timer, times(1)).update(eq(context), any());
            finishNotification.countDown();
            RequestProtocolTestSupport.awaitCondition(
                    () ->
                            context.snapshot().phase() == Phase.CLOSED
                                    && runtime.requests().liveRequestCount() == 0);
            SchedulerTestSupport.awaitRequestWork(runtime);
            assertNull(context.snapshot().close().endOperation());
            assertTrue(context.snapshot().close().notificationsDone());
            assertTrue(context.snapshot().close().timersDone());
            assertTrue(future.isCancelled());
            assertEquals(CancelReason.CLIENT_CANCELLED, context.snapshot().cancellation().reason());
            assertTrue(publications(runtime).isEmpty());
            assertEquals(
                    0L,
                    ((Number) ReflectionTestUtils.getField(runtime, "pendingWork")).longValue());
            assertNull(SchedulerTestSupport.failure(owner));
        } finally {
            finishNotification.countDown();
            doCallRealMethod()
                    .when(owner)
                    .queueControl(eq(context), org.mockito.ArgumentMatchers.isNull());
            ReflectionTestUtils.setField(runtime, "timer", originalTimer);
            runtime.shutdown();
        }
    }

    @SuppressWarnings("unchecked")
    private static Set<ResponseCompletionExecutor.CompletionRegistration> publications(
            SchedulerRuntime runtime) {
        var values =
                (Set<ResponseCompletionExecutor.CompletionRegistration>)
                        ReflectionTestUtils.getField(
                                SchedulerTestSupport.responseCompletions(runtime), "registrations");
        synchronized (values) {
            return Set.copyOf(values);
        }
    }
}
