package org.flexlb.balance.scheduler;

import org.flexlb.config.ConfigService;
import org.flexlb.service.RecentCacheKeyTraceReporter;
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

import static org.flexlb.balance.scheduler.ContextResult.End;
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
import static org.mockito.Mockito.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** A frontend cancel owns its publication; a queued close retains its independently claimed notification. */
@Timeout(15)
class RequestNotificationOwnershipRaceTest {
    @Test
    void externalCancellationDoesNotReleaseItsQueuedCloseOwner() throws Exception {
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
                                mock(RecentCacheKeyTraceReporter.class));
        var runtime = owner.runtime;
        var context = RequestProtocolTestSupport.context(config, 9401L);
        var future = RequestProtocolTestSupport.register(owner, context);
        var originalTimer = runtime.deadlines();
        var timer = spy(originalTimer);
        var notificationStarted = new CountDownLatch(1);
        var finishNotification = new CountDownLatch(1);
        var blockFirstNotification = new AtomicBoolean(true);
        var notificationOwner = new AtomicReference<End>();
        var originalCaller = Thread.currentThread();
        doAnswer(
                        call -> {
                            if (blockFirstNotification.getAndSet(false)) {
                                assertNotSame(
                                        originalCaller,
                                        Thread.currentThread(),
                                        "the existing request work lane owns this notification");
                                var state = context.snapshot();
                                assertEquals(Phase.ENDING, state.phase());
                                notificationOwner.set(state.outcome());
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
        var workStarted = new CountDownLatch(1);
        var finishWork = new CountDownLatch(1);
        try (var work = runtime.registerRequestWork(context)) {
            work.submitClose(
                    () -> {
                        workStarted.countDown();
                        RequestProtocolTestSupport.await(finishWork);
                    }, false);
        }
        owner.start();
        try {
            assertTrue(workStarted.await(5, TimeUnit.SECONDS));
            assertNull(context.snapshot().route());
            assertTrue(
                    future.cancel(false),
                    "the frontend cancellation keeps its synchronous Future contract");
            assertTrue(future.isCancelled());
            finishWork.countDown();
            assertTrue(notificationStarted.await(5, TimeUnit.SECONDS));
            assertEquals(ResponseKind.CANCELLATION, context.snapshot().selectedResponse().kind());
            assertEquals(CancelReason.CLIENT_CANCELLED, context.snapshot().cancellation().reason());
            assertEquals(
                    Phase.ENDING,
                    context.snapshot().phase(),
                    "an external publisher cannot complete a running close owner");
            assertSame(notificationOwner.get(), context.snapshot().outcome());
            assertFalse(context.snapshot().close().notificationsAttempted());
            assertFalse(context.snapshot().close().notificationsDone());
            assertFalse(context.snapshot().close().timersDone());
            assertEquals(1, runtime.requests().liveRequestCount());
            assertSame(context, runtime.requests().findActive(9401L));
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
            assertSame(notificationOwner.get(), context.snapshot().outcome());
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
            finishWork.countDown();
            finishNotification.countDown();
            ReflectionTestUtils.setField(runtime, "timer", originalTimer);
            runtime.shutdown();
        }
    }

    @SuppressWarnings("unchecked")
    private static Set<ResponsePublication> publications(SchedulerRuntime runtime) {
        var values =
                (Set<ResponsePublication>)
                        ReflectionTestUtils.getField(
                                SchedulerTestSupport.responseCompletions(runtime), "registrations");
        synchronized (values) {
            return Set.copyOf(values);
        }
    }
}
