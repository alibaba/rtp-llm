package org.flexlb.balance.schedule;

import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.metrics.RequestSchedulerReporter;
import org.flexlb.balance.requestlog.TheoryCacheHitReporter;
import org.flexlb.balance.schedule.ResponseCompletionExecutor.Registration;
import org.flexlb.balance.schedule.ResponseCompletionExecutor.RequestFuture;
import org.flexlb.config.ConfigService;
import org.flexlb.worker.endpoint.Response;
import org.flexlb.worker.endpoint.StrategyErrorType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.flexlb.balance.schedule.RequestProtocolTestSupport.await;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Frontend result selection precedes unlocked publication and cannot undo TTL cleanup.
 */
class RequestCompletionPublicationRaceTest {

    @ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    @Timeout(15)
    void externalCloseWaitsForNestedCompletionCallbacks(boolean asyncOuter) throws Exception {
        var responseExecutor = new ResponseCompletionExecutor(1);
        var outerRegistration = SchedulerTestSupport.responsePublication(responseExecutor);
        var innerRegistration = SchedulerTestSupport.responsePublication(responseExecutor);
        var outer = new RequestFuture((completion, response, failure) -> false);
        var inner = new RequestFuture((completion, response, failure) -> false);
        var response = new Response();
        var callbackEntered = new CountDownLatch(1);
        var resumeCallback = new CountDownLatch(1);
        var outerCallback =
                outer.thenRun(
                        () -> {
                            assertTrue(
                                    innerRegistration.completeNow(
                                            () -> inner.completeOwned(response)));
                            callbackEntered.countDown();
                            await(resumeCallback);
                        });
        var closingThread = new java.util.concurrent.atomic.AtomicReference<Thread>();
        var closer = Executors.newSingleThreadExecutor();
        try {
            var execution =
                    CompletableFuture.runAsync(
                            () -> {
                                if (asyncOuter) {
                                    outerRegistration.submit(() -> outer.completeOwned(response));
                                } else {
                                    assertTrue(
                                            outerRegistration.completeNow(
                                                    () -> outer.completeOwned(response)));
                                }
                            });
            assertTrue(callbackEntered.await(3, TimeUnit.SECONDS));
            var close =
                    closer.submit(
                            () -> {
                                closingThread.set(Thread.currentThread());
                                responseExecutor.close();
                            });
            RequestProtocolTestSupport.awaitCondition(
                    () ->
                            closingThread.get() != null
                                    && closingThread.get().getState() == Thread.State.WAITING);
            assertThrows(
                    IllegalStateException.class,
                    () -> SchedulerTestSupport.responsePublication(responseExecutor),
                    "close must stop accepting new registrations");
            assertTrue(outer.isDone());
            assertTrue(inner.isDone());
            assertFalse(
                    outerCallback.isDone(),
                    "the source result does not mean its callback has finished");
            assertFalse(close.isDone());
            var workers =
                    (ExecutorService)
                            org.springframework.test.util.ReflectionTestUtils.getField(
                                    responseExecutor, "completionWorkers");
            assertFalse(
                    workers.isShutdown(),
                    "workers must stay alive until accepted callbacks return");
            resumeCallback.countDown();
            outerCallback.get(3, TimeUnit.SECONDS);
            execution.get(3, TimeUnit.SECONDS);
            close.get(3, TimeUnit.SECONDS);
            assertTrue(workers.isTerminated());
        } finally {
            resumeCallback.countDown();
            innerRegistration.close();
            outerRegistration.close();
            closer.shutdownNow();
            assertTrue(closer.awaitTermination(5, TimeUnit.SECONDS));
            responseExecutor.close();
        }
    }

    @Test
    void selectedCancellationCompletesWhenCompletionWorkersRejectSubmission() throws Exception {
        var config = SchedulingTestConfig.newConfig();
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        AbstractRequestScheduler registry =
                org.flexlb.balance.schedule.SchedulerTestSupport.create(
                        service,
                        mock(DeliveryMetricsReporter.class),
                        mock(RequestSchedulerReporter.class),
                        mock(TheoryCacheHitReporter.class));
        try {
            CompletableFuture<Response> future =
                    registry.submit(RequestProtocolTestSupport.context(config, 503L), () -> {});
            var responseExecutor =
                    (ResponseCompletionExecutor)
                            org.springframework.test.util.ReflectionTestUtils.getField(
                                    org.flexlb.balance.schedule.SchedulerTestSupport.runtime(
                                            registry),
                                    "responseCompletions");
            var executor =
                    (ExecutorService)
                            org.springframework.test.util.ReflectionTestUtils.getField(
                                    responseExecutor, "completionWorkers");
            executor.shutdown();
            assertThrows(
                    java.util.concurrent.RejectedExecutionException.class,
                    () -> registry.cancel("503", CancelReason.CLIENT_CANCELLED));
            assertThrows(ExecutionException.class, () -> future.get(2L, TimeUnit.SECONDS));
            assertEquals(
                    RequestResult.State.CANCELLED,
                    org.flexlb.balance.schedule.SchedulerTestSupport.repository(registry)
                            .getRequestState("503")
                            .state());
        } finally {
            if (RequestProtocolTestSupport.closeRegistrationAndAwaitScheduling(registry)) {
                registry.closeOutstandingAndTerminalize();
                SchedulerTestSupport.timer(
                                org.flexlb.balance.schedule.SchedulerTestSupport.runtime(registry))
                        .close();
                SchedulerTestSupport.closeRequestExecutors(
                        org.flexlb.balance.schedule.SchedulerTestSupport.runtime(registry));
            }
        }
    }

    @Test
    @Timeout(15)
    void callbackCompletionsDrainBeforeExternalExecutorClose() throws Exception {
        var config = spy(SchedulingTestConfig.newConfig());
        var runtime = spy(config.getInternalRuntime());
        when(runtime.getResponseCompletionThreads()).thenReturn(1);
        when(config.getInternalRuntime()).thenReturn(runtime);
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        AbstractRequestScheduler scheduler =
                org.flexlb.balance.schedule.SchedulerTestSupport.create(
                        service,
                        mock(DeliveryMetricsReporter.class),
                        mock(RequestSchedulerReporter.class),
                        mock(TheoryCacheHitReporter.class));
        var responseExecutor =
                (ResponseCompletionExecutor)
                        org.springframework.test.util.ReflectionTestUtils.getField(
                                org.flexlb.balance.schedule.SchedulerTestSupport.runtime(scheduler),
                                "responseCompletions");
        int count = 256;
        var callbacks = new java.util.ArrayList<CompletableFuture<Void>>(count);
        var depth = new java.util.concurrent.atomic.AtomicInteger();
        var completed = new java.util.concurrent.atomic.AtomicInteger();
        try {
            for (int index = 0; index < count; index++) {
                long requestId = 10_000L + index;
                var context = RequestProtocolTestSupport.context(config, requestId);
                var future =
                        RequestProtocolTestSupport.submitForManualPlacement(scheduler, context);
                boolean last = index == count - 1;
                callbacks.add(
                        future.thenAccept(
                                response -> {
                                    assertFalse(Thread.holdsLock(context));
                                    assertTrue(
                                            Thread.currentThread()
                                                    .getName()
                                                    .startsWith("response-completion-"));
                                    assertEquals(
                                            1,
                                            depth.incrementAndGet(),
                                            "asynchronous publications must not recurse");
                                    try {
                                        assertEquals(
                                                StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                                                response.getCode());
                                        completed.incrementAndGet();
                                        if (!last) {
                                            scheduler.cancel(
                                                    String.valueOf(requestId + 1),
                                                    CancelReason.CLIENT_CANCELLED);
                                        }
                                    } finally {
                                        depth.decrementAndGet();
                                    }
                                }));
            }
            scheduler.cancel("10000", CancelReason.CLIENT_CANCELLED);
            CompletableFuture.allOf(callbacks.toArray(CompletableFuture[]::new))
                    .get(10L, TimeUnit.SECONDS);
            responseExecutor.close();
            assertEquals(count, completed.get());
            assertEquals(
                    0,
                    org.flexlb.balance.schedule.SchedulerTestSupport.repository(scheduler)
                            .liveRequestCount());
        } finally {
            RequestProtocolTestSupport.closeRegistrationAndAwaitScheduling(scheduler);
            scheduler.closeOutstandingAndTerminalize();
            SchedulerTestSupport.timer(
                            org.flexlb.balance.schedule.SchedulerTestSupport.runtime(scheduler))
                    .close();
            SchedulerTestSupport.closeRequestExecutors(
                    org.flexlb.balance.schedule.SchedulerTestSupport.runtime(scheduler));
        }
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    @Timeout(15)
    void concurrentExecutorCloseSharesResultAndPreservesInterrupt(boolean shutdownFails)
            throws Exception {
        var responseExecutor = new ResponseCompletionExecutor(1);
        var executor =
                (ExecutorService)
                        org.springframework.test.util.ReflectionTestUtils.getField(
                                responseExecutor, "completionWorkers");
        RuntimeException failure =
                shutdownFails ? new IllegalStateException("shutdown failed") : null;
        var failedExecutor =
                shutdownFails ? mock(java.util.concurrent.ThreadPoolExecutor.class) : null;
        if (shutdownFails) {
            executor.shutdown();
            org.mockito.Mockito.doThrow(failure).when(failedExecutor).shutdown();
            org.springframework.test.util.ReflectionTestUtils.setField(
                    responseExecutor, "completionWorkers", failedExecutor);
        }
        var registration = SchedulerTestSupport.responsePublication(responseExecutor);
        assertNotNull(registration);
        var owner = new java.util.concurrent.atomic.AtomicReference<Thread>();
        var follower = new java.util.concurrent.atomic.AtomicReference<Thread>();
        var interruptPreserved = new java.util.concurrent.atomic.AtomicBoolean();
        var closers = Executors.newFixedThreadPool(2);
        try {
            var first =
                    closers.submit(
                            () -> {
                                owner.set(Thread.currentThread());
                                try {
                                    responseExecutor.close();
                                    return null;
                                } catch (Throwable problem) {
                                    return problem;
                                }
                            });
            RequestProtocolTestSupport.awaitCondition(
                    () -> owner.get() != null && owner.get().getState() == Thread.State.WAITING);
            assertThrows(
                    IllegalStateException.class,
                    () -> SchedulerTestSupport.responsePublication(responseExecutor),
                    "close must reject new reservations while an earlier publication is pending");
            var second =
                    closers.submit(
                            () -> {
                                follower.set(Thread.currentThread());
                                Thread.currentThread().interrupt();
                                try {
                                    responseExecutor.close();
                                    return null;
                                } catch (Throwable problem) {
                                    return problem;
                                } finally {
                                    interruptPreserved.set(Thread.currentThread().isInterrupted());
                                }
                            });
            RequestProtocolTestSupport.awaitCondition(
                    () ->
                            follower.get() != null
                                    && follower.get().getState() == Thread.State.WAITING);
            assertFalse(first.isDone());
            assertFalse(second.isDone());
            registration.close();
            assertSame(failure, first.get(5, TimeUnit.SECONDS));
            assertSame(failure, second.get(5, TimeUnit.SECONDS));
            assertTrue(interruptPreserved.get());
            if (shutdownFails) {
                assertSame(
                        failure,
                        assertThrows(IllegalStateException.class, responseExecutor::close));
                verify(failedExecutor, org.mockito.Mockito.times(1)).shutdown();
            } else {
                responseExecutor.close();
                assertTrue(executor.isTerminated());
            }
        } finally {
            registration.close();
            closers.shutdownNow();
            assertTrue(closers.awaitTermination(5, TimeUnit.SECONDS));
            executor.shutdownNow();
            assertTrue(executor.awaitTermination(5, TimeUnit.SECONDS));
        }
    }

    @ParameterizedTest
    @EnumSource(TerminalForm.class)
    void externalFutureOperationUnderContextLockLeavesRequestUnchanged(TerminalForm form) {
        RequestContext requestContext =
                RequestProtocolTestSupport.context(SchedulingTestConfig.newConfig(), 702L);
        AbstractRequestScheduler requestOwner =
                RequestProtocolTestSupport.initialize(
                        mock(ResponseCompletionExecutor.class),
                        requestContext,
                        mock(ExpirationTimer.class));
        synchronized (requestContext) {
            assertThrows(
                    IllegalStateException.class,
                    () -> {
                        switch (form) {
                            case RESPONSE -> requestContext.getFuture().complete(new Response());
                            case FAILURE ->
                                    requestContext
                                            .getFuture()
                                            .completeExceptionally(
                                                    new IllegalStateException("failure"));
                            case CANCELLATION -> requestContext.getFuture().cancel(false);
                        }
                    });
            assertTrue(
                    requestContext.snapshot().phase() == RequestContext.Phase.OPEN
                            && requestContext.snapshot().cancellationReason() == null);
            assertEquals(
                    RequestResult.State.QUEUED,
                    RequestResult.from(requestContext.snapshot()).state());
            assertFalse(requestContext.getFuture().isDone());
        }
    }

    private enum TerminalForm {
        RESPONSE,
        FAILURE,
        CANCELLATION
    }

    private record Fixture(
            AbstractRequestScheduler scheduler,
            RequestContext requestContext,
            RequestResult claim,
            Registration publication,
            RequestResult candidate,
            Response response) {}
}
