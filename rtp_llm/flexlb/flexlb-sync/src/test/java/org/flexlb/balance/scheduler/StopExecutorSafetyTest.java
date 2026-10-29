package org.flexlb.balance.scheduler;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.RETURNS_DEEP_STUBS;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.anyInt;
import static org.mockito.Mockito.anyLong;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

@Timeout(15)
class StopExecutorSafetyTest {
    @Test
    void stopNotificationsCoalesceWithoutBlockingLocalSettlementForTheSameRequest()
            throws Exception {
        var runtime = runtime();
        var context = context();
        var entered = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        var resumed = new CountDownLatch(1);
        var locallySettled = new CountDownLatch(1);
        var calls = new java.util.concurrent.atomic.AtomicInteger();
        Runnable workflow =
                () -> {
                    if (calls.incrementAndGet() == 1) {
                        entered.countDown();
                        RequestProtocolTestSupport.await(release);
                    } else {
                        resumed.countDown();
                    }
                };
        try {
            assertTrue(runtime.executeStopWork(context, workflow));
            assertTrue(entered.await(2, TimeUnit.SECONDS));
            assertTrue(runtime.executeStopWork(context, workflow));
            for (int i = 0; i < 100; i++) {
                assertFalse(runtime.executeStopWork(context, workflow));
            }
            try (var local = runtime.registerRequestWork(context)) {
                local.submitClose(locallySettled::countDown, false);
            }
            assertTrue(locallySettled.await(2, TimeUnit.SECONDS));
            SchedulerTestSupport.awaitRequestWork(runtime);
            release.countDown();
            assertTrue(resumed.await(2, TimeUnit.SECONDS));
        } finally {
            release.countDown();
            runtime.shutdown(ShutdownMode.IMMEDIATE, java.time.Duration.ZERO);
        }
        assertEquals(2, calls.get());
    }

    @Test
    void shutdownRejectsNewStopsEvenWithoutAnExecutorService() {
        var runtime = runtime();
        runtime.shutdown(ShutdownMode.IMMEDIATE, java.time.Duration.ZERO);
        var started = new java.util.concurrent.atomic.AtomicBoolean();
        org.junit.jupiter.api.Assertions.assertThrows(
                IllegalStateException.class,
                () -> runtime.executeStopWork(context(), () -> started.set(true)));
        assertFalse(started.get());
    }

    @Test
    void waitingStopsDoNotOccupyNormalRequestWorkersOrLocalDrain() throws Exception {
        var runtime = runtime();
        var release = new CountDownLatch(1);
        int size = 32;
        var entered = new CountDownLatch(size);
        var normal = new CountDownLatch(1);
        try {
            for (int i = 0; i < size; i++) {
                runtime.executeStopWork(
                        context(),
                        () -> {
                            assertTrue(Thread.currentThread().isVirtual());
                            entered.countDown();
                            RequestProtocolTestSupport.await(release);
                        });
            }
            assertTrue(entered.await(3, TimeUnit.SECONDS));
            runtime.executeRequestWork(context(), normal::countDown);
            assertTrue(normal.await(1, TimeUnit.SECONDS));
            SchedulerTestSupport.awaitRequestWork(runtime);
        } finally {
            release.countDown();
            runtime.shutdown(ShutdownMode.IMMEDIATE, java.time.Duration.ZERO);
        }
    }

    @Test
    void normalPoolSaturationDoesNotPreventStopProgress() throws Exception {
        var runtime = runtime();
        var release = new CountDownLatch(1);
        var workers = (ThreadPoolExecutor) SchedulerTestSupport.requestWorkers(runtime);
        var entered = new CountDownLatch(workers.getMaximumPoolSize());
        var stop = new CountDownLatch(1);
        try {
            for (int i = 0; i < workers.getMaximumPoolSize(); i++) {
                runtime.executeRequestWork(
                        context(),
                        () -> {
                            entered.countDown();
                            RequestProtocolTestSupport.await(release);
                        });
            }
            assertTrue(entered.await(3, TimeUnit.SECONDS));
            runtime.executeStopWork(context(), stop::countDown);
            assertTrue(stop.await(1, TimeUnit.SECONDS));
        } finally {
            release.countDown();
            runtime.shutdown(ShutdownMode.IMMEDIATE, java.time.Duration.ZERO);
        }
    }

    @Test
    void shutdownWaitsForStopToRequestToStopHandoffWithoutClosingEitherPoolEarly()
            throws Exception {
        var runtime = runtime();
        var context = context();
        var entered = new CountDownLatch(1);
        var handoff = new CountDownLatch(1);
        var finalEntered = new CountDownLatch(1);
        var releaseFinal = new CountDownLatch(1);
        var failure = new AtomicReference<Throwable>();
        var closer =
                new Thread(
                        () -> {
                            try {
                                SchedulerTestSupport.closeRequestWork(runtime);
                            } catch (Throwable e) {
                                failure.set(e);
                            }
                        });
        try {
            runtime.executeStopWork(
                    context,
                    () -> {
                        entered.countDown();
                        RequestProtocolTestSupport.await(handoff);
                        runtime.executeRequestWork(
                                context,
                                () ->
                                        runtime.executeStopWork(
                                                context,
                                                () -> {
                                                    finalEntered.countDown();
                                                    RequestProtocolTestSupport.await(releaseFinal);
                                                }));
                    });
            assertTrue(entered.await(3, TimeUnit.SECONDS));
            closer.start();
            RequestProtocolTestSupport.awaitCondition(
                    () -> closer.getState() == Thread.State.WAITING);
            handoff.countDown();
            assertTrue(finalEntered.await(3, TimeUnit.SECONDS));
            assertTrue(closer.isAlive());
            releaseFinal.countDown();
            closer.join(3000);
            assertFalse(closer.isAlive());
            assertNull(failure.get());
        } finally {
            handoff.countDown();
            releaseFinal.countDown();
            closer.join(3000);
            runtime.shutdown(ShutdownMode.IMMEDIATE, java.time.Duration.ZERO);
        }
    }

    @Test
    void stopCallbackCannotSynchronouslyShutDownItsOwnRuntime() throws Exception {
        var runtime = runtime();
        var result = new CompletableFuture<Boolean>();
        try {
            runtime.executeStopWork(
                    context(),
                    () -> {
                        try {
                            runtime.shutdown(ShutdownMode.IMMEDIATE, java.time.Duration.ZERO);
                            result.complete(false);
                        } catch (IllegalStateException expected) {
                            result.complete(runtime.isAccepting());
                        }
                    });
            assertTrue(result.get(3, TimeUnit.SECONDS));
        } finally {
            runtime.shutdown(ShutdownMode.IMMEDIATE, java.time.Duration.ZERO);
        }
    }

    @Test
    void rejectedStopSubmissionIsReportedAndRemainsDrained() throws Exception {
        var runtime = runtime();
        ReflectionTestUtils.setField(
                runtime,
                "stopWorkers",
                (java.util.concurrent.Executor)
                        work -> {
                            throw new java.util.concurrent.RejectedExecutionException();
                        });
        try {
            assertThrows(
                    java.util.concurrent.RejectedExecutionException.class,
                    () -> runtime.executeStopWork(context(), () -> {}));
            SchedulerTestSupport.awaitRequestWork(runtime);
        } finally {
            assertThrows(RuntimeException.class, () -> runtime.shutdown(ShutdownMode.IMMEDIATE, java.time.Duration.ZERO));
        }
    }

    @Test
    void thirtyOutstandingRpcsDoNotOccupyEightStopWorkers() throws Exception {
        var runtime = runtime();
        var cleanup =
                new DeliveryDispatcher(
                        null,
                        DeliveryTestSupport.service(SchedulingTestConfig.batchConfig()),
                        new org.flexlb.service.monitor.DeliveryMetricsReporter(null));
        var client = runtime.engineClient;
        var rpcReplies =
                new java.util.concurrent.CopyOnWriteArrayList<
                        CompletableFuture<org.flexlb.engine.grpc.EngineGrpcClient.CancelReply>>();
        var entered = new CountDownLatch(30);
        var answers = new java.util.ArrayList<CompletableFuture<?>>();
        when(client.cancelAsync(any(), anyInt(), any(), anyLong()))
                .thenAnswer(
                        call -> {
                            var reply =
                                    new CompletableFuture<
                                            org.flexlb.engine.grpc.EngineGrpcClient.CancelReply>();
                            rpcReplies.add(reply);
                            entered.countDown();
                            return reply;
                        });
        try {
            for (int i = 0; i < 30; i++) {
                var context = context();
                ReflectionTestUtils.setField(context.data().getScheduler(), "runtime", runtime);
                when(context.snapshot().delivery()).thenReturn(null);
                when(context.snapshot().cancellation()).thenReturn(null);
                var claim =
                        new PreemptionRegistration(
                                context,
                                i,
                                new org.flexlb.balance.endpoint.DecodeResources.ReservationHandle(
                                        1, i + 1, i + 1),
                                new org.flexlb.balance.preemption.CancelTarget("127.0.0.1", 9000));
                answers.add(cleanup.preempt(claim, null, "priority preemption", 500L));
            }
            assertTrue(
                    entered.await(2, TimeUnit.SECONDS),
                    "unanswered RPCs must not retain worker threads");
            assertEquals(30, rpcReplies.size());
        } finally {
            rpcReplies.forEach(
                    reply ->
                            reply.complete(
                                    org.flexlb.engine.grpc.EngineGrpcClient.CancelReply
                                            .REQUEST_CLEANED));
            CompletableFuture.allOf(answers.toArray(CompletableFuture[]::new))
                    .get(3, TimeUnit.SECONDS);
            runtime.shutdown(ShutdownMode.IMMEDIATE, java.time.Duration.ZERO);
        }
    }

    private RequestContext context() {
        return mock(RequestContext.class, RETURNS_DEEP_STUBS);
    }

    private SchedulerRuntime runtime() {
        var config = DeliveryTestSupport.service(SchedulingTestConfig.batchConfig());
        return new SchedulerRuntime(
                new RequestRepository(),
                mock(org.flexlb.balance.endpoint.EndpointRegistry.class),
                mock(org.flexlb.service.monitor.DeliveryMetricsReporter.class),
                mock(org.flexlb.service.monitor.RequestSchedulerReporter.class),
                new DeliveryDispatcher(
                        null, config, new org.flexlb.service.monitor.DeliveryMetricsReporter(null)),
                config,
                mock(org.flexlb.engine.grpc.EngineGrpcClient.class));
    }
}
