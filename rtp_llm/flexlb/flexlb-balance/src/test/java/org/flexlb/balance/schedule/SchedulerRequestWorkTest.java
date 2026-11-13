package org.flexlb.balance.schedule;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.rpc.EngineGrpcClient;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Exercises accepted request work and exact-request continuations while shutdown drains their owners.
 */
class SchedulerRequestWorkTest {

    @Test
    void workerRetainsPlatformDaemonNameAndInheritedContext() throws Exception {
        InheritableThreadLocal<String> inherited = new InheritableThreadLocal<>();
        inherited.set("request-context");
        SchedulerTestServices executor = runtime();
        CountDownLatch ran = new CountDownLatch(1);
        AtomicReference<Thread> worker = new AtomicReference<>();
        AtomicReference<String> observedContext = new AtomicReference<>();
        try {
            executor.tasks()
                    .execute(
                            () -> {
                                worker.set(Thread.currentThread());
                                observedContext.set(inherited.get());
                                ran.countDown();
                            });
            assertTrue(ran.await(5, TimeUnit.SECONDS));
            assertEquals("request-work-1", worker.get().getName());
            assertTrue(worker.get().isDaemon());
            assertFalse(worker.get().isVirtual());
            assertEquals("request-context", observedContext.get());
        } finally {
            close(executor);
            inherited.remove();
        }
    }

    @Test
    void sameRequestAndOtherRequestsProgressConcurrentlyWhileCloseWaitsForEveryOwner()
            throws Exception {
        RequestContext blockedContext = requestContext(993001L);
        RequestContext independentContext = requestContext(993002L);
        CountDownLatch started = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        CountDownLatch sameRequestFinished = new CountDownLatch(1);
        CountDownLatch independentFinished = new CountDownLatch(1);
        SchedulerTestServices executor = runtime();
        Thread closer = null;
        try {
            executor.tasks()
                    .execute(
                            () -> {
                                started.countDown();
                                try {
                                    assertTrue(release.await(5, TimeUnit.SECONDS));
                                } catch (InterruptedException error) {
                                    throw new AssertionError(error);
                                }
                            });
            assertTrue(started.await(5, TimeUnit.SECONDS));
            executor.tasks().execute(sameRequestFinished::countDown);
            executor.tasks().execute(independentFinished::countDown);
            assertTrue(independentFinished.await(5, TimeUnit.SECONDS));
            assertTrue(
                    sameRequestFinished.await(5, TimeUnit.SECONDS),
                    "same-request effects do not share a FIFO");
            closer = new Thread(() -> SchedulerTestSupport.closeRequestWork(executor));
            closer.start();
            assertTrue(closer.isAlive());
            release.countDown();
            assertTrue(sameRequestFinished.await(5, TimeUnit.SECONDS));
            closer.join(5_000);
            assertFalse(closer.isAlive());
        } finally {
            release.countDown();
            close(executor);
            if (closer != null) {
                closer.join(5_000);
            }
        }
    }

    @Test
    void failedFactDoesNotPreventLaterFactOrDrain() throws Exception {
        SchedulerTestServices executor = runtime();
        CountDownLatch later = new CountDownLatch(1);
        try {
            RequestContext requestContext = requestContext(993004L);
            executor.tasks()
                    .execute(
                            () -> {
                                throw new IllegalStateException("isolated fact");
                            });
            executor.tasks().execute(later::countDown);
            assertTrue(later.await(5, TimeUnit.SECONDS));
            SchedulerTestSupport.closeRequestWork(executor);
            org.junit.jupiter.api.Assertions.assertDoesNotThrow(
                    () -> SchedulerTestServices.throwIfFailed(executor.tasks));
        } finally {
            close(executor);
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    @org.junit.jupiter.api.Timeout(15)
    void drainWaitsForRunningAndReentrantFacts(boolean close) throws Exception {
        SchedulerTestServices executor = runtime();
        RequestContext context = requestContext(993005L);
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        CountDownLatch nestedEntered = new CountDownLatch(1);
        CountDownLatch releaseNested = new CountDownLatch(1);
        var interruptsPreserved = new java.util.concurrent.atomic.AtomicInteger();
        Runnable waitForDrain =
                () -> {
                    Thread.currentThread().interrupt();
                    if (close) {
                        SchedulerTestSupport.closeRequestWork(executor);
                    } else {
                        SchedulerTestSupport.awaitRequestWork(executor);
                    }
                    if (Thread.currentThread().isInterrupted()) {
                        interruptsPreserved.incrementAndGet();
                    }
                };
        Thread waiter = new Thread(waitForDrain);
        Thread concurrentWaiter = new Thread(waitForDrain);
        try {
            executor.tasks()
                    .execute(
                            () -> {
                                entered.countDown();
                                RequestProtocolTestSupport.await(release);
                                executor.tasks()
                                        .execute(
                                                () -> {
                                                    nestedEntered.countDown();
                                                    RequestProtocolTestSupport.await(releaseNested);
                                                });
                            });
            assertTrue(entered.await(5, TimeUnit.SECONDS));
            waiter.start();
            concurrentWaiter.start();
            RequestProtocolTestSupport.awaitCondition(
                    () -> waiter.getState() == Thread.State.WAITING);
            RequestProtocolTestSupport.awaitCondition(
                    () -> concurrentWaiter.getState() == Thread.State.WAITING);
            assertTrue(waiter.isAlive(), "running work still owns its registration");
            release.countDown();
            assertTrue(
                    nestedEntered.await(5, TimeUnit.SECONDS),
                    "closing must admit nested facts until drain");
            assertTrue(waiter.isAlive(), "the nested fact also prevents an early drain");
            assertTrue(concurrentWaiter.isAlive(), "every closer must await the nested fact");
            releaseNested.countDown();
            waiter.join(5_000);
            concurrentWaiter.join(5_000);
            assertFalse(waiter.isAlive());
            assertFalse(concurrentWaiter.isAlive());
            assertEquals(2, interruptsPreserved.get());
            if (close) {
                assertTrue(
                        ((ExecutorService)
                                        ReflectionTestUtils.getField(
                                                executor.tasks(), "requestWorkers"))
                                .isTerminated());
                org.junit.jupiter.api.Assertions.assertThrows(
                        java.util.concurrent.RejectedExecutionException.class,
                        () -> executor.tasks().execute(() -> {}));
            } else {
                CountDownLatch reopened = new CountDownLatch(1);
                executor.tasks().execute(reopened::countDown);
                assertTrue(
                        reopened.await(5, TimeUnit.SECONDS),
                        "an idle runtime can accept more work");
            }
        } finally {
            release.countDown();
            releaseNested.countDown();
            close(executor);
            waiter.join(5_000);
            concurrentWaiter.join(5_000);
        }
    }

    @org.junit.jupiter.api.Test
    void nestedWorkRunsOutsideContextMonitorAndIsIncludedInDrain() throws Exception {
        SchedulerTestServices executor = runtime();
        RequestContext context = requestContext(993006L);
        var observed = new java.util.concurrent.CopyOnWriteArrayList<Integer>();
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        CountDownLatch nested = new CountDownLatch(1);
        try {
            executor.tasks()
                    .execute(
                            () -> {
                                assertFalse(Thread.holdsLock(context));
                                observed.add(1);
                                entered.countDown();
                                RequestProtocolTestSupport.await(release);
                                executor.tasks()
                                        .execute(
                                                () -> {
                                                    assertFalse(Thread.holdsLock(context));
                                                    observed.add(3);
                                                    nested.countDown();
                                                });
                            });
            assertTrue(entered.await(5, TimeUnit.SECONDS));
            executor.tasks()
                    .execute(
                            () -> {
                                assertFalse(Thread.holdsLock(context));
                                observed.add(2);
                            });
            release.countDown();
            assertTrue(nested.await(5, TimeUnit.SECONDS));
            SchedulerTestSupport.awaitRequestWork(executor);
            assertEquals(java.util.Set.of(1, 2, 3), new java.util.HashSet<>(observed));
        } finally {
            release.countDown();
            close(executor);
        }
    }

    @Test
    void rejectedRequestPoolIsReportedToTheCaller() {
        var runtime = runtime();
        SchedulerTestSupport.requestWorkers(runtime).shutdown();
        try {
            assertThrows(RuntimeException.class, () -> runtime.tasks().execute(() -> {}));
        } finally {
            close(runtime);
        }
    }

    private static SchedulerTestServices runtime() {
        var service = org.mockito.Mockito.mock(org.flexlb.config.ConfigService.class);
        org.mockito.Mockito.when(service.loadBalanceConfig())
                .thenReturn(SchedulingTestConfig.newConfig());
        return new SchedulerTestServices(
                new RequestRepository(),
                org.mockito.Mockito.mock(EndpointRegistry.class),
                org.mockito.Mockito.mock(org.flexlb.balance.metrics.DeliveryMetricsReporter.class),
                org.mockito.Mockito.mock(org.flexlb.balance.metrics.RequestSchedulerReporter.class),
                org.mockito.Mockito.mock(EngineCancellation.class),
                service,
                org.mockito.Mockito.mock(EngineGrpcClient.class));
    }

    private static void close(SchedulerTestServices runtime) {
        SchedulerTestSupport.timer(runtime).close();
        SchedulerTestSupport.closeRequestExecutors(runtime);
    }

    private static RequestContext requestContext(long requestId) {
        FlexlbConfig config = SchedulingTestConfig.newConfig();
        RequestContext context = RequestProtocolTestSupport.context(config, requestId);
        var owner = org.mockito.Mockito.mock(AbstractRequestScheduler.class);
        SchedulerTestSupport.bindOwner(context, owner);
        return context;
    }
}
