package org.flexlb.balance.scheduler;

import org.flexlb.config.FlexlbConfig;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SchedulerRequestWorkTest {

    @Test
    void workerRetainsPlatformDaemonNameAndInheritedContext() throws Exception {
        InheritableThreadLocal<String> inherited = new InheritableThreadLocal<>();
        inherited.set("request-context");
        SchedulerRuntime executor = runtime();
        CountDownLatch ran = new CountDownLatch(1);
        AtomicReference<Thread> worker = new AtomicReference<>();
        AtomicReference<String> observedContext = new AtomicReference<>();
        try {
            executor.executeRequestWork(requestContext(993000L), () -> {
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
    void sameRequestAndOtherRequestsProgressConcurrentlyWhileCloseWaitsForEveryOwner() throws Exception {
        RequestContext blockedContext = requestContext(993001L);
        RequestContext independentContext = requestContext(993002L);
        CountDownLatch started = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        CountDownLatch sameRequestFinished = new CountDownLatch(1);
        CountDownLatch independentFinished = new CountDownLatch(1);
        SchedulerRuntime executor = runtime();
        Thread closer = null;
        try {
            executor.executeRequestWork(blockedContext, () -> {
                started.countDown();
                try { assertTrue(release.await(5, TimeUnit.SECONDS)); }
                catch (InterruptedException error) { throw new AssertionError(error); }
            });
            assertTrue(started.await(5, TimeUnit.SECONDS));
            executor.executeRequestWork(blockedContext, sameRequestFinished::countDown);
            executor.executeRequestWork(independentContext, independentFinished::countDown);
            assertTrue(independentFinished.await(5, TimeUnit.SECONDS));
            assertTrue(sameRequestFinished.await(5, TimeUnit.SECONDS), "same-request effects do not share a FIFO");
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
            if (closer != null) { closer.join(5_000); }
        }
    }

    @Test
    void rejectedPoolTaskRecoversOffTheCallingThread() throws Exception {
        SchedulerRuntime executor = runtime();
        ExecutorService workers = (ExecutorService) ReflectionTestUtils.getField(executor, "requestWorkers");
        workers.shutdown();
        CountDownLatch ran = new CountDownLatch(1);
        AtomicReference<String> threadName = new AtomicReference<>();
        try {
            executor.executeRequestWork(requestContext(993003L), () -> {
                threadName.set(Thread.currentThread().getName());
                ran.countDown();
            });
            assertTrue(ran.await(5, TimeUnit.SECONDS));
            assertTrue(threadName.get().startsWith("request-work-recovery-"));
        } finally {
            close(executor);
        }
    }

    @Test
    void failedFactDoesNotPreventLaterFactOrDrain() throws Exception {
        SchedulerRuntime executor = runtime();
        CountDownLatch later = new CountDownLatch(1);
        try {
            RequestContext requestContext = requestContext(993004L);
            executor.executeRequestWork(requestContext, () -> { throw new IllegalStateException("isolated fact"); });
            executor.executeRequestWork(requestContext, later::countDown);
            assertTrue(later.await(5, TimeUnit.SECONDS));
            SchedulerTestSupport.closeRequestWork(executor);
            assertEquals("isolated fact", ((Throwable) ReflectionTestUtils.getField(executor, "failure")).getMessage());
        } finally {
            close(executor);
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.CsvSource({"false,false", "false,true", "true,false", "true,true"})
    @org.junit.jupiter.api.Timeout(15)
    void drainWaitsForRunningAndReentrantFacts(boolean close, boolean recovery) throws Exception {
        SchedulerRuntime executor = runtime();
        if (recovery) {
            ((ExecutorService) ReflectionTestUtils.getField(executor, "requestWorkers")).shutdown();
        }
        RequestContext context = requestContext(993005L);
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        CountDownLatch nestedEntered = new CountDownLatch(1);
        CountDownLatch releaseNested = new CountDownLatch(1);
        var interruptsPreserved = new java.util.concurrent.atomic.AtomicInteger();
        Runnable waitForDrain = () -> {
            Thread.currentThread().interrupt();
            if (close) { SchedulerTestSupport.closeRequestWork(executor); }
            else { SchedulerTestSupport.awaitRequestWork(executor); }
            if (Thread.currentThread().isInterrupted()) { interruptsPreserved.incrementAndGet(); }
        };
        Thread waiter = new Thread(waitForDrain);
        Thread concurrentWaiter = new Thread(waitForDrain);
        try {
            executor.executeRequestWork(context, () -> {
                entered.countDown();
                RequestProtocolTestSupport.await(release);
                executor.executeRequestWork(context, () -> {
                    nestedEntered.countDown();
                    RequestProtocolTestSupport.await(releaseNested);
                });
            });
            assertTrue(entered.await(5, TimeUnit.SECONDS));
            waiter.start();
            concurrentWaiter.start();
            RequestProtocolTestSupport.awaitCondition(() -> waiter.getState() == Thread.State.WAITING);
            RequestProtocolTestSupport.awaitCondition(() -> concurrentWaiter.getState() == Thread.State.WAITING);
            assertTrue(waiter.isAlive(), "running work still owns its registration");
            release.countDown();
            assertTrue(nestedEntered.await(5, TimeUnit.SECONDS), "closing must admit nested facts until drain");
            assertTrue(waiter.isAlive(), "the nested fact also prevents an early drain");
            assertTrue(concurrentWaiter.isAlive(), "every closer must await the nested fact");
            releaseNested.countDown();
            waiter.join(5_000);
            concurrentWaiter.join(5_000);
            assertFalse(waiter.isAlive());
            assertFalse(concurrentWaiter.isAlive());
            assertEquals(2, interruptsPreserved.get());
            if (close) {
                assertTrue(((ExecutorService) ReflectionTestUtils.getField(executor, "requestWorkers")).isTerminated());
                org.junit.jupiter.api.Assertions.assertThrows(IllegalStateException.class,
                        () -> executor.executeRequestWork(context, () -> { }));
            } else {
                CountDownLatch reopened = new CountDownLatch(1);
                executor.executeRequestWork(context, reopened::countDown);
                assertTrue(reopened.await(5, TimeUnit.SECONDS), "an idle runtime can accept more work");
            }
        } finally {
            release.countDown();
            releaseNested.countDown();
            close(executor);
            waiter.join(5_000);
            concurrentWaiter.join(5_000);
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void nestedWorkRunsOutsideContextMonitorAndIsIncludedInDrain(boolean recovery) throws Exception {
        SchedulerRuntime executor = runtime();
        if (recovery) {
            ((ExecutorService) ReflectionTestUtils.getField(executor, "requestWorkers")).shutdown();
        }
        RequestContext context = requestContext(993006L);
        var observed = new java.util.concurrent.CopyOnWriteArrayList<Integer>();
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        CountDownLatch nested = new CountDownLatch(1);
        try {
            executor.executeRequestWork(context, () -> {
                assertFalse(Thread.holdsLock(context));
                observed.add(1);
                entered.countDown();
                RequestProtocolTestSupport.await(release);
                executor.executeRequestWork(context, () -> {
                    assertFalse(Thread.holdsLock(context));
                    observed.add(3);
                    nested.countDown();
                });
            });
            assertTrue(entered.await(5, TimeUnit.SECONDS));
            executor.executeRequestWork(context, () -> {
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
    void closeWaitsForRegistrationBeforeSubmissionAndForSourceAfterTaskFinishes() throws Exception {
        var runtime = runtime();
        var context = requestContext(993007L);
        var registration = runtime.registerRequestWork(context);
        var ran = new CountDownLatch(1);
        var closer = new Thread(() -> SchedulerTestSupport.closeRequestWork(runtime));
        try {
            closer.start();
            RequestProtocolTestSupport.awaitCondition(() -> closer.getState() == Thread.State.WAITING);
            assertTrue(closer.isAlive(), "accepted but not yet submitted work is included");
            registration.submit(ran::countDown);
            assertTrue(ran.await(5, TimeUnit.SECONDS));
            assertTrue(closer.isAlive(), "the source still owns its handoff even after the worker returns");
            registration.close();
            closer.join(5_000);
            assertFalse(closer.isAlive());
        } finally {
            registration.close();
            closer.join(5_000);
            close(runtime);
        }
    }

    @Test
    void rejectionByBothPoolsPreservesAcceptedWorkOffCallerAndRecordsTheFailure() throws Exception {
        var runtime = runtime();
        SchedulerTestSupport.requestWorkers(runtime).shutdown();
        ((ExecutorService) ReflectionTestUtils.getField(runtime, "requestRecovery")).shutdown();
        var worker = new AtomicReference<Thread>();
        try {
            runtime.executeRequestWork(requestContext(993008L), () -> worker.set(Thread.currentThread()));
            SchedulerTestSupport.awaitRequestWork(runtime);
            org.junit.jupiter.api.Assertions.assertNotNull(worker.get());
            org.junit.jupiter.api.Assertions.assertNotSame(Thread.currentThread(), worker.get());
            org.junit.jupiter.api.Assertions.assertNotNull(ReflectionTestUtils.getField(runtime, "failure"));
        } finally { close(runtime); }
    }

    @Test
    void acceptedOwnerMayHandOffExactRequestCleanupDuringCloseButForeignSourcesCannot() throws Exception {
        var runtime = runtime();
        var context = requestContext(993009L);
        var foreign = requestContext(993010L);
        var entered = new CountDownLatch(1);
        var handoff = new CountDownLatch(1);
        var childEntered = new CountDownLatch(1);
        var releaseChild = new CountDownLatch(1);
        var parentDone = new CountDownLatch(1);
        var failure = new AtomicReference<Throwable>();
        var closer = new Thread(() -> SchedulerTestSupport.closeRequestWork(runtime));
        try {
            runtime.executeRequestWork(context, () -> {
                entered.countDown();
                RequestProtocolTestSupport.await(handoff);
                try {
                    assertStoppedSource(runtime, foreign);
                    runtime.executeRequestWork(context, () -> {
                        childEntered.countDown();
                        RequestProtocolTestSupport.await(releaseChild);
                    });
                } catch (Throwable problem) { failure.set(problem); }
                finally { parentDone.countDown(); }
            });
            assertTrue(entered.await(5, TimeUnit.SECONDS));
            closer.start();
            RequestProtocolTestSupport.awaitCondition(() -> closer.getState() == Thread.State.WAITING);
            assertStoppedSource(runtime, context);
            assertStoppedSource(runtime, foreign);
            handoff.countDown();
            assertTrue(parentDone.await(5, TimeUnit.SECONDS));
            org.junit.jupiter.api.Assertions.assertNull(failure.get());
            assertTrue(childEntered.await(5, TimeUnit.SECONDS));
            assertTrue(closer.isAlive(), "close retains the handed-off cleanup after its parent returns");
            releaseChild.countDown();
            closer.join(5_000);
            assertFalse(closer.isAlive());
        } finally {
            handoff.countDown();
            releaseChild.countDown();
            closer.join(5_000);
            close(runtime);
        }
    }

    private static void assertStoppedSource(SchedulerRuntime runtime, RequestContext context) {
        var registration = runtime.tryRegisterRequestWork(context);
        try { org.junit.jupiter.api.Assertions.assertNull(registration); }
        finally { if (registration != null) { registration.close(); } }
    }

    private static SchedulerRuntime runtime() {
        var service = org.mockito.Mockito.mock(org.flexlb.config.ConfigService.class);
        org.mockito.Mockito.when(service.loadBalanceConfig()).thenReturn(SchedulingTestConfig.batchConfig());
        return new SchedulerRuntime(new RequestRepository(),
                org.mockito.Mockito.mock(org.flexlb.balance.endpoint.EndpointRegistry.class),
                org.mockito.Mockito.mock(org.flexlb.service.monitor.DeliveryMetricsReporter.class),
                org.mockito.Mockito.mock(org.flexlb.service.monitor.RequestSchedulerReporter.class),
                org.mockito.Mockito.mock(DeliveryDispatcher.class), service,
                org.mockito.Mockito.mock(org.flexlb.balance.eviction.EngineCancelChannel.class));
    }

    private static void close(SchedulerRuntime runtime) {
        SchedulerTestSupport.timer(runtime).close();
        SchedulerTestSupport.closeRequestExecutors(runtime);
    }

    private static RequestContext requestContext(long requestId) {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        RequestContext context = RequestProtocolTestSupport.context(config, requestId);
        var owner = org.mockito.Mockito.mock(AbstractRequestScheduler.class);
        SchedulerTestSupport.bindOwner(context, owner);
        return context;
    }
}
