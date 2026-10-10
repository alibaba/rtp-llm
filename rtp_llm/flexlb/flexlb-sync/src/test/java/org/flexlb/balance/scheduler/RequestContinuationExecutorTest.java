package org.flexlb.balance.scheduler;

import org.flexlb.config.FlexlbConfig;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RequestContinuationExecutorTest {

    @Test
    void continuationWorkerRetainsPlatformDaemonNameAndInheritedContext() throws Exception {
        InheritableThreadLocal<String> inherited = new InheritableThreadLocal<>();
        inherited.set("request-context");
        RequestContinuationExecutor executor = new RequestContinuationExecutor();
        CountDownLatch ran = new CountDownLatch(1);
        AtomicReference<Thread> worker = new AtomicReference<>();
        AtomicReference<String> observedContext = new AtomicReference<>();
        try {
            executor.submit(requestContext(993000L), () -> {
                worker.set(Thread.currentThread());
                observedContext.set(inherited.get());
                ran.countDown();
            });
            assertTrue(ran.await(5, TimeUnit.SECONDS));
            assertEquals("request-continuation-1", worker.get().getName());
            assertTrue(worker.get().isDaemon());
            assertFalse(worker.get().isVirtual());
            assertEquals("request-context", observedContext.get());
        } finally {
            executor.close();
            inherited.remove();
        }
    }

    @Test
    void sameRequestSerializesFactsWhileOtherRequestsCanProgressAndCloseDrains() throws Exception {
        RequestContext blockedContext = requestContext(993001L);
        RequestContext independentContext = requestContext(993002L);
        CountDownLatch started = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        CountDownLatch sameRequestFinished = new CountDownLatch(1);
        CountDownLatch independentFinished = new CountDownLatch(1);
        RequestContinuationExecutor executor = new RequestContinuationExecutor();
        Thread closer = null;
        try {
            executor.submit(blockedContext, () -> {
                started.countDown();
                try { assertTrue(release.await(5, TimeUnit.SECONDS)); }
                catch (InterruptedException error) { throw new AssertionError(error); }
            });
            assertTrue(started.await(5, TimeUnit.SECONDS));
            executor.submit(blockedContext, sameRequestFinished::countDown);
            executor.submit(independentContext, independentFinished::countDown);
            assertTrue(independentFinished.await(5, TimeUnit.SECONDS));
            assertFalse(sameRequestFinished.await(100, TimeUnit.MILLISECONDS));
            closer = new Thread(executor::close);
            closer.start();
            assertTrue(closer.isAlive());
            release.countDown();
            assertTrue(sameRequestFinished.await(5, TimeUnit.SECONDS));
            closer.join(5_000);
            assertFalse(closer.isAlive());
        } finally {
            release.countDown();
            executor.close();
            if (closer != null) { closer.join(5_000); }
        }
    }

    @Test
    void rejectedPoolTaskRecoversOffTheCallingThread() throws Exception {
        RequestContinuationExecutor executor = new RequestContinuationExecutor();
        ExecutorService workers = (ExecutorService) ReflectionTestUtils.getField(executor, "workers");
        workers.shutdown();
        CountDownLatch ran = new CountDownLatch(1);
        AtomicReference<String> threadName = new AtomicReference<>();
        try {
            executor.submit(requestContext(993003L), () -> {
                threadName.set(Thread.currentThread().getName());
                ran.countDown();
            });
            assertTrue(ran.await(5, TimeUnit.SECONDS));
            assertTrue(threadName.get().startsWith("request-continuation-recovery-"));
        } finally {
            executor.close();
        }
    }

    @Test
    void failedFactDoesNotPreventLaterFactOrDrain() throws Exception {
        RequestContinuationExecutor executor = new RequestContinuationExecutor();
        CountDownLatch later = new CountDownLatch(1);
        try {
            RequestContext requestContext = requestContext(993004L);
            executor.submit(requestContext, () -> { throw new IllegalStateException("isolated fact"); });
            executor.submit(requestContext, later::countDown);
            assertTrue(later.await(5, TimeUnit.SECONDS));
            executor.close();
        } finally {
            executor.close();
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.CsvSource({"false,false", "false,true", "true,false", "true,true"})
    @org.junit.jupiter.api.Timeout(15)
    void drainWaitsForRunningAndReentrantFacts(boolean close, boolean recovery) throws Exception {
        RequestContinuationExecutor executor = new RequestContinuationExecutor();
        if (recovery) {
            ((ExecutorService) ReflectionTestUtils.getField(executor, "workers")).shutdown();
        }
        RequestContext context = requestContext(993005L);
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        CountDownLatch nestedEntered = new CountDownLatch(1);
        CountDownLatch releaseNested = new CountDownLatch(1);
        var interruptsPreserved = new java.util.concurrent.atomic.AtomicInteger();
        Runnable waitForDrain = () -> {
            Thread.currentThread().interrupt();
            if (close) { executor.close(); } else { executor.awaitIdle(); }
            if (Thread.currentThread().isInterrupted()) { interruptsPreserved.incrementAndGet(); }
        };
        Thread waiter = new Thread(waitForDrain);
        Thread concurrentWaiter = new Thread(waitForDrain);
        try {
            executor.submit(context, () -> {
                entered.countDown();
                RequestProtocolTestSupport.await(release);
                executor.submit(context, () -> {
                    nestedEntered.countDown();
                    RequestProtocolTestSupport.await(releaseNested);
                });
            });
            assertTrue(entered.await(5, TimeUnit.SECONDS));
            waiter.start();
            concurrentWaiter.start();
            RequestProtocolTestSupport.awaitCondition(() -> waiter.getState() == Thread.State.WAITING);
            RequestProtocolTestSupport.awaitCondition(() -> concurrentWaiter.getState() == Thread.State.WAITING);
            assertTrue(waiter.isAlive(), "the popped but executing fact still owns its queue entry");
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
                assertTrue(((ExecutorService) ReflectionTestUtils.getField(executor, "workers")).isTerminated());
                org.junit.jupiter.api.Assertions.assertThrows(IllegalStateException.class,
                        () -> executor.submit(context, () -> { }));
            } else {
                CountDownLatch reopened = new CountDownLatch(1);
                executor.submit(context, reopened::countDown);
                assertTrue(reopened.await(5, TimeUnit.SECONDS), "an idle request can acquire a fresh drain owner");
            }
        } finally {
            release.countDown();
            releaseNested.countDown();
            executor.close();
            waiter.join(5_000);
            concurrentWaiter.join(5_000);
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void nestedFactRunsAfterAlreadyQueuedFactsAndNeverUnderContextMonitor(boolean recovery) throws Exception {
        RequestContinuationExecutor executor = new RequestContinuationExecutor();
        if (recovery) {
            ((ExecutorService) ReflectionTestUtils.getField(executor, "workers")).shutdown();
        }
        RequestContext context = requestContext(993006L);
        var observed = new java.util.concurrent.CopyOnWriteArrayList<Integer>();
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        CountDownLatch nested = new CountDownLatch(1);
        try {
            executor.submit(context, () -> {
                assertFalse(Thread.holdsLock(context));
                observed.add(1);
                entered.countDown();
                RequestProtocolTestSupport.await(release);
                executor.submit(context, () -> {
                    assertFalse(Thread.holdsLock(context));
                    observed.add(3);
                    nested.countDown();
                });
            });
            assertTrue(entered.await(5, TimeUnit.SECONDS));
            executor.submit(context, () -> {
                assertFalse(Thread.holdsLock(context));
                observed.add(2);
            });
            release.countDown();
            assertTrue(nested.await(5, TimeUnit.SECONDS));
            executor.awaitIdle();
            assertEquals(java.util.List.of(1, 2, 3), observed);
        } finally {
            release.countDown();
            executor.close();
        }
    }

    private static RequestContext requestContext(long requestId) {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        RequestContext context = RequestProtocolTestSupport.context(config, requestId);
        var owner = org.mockito.Mockito.mock(AbstractRequestScheduler.class);
        context.bindScheduler(owner);
        return context;
    }
}
