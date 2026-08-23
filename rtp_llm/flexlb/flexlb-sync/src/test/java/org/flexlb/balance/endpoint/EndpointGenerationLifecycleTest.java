package org.flexlb.balance.endpoint;

import org.junit.jupiter.api.Test;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EndpointGenerationLifecycleTest {

    @Test
    void finalPermitReleaseRunsDeferredRetirementActionExactlyOnce()
            throws Exception {
        EndpointGenerationLifecycle lifecycle = new EndpointGenerationLifecycle();
        EndpointGenerationLifecycle.HandoffPermit accepted =
                lifecycle.tryAcquireHandoff();
        assertNotNull(accepted);
        assertTrue(lifecycle.tryBeginRetirement());

        AtomicInteger actionRuns = new AtomicInteger();
        AtomicReference<Thread> actionThread = new AtomicReference<>();
        CountDownLatch actionRan = new CountDownLatch(1);
        ExecutorService executor = Executors.newFixedThreadPool(2);
        try {
            Future<?> actionInstallation = executor.submit(() ->
                    lifecycle.runWhenAcceptedHandoffsDrain(() -> {
                        actionThread.set(Thread.currentThread());
                        actionRuns.incrementAndGet();
                        lifecycle.completeRetirement();
                        actionRan.countDown();
                    }));
            actionInstallation.get(1, TimeUnit.SECONDS);
            assertEquals(0, actionRuns.get(),
                    "installing deferred retirement must not wait for the permit");

            CountDownLatch waiterStarted = new CountDownLatch(1);
            Future<?> concurrentClose = executor.submit(() -> {
                waiterStarted.countDown();
                lifecycle.awaitRetirement();
            });
            assertTrue(waiterStarted.await(1, TimeUnit.SECONDS));
            assertFalse(concurrentClose.isDone());

            Thread releasingThread = Thread.currentThread();
            accepted.close();
            assertTrue(actionRan.await(1, TimeUnit.SECONDS));
            concurrentClose.get(1, TimeUnit.SECONDS);
            assertEquals(1, actionRuns.get());
            assertSame(releasingThread, actionThread.get(),
                    "the final permit releaser owns deferred cleanup execution");

            accepted.close();
            assertEquals(1, actionRuns.get(),
                    "a permit and its retirement action are both exactly-once");
        } finally {
            executor.shutdownNow();
        }
    }

    @Test
    void retirementWaitsForAcceptedHandoffsAndGenerationCleanup() throws Exception {
        EndpointGenerationLifecycle lifecycle = new EndpointGenerationLifecycle();
        EndpointGenerationLifecycle.HandoffPermit accepted =
                lifecycle.tryAcquireHandoff();
        assertNotNull(accepted);
        assertTrue(lifecycle.tryBeginRetirement());
        assertNull(lifecycle.tryAcquireHandoff());

        ExecutorService executor = Executors.newFixedThreadPool(2);
        try {
            Future<?> retiringOwner = executor.submit(() -> {
                lifecycle.awaitAcceptedHandoffs();
                lifecycle.completeRetirement();
            });
            CountDownLatch waiterStarted = new CountDownLatch(1);
            Future<?> concurrentClose = executor.submit(() -> {
                waiterStarted.countDown();
                lifecycle.awaitRetirement();
            });

            assertTrue(waiterStarted.await(1, TimeUnit.SECONDS));
            assertFalse(retiringOwner.isDone());
            assertFalse(concurrentClose.isDone());

            accepted.close();
            retiringOwner.get(1, TimeUnit.SECONDS);
            concurrentClose.get(1, TimeUnit.SECONDS);
        } finally {
            executor.shutdownNow();
        }
    }

    @Test
    void emptyHandoffSetDoesNotMeanRetirementIsComplete() throws Exception {
        EndpointGenerationLifecycle lifecycle = new EndpointGenerationLifecycle();
        assertTrue(lifecycle.tryBeginRetirement());

        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            CountDownLatch waiterStarted = new CountDownLatch(1);
            Future<?> concurrentClose = executor.submit(() -> {
                waiterStarted.countDown();
                lifecycle.awaitRetirement();
            });

            assertTrue(waiterStarted.await(1, TimeUnit.SECONDS));
            assertFalse(concurrentClose.isDone(),
                    "close must wait for generation cleanup after handoffs reach zero");

            lifecycle.awaitAcceptedHandoffs();
            assertFalse(concurrentClose.isDone());
            lifecycle.completeRetirement();
            concurrentClose.get(1, TimeUnit.SECONDS);
        } finally {
            executor.shutdownNow();
        }
    }

    @Test
    void concurrentNonOwnerCloseWaitsAndObservesTheSameRetirementFailure()
            throws Exception {
        EndpointGenerationLifecycle lifecycle = new EndpointGenerationLifecycle();
        EndpointGenerationLifecycle.HandoffPermit accepted =
                lifecycle.tryAcquireHandoff();
        assertNotNull(accepted);
        assertTrue(lifecycle.tryBeginRetirement());
        IllegalStateException failure = new IllegalStateException("retirement failed");
        lifecycle.runWhenAcceptedHandoffsDrain(
                () -> lifecycle.completeRetirement(failure));

        CountDownLatch waiterStarted = new CountDownLatch(1);
        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            Future<?> concurrentClose = executor.submit(() -> {
                waiterStarted.countDown();
                lifecycle.awaitRetirement();
            });
            assertTrue(waiterStarted.await(1, TimeUnit.SECONDS));
            assertFalse(concurrentClose.isDone(),
                    "a non-owner close must wait while an accepted handoff is active");

            accepted.close();
            ExecutionException observed = assertThrows(
                    ExecutionException.class,
                    () -> concurrentClose.get(1, TimeUnit.SECONDS));
            assertSame(failure, observed.getCause());
        } finally {
            executor.shutdownNow();
        }

        IllegalStateException repeatedObservation = assertThrows(
                IllegalStateException.class, lifecycle::awaitRetirement);
        assertSame(failure, repeatedObservation);
    }
}
