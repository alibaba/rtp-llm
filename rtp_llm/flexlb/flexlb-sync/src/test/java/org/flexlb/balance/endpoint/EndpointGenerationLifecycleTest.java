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
    void finalPermitReleaseWakesTheSingleCleanupOwnerExactlyOnce()
            throws Exception {
        EndpointGenerationLifecycle lifecycle = new EndpointGenerationLifecycle();
        EndpointGenerationLifecycle.HandoffPermit accepted =
                lifecycle.tryAcquireHandoff();
        assertNotNull(accepted);
        lifecycle.beginRetirement();
        assertTrue(lifecycle.tryClaimCleanup());
        assertFalse(lifecycle.tryClaimCleanup());

        AtomicInteger actionRuns = new AtomicInteger();
        AtomicReference<Thread> actionThread = new AtomicReference<>();
        CountDownLatch actionRan = new CountDownLatch(1);
        ExecutorService executor = Executors.newFixedThreadPool(2);
        try {
            Future<?> cleanup = executor.submit(() -> {
                lifecycle.beginCleanup();
                lifecycle.awaitHandoffs();
                actionThread.set(Thread.currentThread());
                actionRuns.incrementAndGet();
                lifecycle.completeRetirement(null);
                actionRan.countDown();
            });
            assertEquals(0, actionRuns.get(),
                    "cleanup must wait for the accepted permit");

            CountDownLatch waiterStarted = new CountDownLatch(1);
            Future<?> concurrentClose = executor.submit(() -> {
                waiterStarted.countDown();
                lifecycle.awaitRetirement();
            });
            assertTrue(waiterStarted.await(1, TimeUnit.SECONDS));
            assertFalse(concurrentClose.isDone());

            accepted.close();
            assertTrue(actionRan.await(1, TimeUnit.SECONDS));
            cleanup.get(1, TimeUnit.SECONDS);
            concurrentClose.get(1, TimeUnit.SECONDS);
            assertEquals(1, actionRuns.get());
            assertNotNull(actionThread.get());

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
        lifecycle.beginRetirement();
        assertTrue(lifecycle.tryClaimCleanup());
        assertNull(lifecycle.tryAcquireHandoff());

        ExecutorService executor = Executors.newFixedThreadPool(2);
        try {
            Future<?> retiringOwner = executor.submit(() -> {
                lifecycle.beginCleanup();
                lifecycle.awaitHandoffs();
                lifecycle.completeRetirement(null);
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
        lifecycle.beginRetirement();
        assertTrue(lifecycle.tryClaimCleanup());

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

            lifecycle.beginCleanup();
            lifecycle.awaitHandoffs();
            assertFalse(concurrentClose.isDone());
            lifecycle.completeRetirement(null);
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
        lifecycle.beginRetirement();
        assertTrue(lifecycle.tryClaimCleanup());
        IllegalStateException failure = new IllegalStateException("retirement failed");

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

            lifecycle.beginCleanup();
            accepted.close();
            lifecycle.awaitHandoffs();
            lifecycle.completeRetirement(failure);
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
