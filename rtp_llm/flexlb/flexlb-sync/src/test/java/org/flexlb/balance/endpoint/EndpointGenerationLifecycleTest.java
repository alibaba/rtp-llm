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
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EndpointGenerationLifecycleTest {

    @Test
    void finalPermitReleaseRunsTheSingleCleanupContinuationExactlyOnce()
            throws Exception {
        AtomicInteger actionRuns = new AtomicInteger();
        AtomicReference<Thread> actionThread = new AtomicReference<>();
        CountDownLatch actionRan = new CountDownLatch(1);
        AtomicReference<EndpointGenerationLifecycle> lifecycleRef =
                new AtomicReference<>();
        EndpointGenerationLifecycle lifecycle = new EndpointGenerationLifecycle(
                () -> {
                    EndpointGenerationLifecycle exact = lifecycleRef.get();
                    exact.beginCleanup();
                    actionThread.set(Thread.currentThread());
                    actionRuns.incrementAndGet();
                    exact.completeRetirement(null);
                    actionRan.countDown();
                });
        lifecycleRef.set(lifecycle);
        EndpointGenerationLifecycle.HandoffPermit accepted =
                lifecycle.tryAcquireHandoff();
        EndpointGenerationLifecycle.HandoffPermit second =
                lifecycle.tryAcquireHandoff();
        assertNotNull(accepted);
        assertNotNull(second);
        lifecycle.beginRetirement();
        assertFalse(lifecycle.tryStartCleanup());
        assertFalse(lifecycle.tryStartCleanup());

        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
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
            assertEquals(0, actionRuns.get());
            assertFalse(concurrentClose.isDone());
            second.close();
            assertTrue(actionRan.await(1, TimeUnit.SECONDS));
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
    void handoffsDrainedBeforeCleanupClaimLeaveCleanupToTheCaller() {
        AtomicInteger continuations = new AtomicInteger();
        EndpointGenerationLifecycle lifecycle =
                new EndpointGenerationLifecycle(continuations::incrementAndGet);
        var accepted = lifecycle.tryAcquireHandoff();
        assertNotNull(accepted);
        lifecycle.beginRetirement();
        accepted.close();

        assertTrue(lifecycle.tryStartCleanup());
        assertFalse(lifecycle.tryStartCleanup());
        assertEquals(0, continuations.get());
        assertThrows(IllegalStateException.class, () -> lifecycle.completeRetirement(null));
        lifecycle.beginCleanup();
        assertThrows(IllegalStateException.class, lifecycle::beginCleanup);
        assertThrows(IllegalStateException.class, lifecycle::awaitRetirement);
        lifecycle.completeRetirement(null);
        lifecycle.awaitRetirement();
        assertFalse(lifecycle.tryStartCleanup());
    }

    @Test
    void emptyHandoffSetDoesNotMeanRetirementIsComplete() throws Exception {
        EndpointGenerationLifecycle lifecycle =
                new EndpointGenerationLifecycle(() -> { });
        lifecycle.beginRetirement();
        assertTrue(lifecycle.tryStartCleanup());

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
        EndpointGenerationLifecycle lifecycle =
                new EndpointGenerationLifecycle(() -> { });
        EndpointGenerationLifecycle.HandoffPermit accepted =
                lifecycle.tryAcquireHandoff();
        assertNotNull(accepted);
        lifecycle.beginRetirement();
        assertFalse(lifecycle.tryStartCleanup());
        IllegalStateException failure = new IllegalStateException("retirement failed");

        CountDownLatch waiterStarted = new CountDownLatch(1);
        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            Future<?> concurrentClose = executor.submit(() -> {
                Thread.currentThread().interrupt();
                waiterStarted.countDown();
                try {
                    lifecycle.awaitRetirement();
                } finally {
                    assertTrue(Thread.interrupted(), "retirement failure must restore interruption");
                }
            });
            assertTrue(waiterStarted.await(1, TimeUnit.SECONDS));
            assertFalse(concurrentClose.isDone(),
                    "a non-owner close must wait while an accepted handoff is active");

            accepted.close();
            lifecycle.beginCleanup();
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

    @Test
    void leakedHandoffProducesBoundedDiagnosticInsteadOfPermanentWait() {
        EndpointGenerationLifecycle lifecycle =
                new EndpointGenerationLifecycle(() -> { });
        EndpointGenerationLifecycle.HandoffPermit leaked =
                lifecycle.tryAcquireHandoff();
        assertNotNull(leaked);
        lifecycle.beginRetirement();
        assertFalse(lifecycle.tryStartCleanup());

        Thread.currentThread().interrupt();
        try {
            IllegalStateException timeout = assertThrows(
                    IllegalStateException.class,
                    () -> lifecycle.awaitRetirement(25L));
            assertTrue(timeout.getMessage().contains("activeHandoffs=1"));
            assertTrue(Thread.currentThread().isInterrupted(), "timeout must restore interruption");
        } finally {
            Thread.interrupted();
            leaked.close();
        }
    }
}
