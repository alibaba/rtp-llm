package org.flexlb.balance.endpoint;

import org.junit.jupiter.api.Test;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.flexlb.balance.endpoint.EndpointTestSupport.runWithStateLock;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class WorkerEndpointRetirementTest {
    @Test
    void concurrentCloseCleansOnceAndCompletesAfterNotifications() throws Exception {
        var started = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        var cleanups = new AtomicInteger();
        var completions = new AtomicInteger();
        var endpoint =
                new EndpointRetirementTestSupport(
                        () -> {
                            cleanups.incrementAndGet();
                            started.countDown();
                            com.google.common.util.concurrent.Uninterruptibles.awaitUninterruptibly(
                                    release);
                        });
        endpoint.onRetired(completions::incrementAndGet);
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            var closing = executor.submit(endpoint::close);
            try {
                assertTrue(started.await(2, TimeUnit.SECONDS));
                assertTrue(endpoint.isRetired());
                for (int i = 0; i < 20; i++)
                    executor.submit(endpoint::close).get(2, TimeUnit.SECONDS);
                assertEquals(0, completions.get());
                assertThrows(IllegalStateException.class, () -> endpoint.awaitRetirement(25));
            } finally {
                release.countDown();
            }
            closing.get(2, TimeUnit.SECONDS);
            endpoint.awaitRetirement(2000);
            assertEquals(1, cleanups.get());
            assertEquals(1, completions.get());
        }
    }

    @Test
    void failedCleanupIsReportedToEveryWaiterAndDoesNotFinalize() {
        var failure = new IllegalStateException("cleanup failed");
        var completions = new AtomicInteger();
        var endpoint =
                new EndpointRetirementTestSupport(
                        () -> {
                            throw failure;
                        });
        endpoint.onRetired(completions::incrementAndGet);
        assertSame(failure, assertThrows(IllegalStateException.class, endpoint::close));
        for (int i = 0; i < 2; i++) {
            assertSame(
                    failure,
                    assertThrows(
                            IllegalStateException.class, () -> endpoint.awaitRetirement(2000)));
        }
        assertEquals(0, completions.get());
        assertTrue(endpoint.isRetired());
    }

    @Test
    void closingFromAnEventRunsCleanupOutsideThatEvent() throws Exception {
        var cleanupOutsideEvent = new CompletableFuture<Boolean>();
        var ref = new java.util.concurrent.atomic.AtomicReference<WorkerEndpoint>();
        var endpoint =
                new EndpointRetirementTestSupport(
                        () ->
                                cleanupOutsideEvent.complete(
                                        !ref.get().stateLock().isHeldByCurrentThread()));
        ref.set(endpoint);
        runWithStateLock(endpoint.stateLock(), endpoint::close);
        assertTrue(cleanupOutsideEvent.get(2, TimeUnit.SECONDS));
        endpoint.awaitRetirement(2000);
    }
}
