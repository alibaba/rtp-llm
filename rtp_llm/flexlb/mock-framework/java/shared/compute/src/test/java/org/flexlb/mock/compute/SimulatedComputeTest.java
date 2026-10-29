package org.flexlb.mock.compute;

import static org.junit.jupiter.api.Assertions.*;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

class SimulatedComputeTest {
    private static NoiseModel.Sample sample(long nanos) {
        return new NoiseModel.Sample(Duration.ofNanos(nanos), 0, Duration.ofNanos(nanos));
    }

    @Test void waitsForPlannedTimeAndReportsActualElapsed() throws Exception {
        try (var compute = new SimulatedCompute()) {
            var execution = compute.submit(sample(5_000_000));
            var result = execution.completion().toCompletableFuture().get(2, TimeUnit.SECONDS);
            assertEquals(SimulatedCompute.State.COMPLETED, result.state());
            assertTrue(result.elapsedNanos() >= result.sample().planned().toNanos());
            assertFalse(execution.cancel());
            assertEquals(0, compute.activeCount());
        }
    }

    @Test void cancellationHasOneTerminalOutcomeAndDoesNotWaitForTheTimer() throws Exception {
        try (var compute = new SimulatedCompute()) {
            var execution = compute.submit(sample(Duration.ofHours(1).toNanos()));
            assertTrue(execution.cancel()); assertFalse(execution.cancel());
            assertEquals(SimulatedCompute.State.CANCELLED, execution.completion().toCompletableFuture().get(2, TimeUnit.SECONDS).state());
            assertEquals(0, compute.activeCount());
        }
    }

    @Test void completionCancellationRacesNeverCommitTwice() throws Exception {
        try (var compute = new SimulatedCompute(); var callers = Executors.newVirtualThreadPerTaskExecutor()) {
            var futures = new ArrayList<CompletableFuture<SimulatedCompute.Outcome>>();
            var counters = new ArrayList<AtomicInteger>();
            for (int i = 0; i < 500; i++) {
                var execution = compute.submit(sample(i % 2 == 0 ? 0 : 100_000));
                var count = new AtomicInteger(); counters.add(count);
                futures.add(execution.completion().thenApply(value -> { count.incrementAndGet(); return value; }).toCompletableFuture());
                callers.submit(execution::cancel);
            }
            CompletableFuture.allOf(futures.toArray(CompletableFuture[]::new)).get(5, TimeUnit.SECONDS);
            assertTrue(counters.stream().allMatch(count -> count.get() == 1));
            assertEquals(0, compute.activeCount());
        }
    }

    @Test void slowBusinessCallbackDoesNotBlockOtherTimersCancellationOrClose() throws Exception {
        var entered = new CountDownLatch(1); var release = new CountDownLatch(1);
        var compute = new SimulatedCompute();
        try {
            var first = compute.submit(sample(Duration.ofHours(1).toNanos()));
            var callback = first.completion().thenAccept(value -> {
                assertTrue(Thread.currentThread().isVirtual()); entered.countDown();
                try { release.await(); } catch (InterruptedException error) { throw new RuntimeException(error); }
            });
            assertTrue(first.cancel());
            assertTrue(entered.await(2, TimeUnit.SECONDS));
            var second = compute.submit(sample(0));
            assertEquals(SimulatedCompute.State.COMPLETED, second.completion().toCompletableFuture().get(2, TimeUnit.SECONDS).state());
            compute.close();
            release.countDown(); callback.toCompletableFuture().get(2, TimeUnit.SECONDS);
        } finally { release.countDown(); compute.close(); }
    }

    @Test void closeCancelsEveryPendingComputationAndRejectsNewWork() throws Exception {
        var compute = new SimulatedCompute();
        var pending = new ArrayList<SimulatedCompute.Execution>();
        for (int i = 0; i < 100; i++) pending.add(compute.submit(sample(Duration.ofHours(1).toNanos())));
        compute.close(); compute.close();
        for (var execution : pending)
            assertEquals(SimulatedCompute.State.CANCELLED, execution.completion().toCompletableFuture().get(2, TimeUnit.SECONDS).state());
        assertEquals(0, compute.activeCount());
        assertThrows(IllegalStateException.class, () -> compute.submit(sample(0)));
    }

    @Test void closeRacingWithTimerCompletionNeverLosesNotification() throws Exception {
        for (int i = 0; i < 50; i++) {
            var compute = new SimulatedCompute();
            var execution = compute.submit(sample(0));
            compute.close();
            assertNotNull(execution.completion().toCompletableFuture().get(2, TimeUnit.SECONDS));
        }
    }
}
