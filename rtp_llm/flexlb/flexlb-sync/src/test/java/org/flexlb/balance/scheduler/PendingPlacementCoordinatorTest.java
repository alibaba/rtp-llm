package org.flexlb.balance.scheduler;

import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicIntegerArray;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Regression contracts for the production placement retry storm. */
class PendingPlacementCoordinatorTest {

    private static final PlacementKey PREFILL =
            new PlacementKey(RoleType.PREFILL, "na130_online");
    private static final PlacementKey DECODE =
            new PlacementKey(RoleType.DECODE, "na130_online");

    @Test
    @Timeout(10)
    void onlyTheChangedCapacityDomainIsRetried() throws Exception {
        PlacementAvailability availability = new PlacementAvailability();
        try (PendingPlacementCoordinator coordinator =
                     new PendingPlacementCoordinator(availability)) {
            AtomicInteger prefillAttempts = new AtomicInteger();
            AtomicInteger decodeAttempts = new AtomicInteger();
            CountDownLatch initiallyParked = new CountDownLatch(2);
            CountDownLatch prefillRetried = new CountDownLatch(1);

            initialPark(coordinator, work(0, false, () -> {
                if (prefillAttempts.incrementAndGet() == 1) {
                    initiallyParked.countDown();
                } else {
                    prefillRetried.countDown();
                }
                return blocked(PREFILL, PlacementBlockScope.POOL_WIDE);
            }));
            initialPark(coordinator, work(0, false, () -> {
                decodeAttempts.incrementAndGet();
                initiallyParked.countDown();
                return blocked(DECODE, PlacementBlockScope.POOL_WIDE);
            }));

            assertTrue(initiallyParked.await(2, TimeUnit.SECONDS));
            availability.capacityChanged(PREFILL);

            assertTrue(prefillRetried.await(2, TimeUnit.SECONDS));
            assertEquals(2, prefillAttempts.get());
            assertEquals(1, decodeAttempts.get());
        }
    }

    @Test
    @Timeout(20)
    void duplicateSignalsDoNotReplayTheWholeBacklog() throws Exception {
        int requests = 20_000;
        PlacementAvailability availability = new PlacementAvailability();
        try (PendingPlacementCoordinator coordinator =
                     new PendingPlacementCoordinator(availability)) {
            AtomicInteger attempts = new AtomicInteger();
            AtomicBoolean seatConsumed = new AtomicBoolean();
            CountDownLatch initiallyParked = new CountDownLatch(requests);
            CountDownLatch retryEntered = new CountDownLatch(1);
            CountDownLatch releaseRetry = new CountDownLatch(1);
            List<PendingPlacementCoordinator.Handle> handles =
                    new ArrayList<>(requests);

            for (int index = 0; index < requests; index++) {
                handles.add(initialPark(coordinator, work(0, false, () -> {
                    int attempt = attempts.incrementAndGet();
                    if (attempt <= requests) {
                        initiallyParked.countDown();
                        return blocked(PREFILL, PlacementBlockScope.POOL_WIDE);
                    }
                    if (seatConsumed.compareAndSet(false, true)) {
                        retryEntered.countDown();
                        assertTrue(releaseRetry.await(5, TimeUnit.SECONDS));
                        return finished();
                    }
                    return blocked(PREFILL, PlacementBlockScope.POOL_WIDE);
                })));
            }

            assertTrue(initiallyParked.await(5, TimeUnit.SECONDS));
            availability.capacityChanged(PREFILL);
            assertTrue(retryEntered.await(2, TimeUnit.SECONDS));
            for (int signal = 0; signal < 1_000; signal++) {
                availability.capacityChanged(PREFILL);
            }
            releaseRetry.countDown();

            assertTrue(awaitAttemptsAtLeast(attempts, requests + 2, 5_000));
            Thread.sleep(100L);
            assertTrue(
                    attempts.get() <= requests + 4,
                    "duplicate capacity signals replayed pending requests: "
                            + attempts.get());
            handles.forEach(PendingPlacementCoordinator.Handle::close);
        }
    }

    @Test
    @Timeout(20)
    void oneLimitedCapacitySignalCannotScanTheWholeBacklog() throws Exception {
        int requests = 20_000;
        int boundedAttempts = PendingPlacementCoordinator
                .MAX_LIMITED_ATTEMPTS_PER_ACTIVATION;
        PlacementAvailability availability = new PlacementAvailability();
        try (PendingPlacementCoordinator coordinator =
                     new PendingPlacementCoordinator(availability)) {
            AtomicInteger attempts = new AtomicInteger();
            List<PendingPlacementCoordinator.Handle> handles =
                    new ArrayList<>(requests);

            for (int index = 0; index < requests; index++) {
                handles.add(initialPark(coordinator, work(0, false, () -> {
                    attempts.incrementAndGet();
                    return blocked(PREFILL, PlacementBlockScope.LIMITED);
                })));
            }

            assertEquals(requests, attempts.get());
            availability.capacityChanged(PREFILL);

            assertTrue(awaitAttemptsAtLeast(
                    attempts,
                    requests + boundedAttempts,
                    5_000));
            Thread.sleep(100L);
            assertEquals(
                    requests + boundedAttempts,
                    attempts.get(),
                    "one capacity edge scanned beyond its bounded retry budget");
            handles.forEach(PendingPlacementCoordinator.Handle::close);
        }
    }

    @Test
    @Timeout(10)
    void laterLimitedSignalContinuesFromFirstUntriedRequest() throws Exception {
        int boundedAttempts = PendingPlacementCoordinator
                .MAX_LIMITED_ATTEMPTS_PER_ACTIVATION;
        int requests = boundedAttempts * 2 + 10;
        PlacementAvailability availability = new PlacementAvailability();
        try (PendingPlacementCoordinator coordinator =
                     new PendingPlacementCoordinator(availability)) {
            AtomicInteger totalAttempts = new AtomicInteger();
            AtomicIntegerArray attempts = new AtomicIntegerArray(requests);
            List<PendingPlacementCoordinator.Handle> handles =
                    new ArrayList<>(requests);

            for (int index = 0; index < requests; index++) {
                int request = index;
                handles.add(initialPark(coordinator, work(0, false, () -> {
                    attempts.incrementAndGet(request);
                    totalAttempts.incrementAndGet();
                    return blocked(PREFILL, PlacementBlockScope.LIMITED);
                })));
            }

            availability.capacityChanged(PREFILL);
            assertTrue(awaitAttemptsAtLeast(
                    totalAttempts, requests + boundedAttempts, 5_000));
            availability.capacityChanged(PREFILL);
            assertTrue(awaitAttemptsAtLeast(
                    totalAttempts, requests + boundedAttempts * 2, 5_000));

            for (int index = 0; index < boundedAttempts * 2; index++) {
                assertEquals(2, attempts.get(index));
            }
            for (int index = boundedAttempts * 2; index < requests; index++) {
                assertEquals(1, attempts.get(index));
            }
            handles.forEach(PendingPlacementCoordinator.Handle::close);
        }
    }

    @Test
    @Timeout(10)
    void oneRequestSpecificMissDoesNotBlockRunnableLowerPriorityWork()
            throws Exception {
        PlacementAvailability availability = new PlacementAvailability();
        try (PendingPlacementCoordinator coordinator =
                     new PendingPlacementCoordinator(availability)) {
            AtomicInteger highAttempts = new AtomicInteger();
            AtomicInteger lowAttempts = new AtomicInteger();
            CountDownLatch initiallyParked = new CountDownLatch(2);
            CountDownLatch lowCompleted = new CountDownLatch(1);

            initialPark(coordinator, work(100, true, () -> {
                highAttempts.incrementAndGet();
                initiallyParked.countDown();
                return blocked(DECODE, PlacementBlockScope.LIMITED);
            }));
            initialPark(coordinator, work(10, true, () -> {
                int attempt = lowAttempts.incrementAndGet();
                initiallyParked.countDown();
                if (attempt == 1) {
                    return blocked(DECODE, PlacementBlockScope.LIMITED);
                }
                lowCompleted.countDown();
                return finished();
            }));

            assertTrue(initiallyParked.await(2, TimeUnit.SECONDS));
            availability.capacityChanged(DECODE);

            assertTrue(lowCompleted.await(2, TimeUnit.SECONDS));
            assertEquals(2, highAttempts.get());
            assertEquals(2, lowAttempts.get());
        }
    }

    @Test
    @Timeout(10)
    void capacityChangeRacingWithParkCannotBeLost() throws Exception {
        PlacementAvailability availability = new PlacementAvailability();
        try (PendingPlacementCoordinator coordinator =
                     new PendingPlacementCoordinator(availability)) {
            AtomicInteger attempts = new AtomicInteger();
            CountDownLatch selectionObservedNoCapacity = new CountDownLatch(1);
            CountDownLatch returnBlocked = new CountDownLatch(1);
            CountDownLatch completed = new CountDownLatch(1);

            PendingPlacementCoordinator.Work work = work(0, false, () -> {
                if (attempts.incrementAndGet() == 1) {
                    selectionObservedNoCapacity.countDown();
                    assertTrue(returnBlocked.await(2, TimeUnit.SECONDS));
                    return blocked(PREFILL, PlacementBlockScope.POOL_WIDE);
                }
                completed.countDown();
                return finished();
            });
            ExecutorService caller = Executors.newSingleThreadExecutor();
            try {
                long observed = coordinator.availabilitySequence();
                Future<PendingPlacementCoordinator.AttemptResult> initial =
                        caller.submit(work::attempt);
                assertTrue(selectionObservedNoCapacity.await(
                        2, TimeUnit.SECONDS));
                availability.capacityChanged(PREFILL);
                returnBlocked.countDown();
                coordinator.park(
                        work,
                        (PendingPlacementCoordinator.AttemptResult.Blocked)
                                initial.get(2, TimeUnit.SECONDS),
                        observed);
            } finally {
                caller.shutdownNow();
            }

            assertTrue(completed.await(2, TimeUnit.SECONDS));
            assertEquals(2, attempts.get());
        }
    }

    @Test
    @Timeout(10)
    void capacityChangeBeforeExplicitParkCannotBeLost() throws Exception {
        PlacementAvailability availability = new PlacementAvailability();
        try (PendingPlacementCoordinator coordinator =
                     new PendingPlacementCoordinator(availability)) {
            CountDownLatch completed = new CountDownLatch(1);
            PendingPlacementCoordinator.Work work = work(0, false, () -> {
                completed.countDown();
                return finished();
            });

            long observed = coordinator.availabilitySequence();
            availability.capacityChanged(PREFILL);
            coordinator.park(
                    work,
                    new PendingPlacementCoordinator.AttemptResult.Blocked(
                            PREFILL, PlacementBlockScope.POOL_WIDE),
                    observed);

            assertTrue(completed.await(2, TimeUnit.SECONDS));
        }
    }

    @Test
    @Timeout(10)
    void cancelledEntriesAreRemovedWithoutWaitingForAnotherSignal()
            throws Exception {
        PlacementAvailability availability = new PlacementAvailability();
        try (PendingPlacementCoordinator coordinator =
                     new PendingPlacementCoordinator(availability)) {
            int requests = 2_000;
            CountDownLatch parked = new CountDownLatch(requests);
            List<PendingPlacementCoordinator.Handle> handles =
                    new ArrayList<>(requests);
            for (int index = 0; index < requests; index++) {
                handles.add(initialPark(coordinator, work(0, false, () -> {
                    parked.countDown();
                    return blocked(PREFILL, PlacementBlockScope.POOL_WIDE);
                })));
            }
            assertTrue(parked.await(3, TimeUnit.SECONDS));

            handles.forEach(PendingPlacementCoordinator.Handle::close);

            assertTrue(awaitSize(coordinator, 0, 2_000));
            assertEquals(0, coordinator.size());
        }
    }

    @Test
    @Timeout(10)
    void aRequestIsAttemptedAtMostOnceWithoutCapacityChange()
            throws Exception {
        PlacementAvailability availability = new PlacementAvailability();
        try (PendingPlacementCoordinator coordinator =
                     new PendingPlacementCoordinator(availability)) {
            int requests = 512;
            AtomicIntegerArray attempts = new AtomicIntegerArray(requests);
            CountDownLatch parked = new CountDownLatch(requests);
            for (int index = 0; index < requests; index++) {
                int request = index;
                initialPark(coordinator, work(0, false, () -> {
                    attempts.incrementAndGet(request);
                    parked.countDown();
                    return blocked(PREFILL, PlacementBlockScope.POOL_WIDE);
                }));
            }

            assertTrue(parked.await(3, TimeUnit.SECONDS));
            Thread.sleep(100L);
            for (int index = 0; index < requests; index++) {
                assertEquals(1, attempts.get(index));
            }
            assertEquals(requests, coordinator.size());
        }
    }

    private static PendingPlacementCoordinator.AttemptResult blocked(
            PlacementKey key, PlacementBlockScope scope) {
        return new PendingPlacementCoordinator.AttemptResult.Blocked(
                key, scope);
    }

    private static PendingPlacementCoordinator.AttemptResult finished() {
        return PendingPlacementCoordinator.AttemptResult.Finished.INSTANCE;
    }

    private static PendingPlacementCoordinator.Handle initialPark(
            PendingPlacementCoordinator coordinator,
            PendingPlacementCoordinator.Work work) {
        long observed = coordinator.availabilitySequence();
        PendingPlacementCoordinator.AttemptResult result = work.attempt();
        if (!(result instanceof
                PendingPlacementCoordinator.AttemptResult.Blocked blocked)) {
            throw new AssertionError("initial attempt did not block");
        }
        return coordinator.park(work, blocked, observed);
    }

    private static PendingPlacementCoordinator.Work work(
            int priority,
            boolean priorityOrdering,
            Attempt attempt) {
        return new PendingPlacementCoordinator.Work() {
            @Override
            public int priority() {
                return priority;
            }

            @Override
            public boolean priorityOrdering() {
                return priorityOrdering;
            }

            @Override
            public boolean done() {
                return false;
            }

            @Override
            public PendingPlacementCoordinator.AttemptResult attempt() {
                try {
                    return attempt.run();
                } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                    throw new AssertionError(interrupted);
                }
            }

            @Override
            public void fail(Throwable failure) {
                // Some tests intentionally leave work parked when the coordinator is
                // closed. Production work completes its request exceptionally here;
                // the fixture has no request future to complete.
            }
        };
    }

    private static boolean awaitAttemptsAtLeast(
            AtomicInteger attempts, int expected, long timeoutMs)
            throws InterruptedException {
        long deadline = System.nanoTime()
                + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        while (attempts.get() < expected && System.nanoTime() < deadline) {
            Thread.sleep(1L);
        }
        return attempts.get() >= expected;
    }

    private static boolean awaitSize(
            PendingPlacementCoordinator coordinator,
            int expected,
            long timeoutMs) throws InterruptedException {
        long deadline = System.nanoTime()
                + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        while (coordinator.size() != expected
                && System.nanoTime() < deadline) {
            Thread.sleep(1L);
        }
        return coordinator.size() == expected;
    }

    @FunctionalInterface
    private interface Attempt {
        PendingPlacementCoordinator.AttemptResult run()
                throws InterruptedException;
    }
}
