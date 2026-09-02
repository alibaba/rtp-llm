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
class PlacementWaitRegistryTest {

    private static final PlacementKey PREFILL =
            new PlacementKey(RoleType.PREFILL, "na130_online");
    private static final PlacementKey DECODE =
            new PlacementKey(RoleType.DECODE, "na130_online");

    @Test
    @Timeout(10)
    void onlyTheChangedCapacityDomainIsRetried() throws Exception {
        PlacementAvailability availability = new PlacementAvailability();
        try (PlacementWaitRegistry coordinator =
                     new PlacementWaitRegistry(availability)) {
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
                return blocked(PREFILL);
            }));
            initialPark(coordinator, work(0, false, () -> {
                decodeAttempts.incrementAndGet();
                initiallyParked.countDown();
                return blocked(DECODE);
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
        try (PlacementWaitRegistry coordinator =
                     new PlacementWaitRegistry(availability)) {
            AtomicInteger attempts = new AtomicInteger();
            AtomicBoolean seatConsumed = new AtomicBoolean();
            CountDownLatch initiallyParked = new CountDownLatch(requests);
            CountDownLatch retryEntered = new CountDownLatch(1);
            CountDownLatch releaseRetry = new CountDownLatch(1);
            List<PlacementWaitRegistry.Handle> handles =
                    new ArrayList<>(requests);

            for (int index = 0; index < requests; index++) {
                handles.add(initialPark(coordinator, work(0, false, () -> {
                    int attempt = attempts.incrementAndGet();
                    if (attempt <= requests) {
                        initiallyParked.countDown();
                        return blocked(PREFILL);
                    }
                    if (seatConsumed.compareAndSet(false, true)) {
                        retryEntered.countDown();
                        assertTrue(releaseRetry.await(5, TimeUnit.SECONDS));
                        return finished();
                    }
                    return blocked(PREFILL);
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
            handles.forEach(PlacementWaitRegistry.Handle::close);
        }
    }

    @Test
    @Timeout(20)
    void oneCapacitySignalStopsAtTheFirstBlockedHead() throws Exception {
        int requests = 20_000;
        PlacementAvailability availability = new PlacementAvailability();
        try (PlacementWaitRegistry coordinator =
                     new PlacementWaitRegistry(availability)) {
            AtomicInteger attempts = new AtomicInteger();
            List<PlacementWaitRegistry.Handle> handles =
                    new ArrayList<>(requests);

            for (int index = 0; index < requests; index++) {
                handles.add(initialPark(coordinator, work(0, false, () -> {
                    attempts.incrementAndGet();
                    return blocked(PREFILL);
                })));
            }

            assertEquals(requests, attempts.get());
            availability.capacityChanged(PREFILL);

            assertTrue(awaitAttemptsAtLeast(attempts, requests + 1, 5_000));
            Thread.sleep(100L);
            assertEquals(
                    requests + 1,
                    attempts.get(),
                    "one capacity edge bypassed its blocked ordered head");
            handles.forEach(PlacementWaitRegistry.Handle::close);
        }
    }

    @Test
    @Timeout(10)
    void laterSignalRetriesTheSameOrderedHead() throws Exception {
        int requests = 12;
        PlacementAvailability availability = new PlacementAvailability();
        try (PlacementWaitRegistry coordinator =
                     new PlacementWaitRegistry(availability)) {
            AtomicInteger totalAttempts = new AtomicInteger();
            AtomicIntegerArray attempts = new AtomicIntegerArray(requests);
            List<PlacementWaitRegistry.Handle> handles =
                    new ArrayList<>(requests);

            for (int index = 0; index < requests; index++) {
                int request = index;
                handles.add(initialPark(coordinator, work(0, false, () -> {
                    attempts.incrementAndGet(request);
                    totalAttempts.incrementAndGet();
                    return blocked(PREFILL);
                })));
            }

            availability.capacityChanged(PREFILL);
            assertTrue(awaitAttemptsAtLeast(totalAttempts, requests + 1, 5_000));
            availability.capacityChanged(PREFILL);
            assertTrue(awaitAttemptsAtLeast(totalAttempts, requests + 2, 5_000));

            assertEquals(3, attempts.get(0));
            for (int index = 1; index < requests; index++) {
                assertEquals(1, attempts.get(index));
            }
            handles.forEach(PlacementWaitRegistry.Handle::close);
        }
    }

    @Test
    @Timeout(10)
    void blockedHighPriorityHeadIsNotBypassedByLowerPriorityWork()
            throws Exception {
        PlacementAvailability availability = new PlacementAvailability();
        try (PlacementWaitRegistry coordinator =
                     new PlacementWaitRegistry(availability)) {
            AtomicInteger highAttempts = new AtomicInteger();
            AtomicInteger lowAttempts = new AtomicInteger();
            CountDownLatch initiallyParked = new CountDownLatch(2);

            initialPark(coordinator, work(100, true, () -> {
                highAttempts.incrementAndGet();
                initiallyParked.countDown();
                return blocked(DECODE);
            }));
            initialPark(coordinator, work(10, true, () -> {
                int attempt = lowAttempts.incrementAndGet();
                initiallyParked.countDown();
                if (attempt == 1) {
                    return blocked(DECODE);
                }
                return finished();
            }));

            assertTrue(initiallyParked.await(2, TimeUnit.SECONDS));
            availability.capacityChanged(DECODE);

            assertTrue(awaitAttemptsAtLeast(highAttempts, 2, 2_000));
            Thread.sleep(100L);
            assertEquals(2, highAttempts.get());
            assertEquals(1, lowAttempts.get());
        }
    }

    @Test
    @Timeout(10)
    void capacityChangeRacingWithParkCannotBeLost() throws Exception {
        PlacementAvailability availability = new PlacementAvailability();
        try (PlacementWaitRegistry coordinator =
                     new PlacementWaitRegistry(availability)) {
            AtomicInteger attempts = new AtomicInteger();
            CountDownLatch selectionObservedNoCapacity = new CountDownLatch(1);
            CountDownLatch returnBlocked = new CountDownLatch(1);
            CountDownLatch completed = new CountDownLatch(1);

            OrderedWork orderedWork = work(0, false, () -> {
                if (attempts.incrementAndGet() == 1) {
                    selectionObservedNoCapacity.countDown();
                    assertTrue(returnBlocked.await(2, TimeUnit.SECONDS));
                    return blocked(PREFILL);
                }
                completed.countDown();
                return finished();
            });
            PlacementWaitRegistry.Work work = orderedWork.work();
            ExecutorService caller = Executors.newSingleThreadExecutor();
            try {
                long observed = coordinator.availabilitySequence();
                Future<PlacementWaitRegistry.AttemptResult> initial =
                        caller.submit(work::attempt);
                assertTrue(selectionObservedNoCapacity.await(
                        2, TimeUnit.SECONDS));
                availability.capacityChanged(PREFILL);
                returnBlocked.countDown();
                coordinator.park(
                        work,
                        coordinator.newOrder(
                                orderedWork.priority(),
                                orderedWork.priorityOrdering()),
                        (PlacementWaitRegistry.AttemptResult.Blocked)
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
        try (PlacementWaitRegistry coordinator =
                     new PlacementWaitRegistry(availability)) {
            CountDownLatch completed = new CountDownLatch(1);
            OrderedWork orderedWork = work(0, false, () -> {
                completed.countDown();
                return finished();
            });
            PlacementWaitRegistry.Work work = orderedWork.work();

            long observed = coordinator.availabilitySequence();
            availability.capacityChanged(PREFILL);
            coordinator.park(
                    work,
                    coordinator.newOrder(
                            orderedWork.priority(),
                            orderedWork.priorityOrdering()),
                    new PlacementWaitRegistry.AttemptResult.Blocked(
                            PREFILL),
                    observed);

            assertTrue(completed.await(2, TimeUnit.SECONDS));
        }
    }

    @Test
    @Timeout(10)
    void capacityOpportunityContinuesPastWorkParkedBehindActiveHead()
            throws Exception {
        PlacementAvailability availability = new PlacementAvailability();
        try (PlacementWaitRegistry coordinator =
                     new PlacementWaitRegistry(availability)) {
            CountDownLatch headRetryEntered = new CountDownLatch(1);
            CountDownLatch releaseHead = new CountDownLatch(1);
            CountDownLatch followerRetried = new CountDownLatch(1);
            AtomicInteger headAttempts = new AtomicInteger();
            AtomicInteger followerAttempts = new AtomicInteger();

            initialPark(coordinator, work(0, false, () -> {
                if (headAttempts.incrementAndGet() == 1) {
                    return blocked(PREFILL);
                }
                headRetryEntered.countDown();
                assertTrue(releaseHead.await(2, TimeUnit.SECONDS));
                return finished();
            }));

            availability.capacityChanged(PREFILL);
            assertTrue(headRetryEntered.await(2, TimeUnit.SECONDS));

            OrderedWork follower = work(0, false, () -> {
                if (followerAttempts.incrementAndGet() == 1) {
                    // The real scheduler reaches this state when selection sees
                    // spare capacity but ordering keeps it behind the active head.
                    return blocked(PREFILL);
                }
                followerRetried.countDown();
                return finished();
            });
            initialPark(coordinator, follower);
            releaseHead.countDown();

            assertTrue(
                    followerRetried.await(2, TimeUnit.SECONDS),
                    "the live capacity opportunity stopped after its active head");
            assertEquals(2, headAttempts.get());
            assertEquals(2, followerAttempts.get());
        }
    }

    @Test
    @Timeout(20)
    void liveCapacityOpportunityDrainsNewFollowersUntilFirstRealMiss()
            throws Exception {
        int requests = 10_000;
        int availableSeats = 128;
        PlacementAvailability availability = new PlacementAvailability();
        try (PlacementWaitRegistry coordinator =
                     new PlacementWaitRegistry(availability)) {
            CountDownLatch headRetryEntered = new CountDownLatch(1);
            CountDownLatch releaseHead = new CountDownLatch(1);
            AtomicInteger headAttempts = new AtomicInteger();
            initialPark(coordinator, work(0, false, () -> {
                if (headAttempts.incrementAndGet() == 1) {
                    return blocked(PREFILL);
                }
                headRetryEntered.countDown();
                assertTrue(releaseHead.await(5, TimeUnit.SECONDS));
                return finished();
            }));

            availability.capacityChanged(PREFILL);
            assertTrue(headRetryEntered.await(2, TimeUnit.SECONDS));

            AtomicInteger seats = new AtomicInteger(availableSeats);
            AtomicInteger retries = new AtomicInteger();
            AtomicIntegerArray attempts = new AtomicIntegerArray(requests);
            List<PlacementWaitRegistry.Handle> handles =
                    new ArrayList<>(requests);
            for (int index = 0; index < requests; index++) {
                int request = index;
                handles.add(initialPark(coordinator, work(0, false, () -> {
                    if (attempts.incrementAndGet(request) == 1) {
                        return blocked(PREFILL);
                    }
                    retries.incrementAndGet();
                    return seats.getAndDecrement() > 0
                            ? finished() : blocked(PREFILL);
                })));
            }

            releaseHead.countDown();
            assertTrue(awaitAttemptsAtLeast(
                    retries, availableSeats + 1, 5_000));
            Thread.sleep(100L);

            assertEquals(2, headAttempts.get());
            assertEquals(
                    availableSeats + 1,
                    retries.get(),
                    "the opportunity must stop at its first real miss");
            assertEquals(requests - availableSeats, coordinator.size());
            for (int index = 0; index < requests; index++) {
                assertTrue(
                        attempts.get(index) <= 2,
                        "request " + index + " was retried more than once");
            }
            handles.forEach(PlacementWaitRegistry.Handle::close);
        }
    }

    @Test
    @Timeout(10)
    void cancelledEntriesAreRemovedWithoutWaitingForAnotherSignal()
            throws Exception {
        PlacementAvailability availability = new PlacementAvailability();
        try (PlacementWaitRegistry coordinator =
                     new PlacementWaitRegistry(availability)) {
            int requests = 2_000;
            CountDownLatch parked = new CountDownLatch(requests);
            List<PlacementWaitRegistry.Handle> handles =
                    new ArrayList<>(requests);
            for (int index = 0; index < requests; index++) {
                handles.add(initialPark(coordinator, work(0, false, () -> {
                    parked.countDown();
                    return blocked(PREFILL);
                })));
            }
            assertTrue(parked.await(3, TimeUnit.SECONDS));

            handles.forEach(PlacementWaitRegistry.Handle::close);

            assertTrue(awaitSize(coordinator, 0, 2_000));
            assertEquals(0, coordinator.size());
        }
    }

    @Test
    @Timeout(10)
    void aRequestIsAttemptedAtMostOnceWithoutCapacityChange()
            throws Exception {
        PlacementAvailability availability = new PlacementAvailability();
        try (PlacementWaitRegistry coordinator =
                     new PlacementWaitRegistry(availability)) {
            int requests = 512;
            AtomicIntegerArray attempts = new AtomicIntegerArray(requests);
            CountDownLatch parked = new CountDownLatch(requests);
            for (int index = 0; index < requests; index++) {
                int request = index;
                initialPark(coordinator, work(0, false, () -> {
                    attempts.incrementAndGet(request);
                    parked.countDown();
                    return blocked(PREFILL);
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

    private static PlacementWaitRegistry.AttemptResult blocked(
            PlacementKey key) {
        return new PlacementWaitRegistry.AttemptResult.Blocked(key);
    }

    private static PlacementWaitRegistry.AttemptResult finished() {
        return PlacementWaitRegistry.AttemptResult.Finished.INSTANCE;
    }

    private static PlacementWaitRegistry.Handle initialPark(
            PlacementWaitRegistry coordinator,
            OrderedWork orderedWork) {
        PlacementWaitRegistry.Work work = orderedWork.work();
        long observed = coordinator.availabilitySequence();
        PlacementWaitRegistry.AttemptResult result = work.attempt();
        if (!(result instanceof
                PlacementWaitRegistry.AttemptResult.Blocked blocked)) {
            throw new AssertionError("initial attempt did not block");
        }
        return coordinator.park(
                work,
                coordinator.newOrder(
                        orderedWork.priority(),
                        orderedWork.priorityOrdering()),
                blocked,
                observed);
    }

    private static OrderedWork work(
            int priority,
            boolean priorityOrdering,
            Attempt attempt) {
        PlacementWaitRegistry.Work work = new PlacementWaitRegistry.Work() {
            @Override
            public boolean done() {
                return false;
            }

            @Override
            public PlacementWaitRegistry.AttemptResult attempt() {
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
        return new OrderedWork(priority, priorityOrdering, work);
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
            PlacementWaitRegistry coordinator,
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
        PlacementWaitRegistry.AttemptResult run()
                throws InterruptedException;
    }

    private record OrderedWork(
            int priority,
            boolean priorityOrdering,
            PlacementWaitRegistry.Work work) {
    }
}
