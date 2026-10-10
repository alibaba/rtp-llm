package org.flexlb.balance.strategy;

import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import static org.junit.jupiter.api.Assertions.assertEquals;

class EndpointRoundRobinTest {
    @Test
    void changingEligibleSetDoesNotSkipAContinuouslyEligibleWorker() {
        var rotation = new EndpointRoundRobin();
        assertEquals("a", pick(rotation, "g", List.of("a", "b", "c")));
        assertEquals("b", pick(rotation, "g", List.of("b", "c")));
        assertEquals("c", pick(rotation, "g", List.of("a", "c")));
        assertEquals("a", pick(rotation, "g", List.of("c", "a", "b")));
    }

    @Test
    void interleavedGroupsKeepIndependentPosition() {
        var rotation = new EndpointRoundRobin();
        assertEquals("a", pick(rotation, "one", List.of("a", "b")));
        assertEquals("a", pick(rotation, "two", List.of("a", "b")));
        assertEquals("b", pick(rotation, "one", List.of("a", "b")));
        assertEquals("b", pick(rotation, "two", List.of("a", "b")));
    }

    @Test
    void concurrentEqualSelectionsAdvanceExactlyOnce() throws Exception {
        var rotation = new EndpointRoundRobin();
        try (var executor = Executors.newFixedThreadPool(8)) {
            List<Future<String>> selected = new ArrayList<>();
            for (int i = 0; i < 120; i++) {
                selected.add(executor.submit(() -> pick(rotation, "g", List.of("c", "a", "b"))));
            }
            var counts = new java.util.HashMap<String, Integer>();
            for (var future : selected) { counts.merge(future.get(), 1, Integer::sum); }
            assertEquals(java.util.Map.of("a", 40, "b", 40, "c", 40), counts);
        }
    }

    @Test
    void capturingOneSnapshotDoesNotBlockAnotherSelection() throws Exception {
        var rotation = new EndpointRoundRobin();
        var captureEntered = new java.util.concurrent.CountDownLatch(1);
        var releaseCapture = new java.util.concurrent.CountDownLatch(1);
        List<String> addresses = List.of("c", "a", "b");
        try (var executor = Executors.newFixedThreadPool(2)) {
            var delayed = executor.submit(() -> rotation.next(RoleType.PREFILL, "g", addresses.size(), i -> {
                if (i == 0) {
                    captureEntered.countDown();
                    try { releaseCapture.await(); }
                    catch (InterruptedException interrupted) {
                        Thread.currentThread().interrupt();
                        throw new IllegalStateException(interrupted);
                    }
                }
                return true;
            }, addresses::get));
            try {
                org.junit.jupiter.api.Assertions.assertTrue(captureEntered.await(5, java.util.concurrent.TimeUnit.SECONDS));
                var independent = executor.submit(() -> pick(rotation, "g", addresses));
                assertEquals("a", independent.get(5, java.util.concurrent.TimeUnit.SECONDS));
            } finally {
                releaseCapture.countDown();
            }
            assertEquals("b", addresses.get(delayed.get(5, java.util.concurrent.TimeUnit.SECONDS)));
        }
    }

    @Test
    void emptyAndFailedCapturesDoNotAdvanceCursor() {
        var rotation = new EndpointRoundRobin();
        List<String> addresses = List.of("b", "a", "c");
        assertEquals("a", pick(rotation, "g", addresses));
        assertEquals(-1, rotation.next(RoleType.PREFILL, "g", addresses.size(), i -> false, addresses::get));
        org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class,
                () -> rotation.next(RoleType.PREFILL, "g", addresses.size(), i -> {
                    if (i == 1) { throw new IllegalArgumentException("capture failed"); }
                    return true;
                }, addresses::get));
        assertEquals("b", pick(rotation, "g", addresses));
    }

    private static String pick(EndpointRoundRobin rotation, String group, List<String> candidates) {
        return candidates.get(rotation.next(RoleType.PREFILL, group, candidates.size(), i -> true, candidates::get));
    }
}
