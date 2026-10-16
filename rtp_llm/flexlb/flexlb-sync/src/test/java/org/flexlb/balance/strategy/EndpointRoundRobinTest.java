package org.flexlb.balance.strategy;

import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import static org.junit.jupiter.api.Assertions.*;

class EndpointRoundRobinTest {
    @Test
    void dynamicCapturesFollowAddressSuccessorWithoutRepeatingCallbacks() {
        var rotation = new EndpointRoundRobin();
        var last = new java.util.HashMap<String, String>();
        var random = new java.util.Random(187237);
        for (int iteration = 0; iteration < 2000; iteration++) {
            int size = random.nextInt(80);
            String group = "g" + random.nextInt(4);
            var role = random.nextBoolean() ? RoleType.PREFILL : RoleType.DECODE;
            String scope = role + group;
            String[] addresses = new String[size];
            boolean[] eligible = new boolean[size];
            int[] predicates = new int[size], captures = new int[size];
            var ring = new java.util.TreeSet<String>();
            for (int i = 0; i < size; i++) {
                addresses[i] = "worker-" + random.nextInt(100);
                eligible[i] = random.nextBoolean();
                if (eligible[i]) { ring.add(addresses[i]); }
            }
            String expected = ring.isEmpty() ? null : last.containsKey(scope) ? ring.higher(last.get(scope)) : ring.first();
            if (expected == null && !ring.isEmpty()) { expected = ring.first(); }
            int selected = rotation.next(role, group, size,
                    i -> { predicates[i]++; return eligible[i]; }, i -> { captures[i]++; return addresses[i]; });
            if (expected == null) { assertEquals(-1, selected); }
            else {
                assertEquals(expected, addresses[selected]);
                assertTrue(eligible[selected]);
                last.put(scope, expected);
            }
            for (int i = 0; i < size; i++) {
                assertEquals(1, predicates[i]);
                assertEquals(eligible[i] ? 1 : 0, captures[i]);
            }
            assertScratchReleased();
        }
    }

    @Test
    void contendedLargeRingPreservesExactFairness() throws Exception {
        var rotation = new EndpointRoundRobin();
        String[] addresses = java.util.stream.IntStream.range(0, 750).mapToObj(i -> "worker-" + i).toArray(String[]::new);
        try (var executor = Executors.newFixedThreadPool(16)) {
            var calls = new ArrayList<Future<Integer>>();
            for (int i = 0; i < 6000; i++) {
                calls.add(executor.submit(() -> rotation.next(RoleType.DECODE, "fleet", addresses.length,
                        j -> true, j -> addresses[j])));
            }
            int[] counts = new int[addresses.length];
            for (var call : calls) { counts[call.get(10, java.util.concurrent.TimeUnit.SECONDS)]++; }
            for (int count : counts) { assertEquals(8, count); }
        }
    }

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

    @Test
    void callbackFailuresReleaseCapturedReferencesWithoutAdvancing() {
        for (boolean failAddress : new boolean[]{false, true}) {
            var rotation = new EndpointRoundRobin();
            String[] addresses = {"b", "a", "c"};
            assertEquals(1, rotation.next(RoleType.PREFILL, "g", 3, i -> true, i -> addresses[i]));
            var failure = new AssertionError("callback failure");
            assertSame(failure, assertThrows(AssertionError.class, () -> rotation.next(RoleType.PREFILL, "g", 3,
                    i -> { if (i == 1 && !failAddress) { throw failure; } return true; },
                    i -> { if (i == 1 && failAddress) { throw failure; } return addresses[i]; })));
            assertScratchReleased();
            assertEquals(0, rotation.next(RoleType.PREFILL, "g", 3, i -> true, i -> addresses[i]));
            assertScratchReleased();
        }
    }

    @Test
    void callbacksMayReenterSameOrDifferentRotationWithoutReplacingOuterCapture() {
        for (boolean otherInstance : new boolean[]{false, true}) {
            for (boolean inAddress : new boolean[]{false, true}) {
                var rotation = new EndpointRoundRobin();
                var nested = otherInstance ? new EndpointRoundRobin() : rotation;
                String[] outer = {"z", "a", "m"}, inner = {"r", "q", "p", "o"};
                rotation.next(RoleType.PREFILL, "warm", 3, i -> true, i -> outer[i]);
                var cached = scratch();
                var captured = scratchArray(cached, "eligibleAddresses");
                Runnable reenter = () -> {
                    assertEquals(3, nested.next(RoleType.PREFILL, "g", 4, i -> true, i -> inner[i]));
                    assertSame(captured, scratchArray(cached, "eligibleAddresses"));
                    assertEquals("z", captured[0]);
                };
                int selected = rotation.next(RoleType.PREFILL, "g", 3,
                        i -> { if (i == 1 && !inAddress) { reenter.run(); } return true; },
                        i -> { if (i == 1 && inAddress) { reenter.run(); } return outer[i]; });
                assertEquals(otherInstance ? 1 : 0, selected);
                assertScratchReleased();
            }
        }
    }

    @Test
    void nestedFailureAndOversizedCaptureKeepCachedStorageBoundedAndReleased() {
        var rotation = new EndpointRoundRobin();
        rotation.next(RoleType.PREFILL, "warm", 750, i -> true, i -> "worker-" + i);
        var cached = scratch();
        var addresses = scratchArray(cached, "eligibleAddresses");
        var failure = new IllegalStateException("nested failure");
        assertSame(failure, assertThrows(IllegalStateException.class, () -> rotation.next(RoleType.PREFILL, "g", 3,
                i -> {
                    if (i == 1) {
                        rotation.next(RoleType.DECODE, "nested", 4097, j -> { throw failure; }, j -> "worker");
                    }
                    return true;
                }, i -> "outer-" + i)));
        assertScratchReleased();
        rotation.next(RoleType.PREFILL, "large", 4097, i -> true, i -> "worker-" + i);
        assertSame(addresses, scratchArray(cached, "eligibleAddresses"));
        assertTrue(addresses.length <= 4096);
        assertEquals(0, rotation.next(RoleType.PREFILL, "small", 1, i -> true, i -> "worker"));
        assertScratchReleased();
    }

    @Test
    void bufferGrowthPreservesDuplicateAndNullOriginalIndexContracts() {
        var rotation = new EndpointRoundRobin();
        rotation.next(RoleType.PREFILL, "warm", 60, i -> true, i -> "worker-" + i);
        String[] addresses = {"b", "a", "a", "c"};
        for (int expected : new int[]{1, 0, 3, 1}) {
            assertEquals(expected, rotation.next(RoleType.PREFILL, "g", 4, i -> true, i -> addresses[i]));
        }
        // The existing contract returns the first null slot, including an ineligible slot.
        assertEquals(0, rotation.next(RoleType.PREFILL, "null", 3, i -> i == 2, i -> null));
        assertThrows(NullPointerException.class,
                () -> rotation.next(RoleType.PREFILL, "sort-null", 2, i -> true, i -> null));
        assertThrows(NegativeArraySizeException.class,
                () -> rotation.next(RoleType.PREFILL, "negative", -1, null, null));
        assertScratchReleased();
    }

    @Test
    void concurrentCapturesHaveIndependentScratchAndShareOnlyCursor() throws Exception {
        var rotation = new EndpointRoundRobin();
        var entered = new java.util.concurrent.CountDownLatch(2);
        var release = new java.util.concurrent.CountDownLatch(1);
        var arrays = new java.util.concurrent.ConcurrentLinkedQueue<String[]>();
        try (var executor = Executors.newFixedThreadPool(2)) {
            java.util.concurrent.Callable<Integer> capture = () -> {
                int selected = rotation.next(RoleType.PREFILL, "g", 3, i -> {
                    if (i == 0) {
                        arrays.add(scratchArray(scratch(), "eligibleAddresses"));
                        entered.countDown();
                        try { assertTrue(release.await(5, java.util.concurrent.TimeUnit.SECONDS)); }
                        catch (InterruptedException interrupted) { throw new AssertionError(interrupted); }
                    }
                    return true;
                }, i -> "worker-" + i);
                assertScratchReleased();
                return selected;
            };
            var first = executor.submit(capture);
            var second = executor.submit(capture);
            try {
                assertTrue(entered.await(5, java.util.concurrent.TimeUnit.SECONDS));
                assertEquals(2, arrays.size());
                assertNotSame(arrays.toArray()[0], arrays.toArray()[1]);
            } finally { release.countDown(); }
            assertEquals(java.util.Set.of(0, 1), java.util.Set.of(
                    first.get(5, java.util.concurrent.TimeUnit.SECONDS), second.get(5, java.util.concurrent.TimeUnit.SECONDS)));
        }
    }

    private static Object scratch() {
        try {
            var field = EndpointRoundRobin.class.getDeclaredField("SCRATCH");
            field.setAccessible(true);
            return ((ThreadLocal<?>) field.get(null)).get();
        } catch (ReflectiveOperationException failure) { throw new AssertionError(failure); }
    }

    private static String[] scratchArray(Object scratch, String name) {
        try {
            var field = scratch.getClass().getDeclaredField(name);
            field.setAccessible(true);
            return (String[]) field.get(scratch);
        } catch (ReflectiveOperationException failure) { throw new AssertionError(failure); }
    }

    private static void assertScratchReleased() {
        var value = scratch();
        for (String name : new String[]{"eligibleAddresses", "orderedAddresses"}) {
            for (String address : scratchArray(value, name)) { assertNull(address); }
        }
        try {
            var field = value.getClass().getDeclaredField("inUse");
            field.setAccessible(true);
            assertFalse(field.getBoolean(value));
        } catch (ReflectiveOperationException failure) { throw new AssertionError(failure); }
    }

    private static String pick(EndpointRoundRobin rotation, String group, List<String> candidates) {
        return candidates.get(rotation.next(RoleType.PREFILL, group, candidates.size(), i -> true, candidates::get));
    }
}
