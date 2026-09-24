package org.flexlb.balance.strategy;

import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertEquals;

class EndpointRoundRobinTest {
    @Test
    void batchesAndSingleSelectionsContinueAcrossMembershipChanges() {
        var rotation = new EndpointRoundRobin();
        assertEquals(List.of("a", "b"), rotation.nextBatch(RoleType.PREFILL, "g",
                Map.of("c", "c", "a", "a", "b", "b"), 2));
        assertEquals(List.of("c"), rotation.nextBatch(RoleType.PREFILL, "g",
                Map.of("a", "a", "c", "c"), 1));
        assertEquals("a", pick(rotation, "g", List.of("c", "a")));
        assertEquals(List.of("c", "d", "a", "c"), rotation.nextBatch(RoleType.PREFILL, "g",
                Map.of("d", "d", "a", "a", "c", "c"), 4));
        assertEquals(List.of("d"), rotation.nextBatch(RoleType.PREFILL, "g", Map.of("d", "d"), 1));
        assertEquals(List.of("a", "c"), rotation.nextBatch(RoleType.PREFILL, "g", Map.of("c", "c", "a", "a"), 2));
        assertEquals(List.of("a"), rotation.nextBatch(RoleType.PREFILL, "other",
                Map.of("c", "c", "a", "a"), 1));
        assertEquals(List.of("a"), rotation.nextBatch(RoleType.DECODE, "g",
                Map.of("c", "c", "a", "a"), 1));
    }

    @ParameterizedTest
    @ValueSource(ints = {2, 5, 7})
    void concurrentBatchesKeepTheirSelectionsContiguous(int count) throws Exception {
        var rotation = new EndpointRoundRobin();
        List<String> ring = List.of("a", "b", "c", "d", "e", "f");
        try (var executor = Executors.newFixedThreadPool(8)) {
            List<Future<List<String>>> batches = new ArrayList<>();
            for (int i = 0; i < 120; i++) {
                batches.add(executor.submit(() -> rotation.nextBatch(RoleType.PREFILL, "g",
                        Map.of("f", "f", "c", "c", "a", "a", "e", "e", "d", "d", "b", "b"), count)));
            }
            var totals = new java.util.HashMap<String, Integer>();
            for (var batch : batches) {
                List<String> selected = batch.get();
                int start = ring.indexOf(selected.getFirst());
                assertEquals(IntStream.range(0, count).mapToObj(i -> ring.get((start + i) % ring.size())).toList(), selected);
                selected.forEach(address -> totals.merge(address, 1, Integer::sum));
            }
            ring.forEach(address -> assertEquals(120 * count / ring.size(), totals.get(address)));
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

    private static String pick(EndpointRoundRobin rotation, String group, List<String> candidates) {
        return candidates.get(rotation.next(RoleType.PREFILL, group, candidates.size(), i -> true, candidates::get));
    }
}
