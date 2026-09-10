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

    private static String pick(EndpointRoundRobin rotation, String group, List<String> candidates) {
        return candidates.get(rotation.next(RoleType.PREFILL, group, candidates.size(), i -> true, candidates::get));
    }
}
