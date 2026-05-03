package org.flexlb.balance.dp;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

class RoundRobinGroupSelectorTest {

    private final RoundRobinGroupSelector selector = new RoundRobinGroupSelector();

    @Test
    void cycles_through_candidates_per_batch() {
        List<WorkerStatus> candidates = workers("10.0.0.3", "10.0.0.1", "10.0.0.2");
        DispatchContext ctx = ctx();

        // Six picks → each candidate selected exactly twice (sorted by ipPort: .1, .2, .3)
        Map<String, Integer> counts = new HashMap<>();
        List<String> sequence = new ArrayList<>();
        for (int i = 0; i < 6; i++) {
            WorkerStatus w = selector.select(candidates, ctx);
            counts.merge(w.getIpPort(), 1, Integer::sum);
            sequence.add(w.getIp());
        }
        assertEquals(2, counts.get("10.0.0.1:8080"));
        assertEquals(2, counts.get("10.0.0.2:8080"));
        assertEquals(2, counts.get("10.0.0.3:8080"));
        // Stable sort means cursor 0..5 lands on .1, .2, .3, .1, .2, .3 in order.
        assertEquals(List.of("10.0.0.1", "10.0.0.2", "10.0.0.3",
                             "10.0.0.1", "10.0.0.2", "10.0.0.3"), sequence);
    }

    @Test
    void empty_candidates_returns_null() {
        assertNull(selector.select(List.of(), ctx()));
        assertNull(selector.select(null, ctx()));
    }

    @Test
    void single_candidate_always_selected() {
        List<WorkerStatus> candidates = workers("10.0.0.7");
        for (int i = 0; i < 3; i++) {
            assertEquals("10.0.0.7", selector.select(candidates, ctx()).getIp());
        }
    }

    private static DispatchContext ctx() {
        return new DispatchContext("m", 4, new FlexlbConfig(), List.of());
    }

    @Test
    void name_is_RR() {
        assertEquals("RR", selector.name());
    }

    private static List<WorkerStatus> workers(String... ips) {
        List<WorkerStatus> out = new ArrayList<>();
        for (String ip : ips) {
            WorkerStatus w = new WorkerStatus();
            w.setIp(ip);
            w.setPort(8080);
            w.setDpSize(4);
            w.setAlive(true);
            out.add(w);
        }
        return out;
    }
}
