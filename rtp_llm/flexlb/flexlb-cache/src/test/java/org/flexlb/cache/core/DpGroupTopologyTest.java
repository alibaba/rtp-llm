package org.flexlb.cache.core;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DpGroupTopologyTest {

    private DpGroupTopology topology;

    @BeforeEach
    void setUp() {
        topology = new DpGroupTopology();
    }

    @Test
    void update_builds_both_forward_and_reverse_maps() {
        topology.update("10.0.0.1:9080", List.of(
                new DpRankAddress(0, "10.0.0.1", 9080),
                new DpRankAddress(1, "10.0.0.1", 9081),
                new DpRankAddress(2, "10.0.0.1", 9082)));

        // Forward
        assertEquals(3, topology.ranksOf("10.0.0.1:9080").size());
        assertEquals(0, topology.ranksOf("10.0.0.1:9080").get(0).dpRank());
        assertEquals("10.0.0.1:9082", topology.ranksOf("10.0.0.1:9080").get(2).ipPort());

        // Reverse
        assertEquals("10.0.0.1:9080", topology.groupOf("10.0.0.1:9080"));
        assertEquals("10.0.0.1:9080", topology.groupOf("10.0.0.1:9081"));
        assertEquals("10.0.0.1:9080", topology.groupOf("10.0.0.1:9082"));
        assertEquals(3, topology.totalRankMappings());
    }

    @Test
    void update_replaces_old_rank_set_atomically() {
        topology.update("g1", List.of(
                new DpRankAddress(0, "10.0.0.1", 9080),
                new DpRankAddress(1, "10.0.0.1", 9081)));
        assertEquals(2, topology.totalRankMappings());

        // Re-update with a different rank composition (e.g., engine resharded).
        topology.update("g1", List.of(
                new DpRankAddress(0, "10.0.0.1", 9090),
                new DpRankAddress(1, "10.0.0.1", 9091),
                new DpRankAddress(2, "10.0.0.1", 9092)));

        // Old rank ipPorts must be gone from the reverse map.
        assertNull(topology.groupOf("10.0.0.1:9080"));
        assertNull(topology.groupOf("10.0.0.1:9081"));
        assertEquals("g1", topology.groupOf("10.0.0.1:9090"));
        assertEquals("g1", topology.groupOf("10.0.0.1:9092"));
        assertEquals(3, topology.totalRankMappings(), "no leftover entries from the previous rank set");
    }

    @Test
    void empty_or_null_update_is_noop_and_does_not_clear_existing_entry() {
        topology.update("g1", List.of(new DpRankAddress(0, "10.0.0.1", 9080)));
        assertEquals(1, topology.totalRankMappings());

        topology.update("g1", List.of());      // empty
        topology.update("g1", null);            // null
        topology.update(null, List.of(new DpRankAddress(0, "10.0.0.2", 9080)));  // null group

        assertEquals("g1", topology.groupOf("10.0.0.1:9080"),
                "an empty / null update must not erase a previously-known group");
        assertEquals(1, topology.totalRankMappings());
    }

    @Test
    void remove_clears_both_maps_for_a_group() {
        topology.update("g1", List.of(
                new DpRankAddress(0, "10.0.0.1", 9080),
                new DpRankAddress(1, "10.0.0.1", 9081)));
        topology.update("g2", List.of(
                new DpRankAddress(0, "10.0.0.2", 9080)));

        topology.remove("g1");

        assertNull(topology.groupOf("10.0.0.1:9080"));
        assertNull(topology.groupOf("10.0.0.1:9081"));
        assertTrue(topology.ranksOf("g1").isEmpty());

        // g2 untouched
        assertEquals("g2", topology.groupOf("10.0.0.2:9080"));
        assertEquals(1, topology.totalRankMappings());
        assertEquals(1, topology.groups().size());
    }

    @Test
    void unknown_lookups_return_safe_defaults() {
        assertNull(topology.groupOf("doesnt-exist:1234"));
        assertNull(topology.groupOf(null));
        assertTrue(topology.ranksOf("doesnt-exist").isEmpty());
        assertTrue(topology.ranksOf(null).isEmpty());
        assertEquals(0, topology.rankCount("doesnt-exist"));
    }

    @Test
    void multiple_groups_coexist_independently() {
        topology.update("g1", List.of(
                new DpRankAddress(0, "10.0.0.1", 9080),
                new DpRankAddress(1, "10.0.0.1", 9081)));
        topology.update("g2", List.of(
                new DpRankAddress(0, "10.0.0.2", 9080),
                new DpRankAddress(1, "10.0.0.2", 9081),
                new DpRankAddress(2, "10.0.0.2", 9082)));

        assertEquals(2, topology.groups().size());
        assertEquals(2, topology.rankCount("g1"));
        assertEquals(3, topology.rankCount("g2"));
        assertEquals(5, topology.totalRankMappings());
    }

    @Test
    void clear_drops_everything() {
        topology.update("g1", List.of(new DpRankAddress(0, "10.0.0.1", 9080)));
        topology.update("g2", List.of(new DpRankAddress(0, "10.0.0.2", 9080)));
        topology.clear();
        assertTrue(topology.groups().isEmpty());
        assertEquals(0, topology.totalRankMappings());
    }

    @Test
    void stored_rank_list_is_immutable_to_caller_mutations() {
        java.util.ArrayList<DpRankAddress> mutable = new java.util.ArrayList<>();
        mutable.add(new DpRankAddress(0, "10.0.0.1", 9080));
        topology.update("g1", mutable);

        // Caller mutation of the original list MUST NOT corrupt the topology.
        mutable.add(new DpRankAddress(99, "evil", 6666));
        mutable.clear();

        assertEquals(1, topology.ranksOf("g1").size());
        assertEquals("10.0.0.1:9080", topology.ranksOf("g1").get(0).ipPort());
    }
}
