package org.flexlb.balance.scheduler;

import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PlacementAvailabilityTest {

    @Test
    void exactReleaseAdvancesExactGroupAndRoleEdges() {
        PlacementAvailability availability = new PlacementAvailability();
        List<PlacementAvailability.Event> changed = new ArrayList<>();
        availability.addListener(changed::add);

        PlacementKey exact = PlacementKey.exact(
                RoleType.PREFILL, "g1", "127.0.0.1:8000");
        availability.capacityChanged(exact);

        assertEquals(List.of(new PlacementAvailability.Event(
                exact,
                availability.lastChangedSequence(exact),
                PlacementAvailability.ChangeKind.CAPACITY)), changed);
        assertTrue(availability.lastChangedSequence(exact) > 0L);
        assertEquals(availability.lastChangedSequence(exact),
                availability.lastChangedSequence(
                        new PlacementKey(RoleType.PREFILL, "g1")));
        assertEquals(availability.lastChangedSequence(exact),
                availability.lastChangedSequence(
                        PlacementKey.anyGroup(RoleType.PREFILL)));
    }

    @Test
    void anotherEndpointDoesNotAdvanceExactEdge() {
        PlacementAvailability availability = new PlacementAvailability();
        PlacementKey waiting = PlacementKey.exact(
                RoleType.PREFILL, "g1", "127.0.0.1:8000");

        availability.capacityChanged(PlacementKey.exact(
                RoleType.PREFILL, "g1", "127.0.0.2:8000"));

        assertEquals(0L, availability.lastChangedSequence(waiting));
    }

    @Test
    void topologyChangeIsDistinctFromCapacityRelease() {
        PlacementAvailability availability = new PlacementAvailability();
        List<PlacementAvailability.Event> changed = new ArrayList<>();
        availability.addListener(changed::add);
        PlacementKey exact = PlacementKey.exact(
                RoleType.DECODE, "g1", "127.0.0.1:9000");

        availability.topologyChanged(exact);

        assertEquals(1, changed.size());
        assertEquals(exact, changed.getFirst().key());
        assertEquals(PlacementAvailability.ChangeKind.TOPOLOGY,
                changed.getFirst().kind());
    }
}
