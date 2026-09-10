package org.flexlb.balance.scheduler;

import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

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

    @Test
    void delayedPublisherCannotRegressGroupOrRoleVersion() {
        PlacementAvailability availability = new PlacementAvailability();
        PlacementKey group = new PlacementKey(RoleType.PREFILL, "g1");
        PlacementKey role = PlacementKey.anyGroup(RoleType.PREFILL);
        PlacementKey olderExact = mock(PlacementKey.class);
        PlacementKey newerExact = PlacementKey.exact(RoleType.PREFILL, "g1", "127.0.0.2:8000");
        when(olderExact.role()).thenReturn(RoleType.PREFILL);
        when(olderExact.group()).thenReturn("g1");
        // Interleave the newer publication after the older exact write but
        // before the older publisher reaches its shared group and role keys.
        when(olderExact.endpoint()).thenAnswer(ignored -> {
            availability.topologyChanged(newerExact);
            assertEquals(2L, availability.lastChangedSequence(group));
            assertEquals(2L, availability.lastChangedSequence(role));
            return "127.0.0.1:8000";
        });
        List<PlacementAvailability.Event> events = new ArrayList<>();
        availability.addListener(events::add);

        availability.capacityChanged(olderExact);

        assertAll(
                () -> assertEquals(2L, availability.lastChangedSequence(group),
                        "a delayed exact publication must preserve the newer group edge"),
                () -> assertEquals(2L, availability.lastChangedSequence(role),
                        "a delayed exact publication must preserve the newer role edge"));
        assertEquals(1L, availability.lastChangedSequence(olderExact));
        assertEquals(2L, availability.lastChangedSequence(newerExact));
        assertEquals(List.of(
                new PlacementAvailability.Event(newerExact, 2L, PlacementAvailability.ChangeKind.TOPOLOGY),
                new PlacementAvailability.Event(olderExact, 1L, PlacementAvailability.ChangeKind.CAPACITY)), events);
    }
}
