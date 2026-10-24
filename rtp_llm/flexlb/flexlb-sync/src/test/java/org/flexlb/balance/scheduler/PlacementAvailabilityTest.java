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
        List<PlacementKey> changed = new ArrayList<>();
        availability.addListener(changed::add);

        PlacementKey exact = PlacementKey.exact(
                RoleType.PREFILL, "g1", "127.0.0.1:8000");
        availability.changed(exact);

        assertEquals(List.of(exact), changed);
        assertTrue(availability.lastChangedSequence(exact) > 0L);
        assertEquals(availability.lastChangedSequence(exact),
                availability.lastChangedSequence(
                        new PlacementKey(RoleType.PREFILL, "g1", null)));
        assertEquals(availability.lastChangedSequence(exact),
                availability.lastChangedSequence(
                        PlacementKey.anyGroup(RoleType.PREFILL)));
    }

    @Test
    void anotherEndpointDoesNotAdvanceExactEdge() {
        PlacementAvailability availability = new PlacementAvailability();
        PlacementKey waiting = PlacementKey.exact(
                RoleType.PREFILL, "g1", "127.0.0.1:8000");

        availability.changed(PlacementKey.exact(
                RoleType.PREFILL, "g1", "127.0.0.2:8000"));

        assertEquals(0L, availability.lastChangedSequence(waiting));
    }

    @Test
    void endpointTopologyChangeAdvancesItsPlacementEdge() {
        PlacementAvailability availability = new PlacementAvailability();
        List<PlacementKey> changed = new ArrayList<>();
        availability.addListener(changed::add);
        PlacementKey exact = PlacementKey.exact(
                RoleType.DECODE, "g1", "127.0.0.1:9000");

        availability.changed(exact);

        assertEquals(1, changed.size());
        assertEquals(exact, changed.getFirst());
        assertTrue(availability.lastChangedSequence(exact) > 0L);
    }

    @Test
    void endpointGroupChangesShareOneExactCapacityVersion() {
        var availability = new PlacementAvailability();
        var oldGroup = PlacementKey.exact(RoleType.PREFILL, "old", "127.0.0.1:8000");
        var newGroup = PlacementKey.exact(RoleType.PREFILL, "new", oldGroup.endpoint());
        availability.changed(oldGroup);
        availability.changed(newGroup);

        assertEquals(2L, availability.lastChangedSequence(oldGroup));
        assertEquals(2L, availability.lastChangedSequence(newGroup));
        assertEquals(1L, availability.lastChangedSequence(new PlacementKey(RoleType.PREFILL, "old", null)));
        assertEquals(2L, availability.lastChangedSequence(new PlacementKey(RoleType.PREFILL, "new", null)));
        assertEquals(2L, availability.lastChangedSequence(PlacementKey.anyGroup(RoleType.PREFILL)));
        assertEquals(0L, availability.lastChangedSequence(
                PlacementKey.exact(RoleType.PREFILL, "new", "127.0.0.2:8000")));
    }

    @Test
    void delayedPublisherCannotRegressGroupOrRoleVersion() {
        PlacementAvailability availability = new PlacementAvailability();
        PlacementKey group = new PlacementKey(RoleType.PREFILL, "g1", null);
        PlacementKey role = PlacementKey.anyGroup(RoleType.PREFILL);
        PlacementKey olderExact = mock(PlacementKey.class);
        PlacementKey newerExact = PlacementKey.exact(RoleType.PREFILL, "g1", "127.0.0.2:8000");
        when(olderExact.role()).thenReturn(RoleType.PREFILL);
        when(olderExact.group()).thenReturn("g1");
        when(olderExact.capacityDomain()).thenCallRealMethod();
        // Interleave before the older publisher reaches its shared group and role keys.
        var interleaved = new java.util.concurrent.atomic.AtomicBoolean();
        when(olderExact.endpoint()).thenAnswer(ignored -> {
            if (interleaved.compareAndSet(false, true)) {
                availability.changed(newerExact);
            }
            assertEquals(2L, availability.lastChangedSequence(group));
            assertEquals(2L, availability.lastChangedSequence(role));
            return "127.0.0.1:8000";
        });
        List<PlacementKey> events = new ArrayList<>();
        availability.addListener(events::add);

        availability.changed(olderExact);

        assertAll(
                () -> assertEquals(2L, availability.lastChangedSequence(group),
                        "a delayed exact publication must preserve the newer group edge"),
                () -> assertEquals(2L, availability.lastChangedSequence(role),
                        "a delayed exact publication must preserve the newer role edge"));
        assertEquals(1L, availability.lastChangedSequence(olderExact));
        assertEquals(2L, availability.lastChangedSequence(newerExact));
        assertEquals(List.of(newerExact, olderExact), events);
    }
}
