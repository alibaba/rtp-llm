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

/**
 * Checks monotonic role and group capacity versions, including delayed events and retired history.
 */
class PlacementAvailabilityTest {

    @Test
    void workerReleaseAdvancesGroupAndRoleEdges() {
        PlacementAvailability availability = new PlacementAvailability();
        List<PlacementKey> changed = new ArrayList<>();
        availability.addListener(changed::add);

        PlacementKey exact = PlacementKey.exact(RoleType.PREFILL, "g1", "127.0.0.1:8000");
        PlacementAvailabilityTestSupport.changed(availability, exact);

        assertEquals(List.of(exact), changed);
        assertTrue(availability.lastChangedSequence(exact) > 0L);
        assertEquals(
                availability.lastChangedSequence(exact),
                availability.lastChangedSequence(new PlacementKey(RoleType.PREFILL, "g1", null)));
        assertEquals(
                availability.lastChangedSequence(exact),
                availability.lastChangedSequence(PlacementKey.anyGroup(RoleType.PREFILL)));
    }

    @Test
    void anotherEndpointInTheSameGroupAdvancesTheWaitingEdge() {
        PlacementAvailability availability = new PlacementAvailability();
        PlacementKey waiting = PlacementKey.exact(RoleType.PREFILL, "g1", "127.0.0.1:8000");

        PlacementAvailabilityTestSupport.changed(
                availability, PlacementKey.exact(RoleType.PREFILL, "g1", "127.0.0.2:8000"));

        assertTrue(availability.lastChangedSequence(waiting) > 0L);
    }

    @Test
    void endpointTopologyChangeAdvancesItsPlacementEdge() {
        PlacementAvailability availability = new PlacementAvailability();
        List<PlacementKey> changed = new ArrayList<>();
        availability.addListener(changed::add);
        PlacementKey exact = PlacementKey.exact(RoleType.DECODE, "g1", "127.0.0.1:9000");

        PlacementAvailabilityTestSupport.changed(availability, exact);

        assertEquals(1, changed.size());
        assertEquals(exact, changed.getFirst());
        assertTrue(availability.lastChangedSequence(exact) > 0L);
    }

    @Test
    void endpointGroupChangesHaveIndependentGroupVersions() {
        var availability = new PlacementAvailability();
        var oldGroup = PlacementKey.exact(RoleType.PREFILL, "old", "127.0.0.1:8000");
        var newGroup = PlacementKey.exact(RoleType.PREFILL, "new", oldGroup.endpoint());
        PlacementAvailabilityTestSupport.changed(availability, oldGroup);
        PlacementAvailabilityTestSupport.changed(availability, newGroup);

        assertEquals(1L, availability.lastChangedSequence(oldGroup));
        assertEquals(2L, availability.lastChangedSequence(newGroup));
        assertEquals(
                1L,
                availability.lastChangedSequence(new PlacementKey(RoleType.PREFILL, "old", null)));
        assertEquals(
                2L,
                availability.lastChangedSequence(new PlacementKey(RoleType.PREFILL, "new", null)));
        assertEquals(2L, availability.lastChangedSequence(PlacementKey.anyGroup(RoleType.PREFILL)));
        assertEquals(
                2L,
                availability.lastChangedSequence(
                        PlacementKey.exact(RoleType.PREFILL, "new", "127.0.0.2:8000")));
    }

    @Test
    void retirementFloorPreventsSleepingOnDiscardedHistory() {
        var availability = new PlacementAvailability();
        var worker =
                org.flexlb.dao.master.WorkerStatus.createDiscovered(
                        RoleType.PREFILL, "g1", "127.0.0.1", 8000, 8001, null);
        var key = new PlacementKey(RoleType.PREFILL, "g1", null);
        availability.changed(worker, worker.getLogicalIpPort());
        long observed = availability.sequence();
        worker.lock.lock();
        try {
            worker.beginRetirementAfterEndpointGateClosed();
        } finally {
            worker.lock.unlock();
        }
        availability.clearRetired(worker);
        assertTrue(availability.lastChangedSequence(key) > observed);
        assertTrue(
                availability.lastChangedSequence(PlacementKey.anyGroup(RoleType.PREFILL))
                        > observed);
    }

    @Test
    void delayedPublisherCannotRegressGroupOrRoleVersion() {
        PlacementAvailability availability = new PlacementAvailability();
        PlacementKey group = new PlacementKey(RoleType.PREFILL, "g1", null);
        PlacementKey role = PlacementKey.anyGroup(RoleType.PREFILL);
        PlacementKey olderExact = PlacementKey.exact(RoleType.PREFILL, "g1", "127.0.0.1:8000");
        PlacementKey newerExact = PlacementKey.exact(RoleType.PREFILL, "g1", "127.0.0.2:8000");
        var olderWorker = mock(org.flexlb.dao.master.WorkerStatus.class);
        var topology =
                org.flexlb.dao.master.WorkerStatus.createDiscovered(
                                RoleType.PREFILL, "g1", "127.0.0.1", 8000, 8001, null)
                        .topologySnapshot();
        when(olderWorker.getRole()).thenReturn(RoleType.PREFILL);
        when(olderWorker.topologySnapshot()).thenReturn(topology);
        var interleaved = new java.util.concurrent.atomic.AtomicBoolean();
        when(olderWorker.isActiveGeneration())
                .thenAnswer(
                        ignored -> {
                            if (interleaved.compareAndSet(false, true)) {
                                PlacementAvailabilityTestSupport.changed(availability, newerExact);
                            }
                            assertEquals(2L, availability.lastChangedSequence(group));
                            assertEquals(2L, availability.lastChangedSequence(role));
                            return true;
                        });
        List<PlacementKey> events = new ArrayList<>();
        availability.addListener(events::add);

        availability.changed(olderWorker, olderExact.endpoint());

        assertAll(
                () ->
                        assertEquals(
                                2L,
                                availability.lastChangedSequence(group),
                                "a delayed exact publication must preserve the newer group edge"),
                () ->
                        assertEquals(
                                2L,
                                availability.lastChangedSequence(role),
                                "a delayed exact publication must preserve the newer role edge"));
        assertEquals(2L, availability.lastChangedSequence(olderExact));
        assertEquals(2L, availability.lastChangedSequence(newerExact));
        assertEquals(List.of(newerExact, olderExact), events);
    }
}
