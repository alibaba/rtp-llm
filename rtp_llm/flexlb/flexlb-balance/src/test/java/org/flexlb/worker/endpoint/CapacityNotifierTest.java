package org.flexlb.worker.endpoint;

import org.flexlb.config.service.RoleType;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Checks monotonic role and group capacity versions, including delayed events and closed history.
 */
class CapacityNotifierTest {

    @Test
    void workerReleaseAdvancesGroupAndRoleEdges() {
        CapacityNotifier capacityNotifier = new CapacityNotifier();
        List<CapacityKey> changed = new ArrayList<>();
        capacityNotifier.addListener(changed::add);

        CapacityKey exact = CapacityKey.exact(RoleType.PREFILL, "g1", "127.0.0.1:8000");
        CapacityNotifierTestSupport.notifyChanged(capacityNotifier, exact);

        assertEquals(List.of(exact), changed);
        assertTrue(capacityNotifier.lastChangedSequence(exact) > 0L);
        assertEquals(
                capacityNotifier.lastChangedSequence(exact),
                capacityNotifier.lastChangedSequence(
                        new CapacityKey(RoleType.PREFILL, "g1", null)));
        assertEquals(
                capacityNotifier.lastChangedSequence(exact),
                capacityNotifier.lastChangedSequence(CapacityKey.anyGroup(RoleType.PREFILL)));
    }

    @Test
    void anotherEndpointInTheSameGroupAdvancesTheWaitingEdge() {
        CapacityNotifier capacityNotifier = new CapacityNotifier();
        CapacityKey waiting = CapacityKey.exact(RoleType.PREFILL, "g1", "127.0.0.1:8000");

        CapacityNotifierTestSupport.notifyChanged(
                capacityNotifier, CapacityKey.exact(RoleType.PREFILL, "g1", "127.0.0.2:8000"));

        assertTrue(capacityNotifier.lastChangedSequence(waiting) > 0L);
    }

    @Test
    void endpointTopologyChangeAdvancesItsPlacementEdge() {
        CapacityNotifier capacityNotifier = new CapacityNotifier();
        List<CapacityKey> changed = new ArrayList<>();
        capacityNotifier.addListener(changed::add);
        CapacityKey exact = CapacityKey.exact(RoleType.DECODE, "g1", "127.0.0.1:9000");

        CapacityNotifierTestSupport.notifyChanged(capacityNotifier, exact);

        assertEquals(1, changed.size());
        assertEquals(exact, changed.getFirst());
        assertTrue(capacityNotifier.lastChangedSequence(exact) > 0L);
    }

    @Test
    void endpointGroupChangesHaveIndependentGroupVersions() {
        var capacityNotifier = new CapacityNotifier();
        var oldGroup = CapacityKey.exact(RoleType.PREFILL, "old", "127.0.0.1:8000");
        var newGroup = CapacityKey.exact(RoleType.PREFILL, "new", oldGroup.endpoint());
        CapacityNotifierTestSupport.notifyChanged(capacityNotifier, oldGroup);
        CapacityNotifierTestSupport.notifyChanged(capacityNotifier, newGroup);

        assertEquals(1L, capacityNotifier.lastChangedSequence(oldGroup));
        assertEquals(2L, capacityNotifier.lastChangedSequence(newGroup));
        assertEquals(
                1L,
                capacityNotifier.lastChangedSequence(
                        new CapacityKey(RoleType.PREFILL, "old", null)));
        assertEquals(
                2L,
                capacityNotifier.lastChangedSequence(
                        new CapacityKey(RoleType.PREFILL, "new", null)));
        assertEquals(
                2L, capacityNotifier.lastChangedSequence(CapacityKey.anyGroup(RoleType.PREFILL)));
        assertEquals(
                2L,
                capacityNotifier.lastChangedSequence(
                        CapacityKey.exact(RoleType.PREFILL, "new", "127.0.0.2:8000")));
    }

    @Test
    void retirementFloorPreventsSleepingOnDiscardedHistory() {
        var capacityNotifier = new CapacityNotifier();
        var worker =
                org.flexlb.worker.status.WorkerStatus.createDiscovered(
                        RoleType.PREFILL, "g1", "127.0.0.1", 8000, 8001, null);
        var key = new CapacityKey(RoleType.PREFILL, "g1", null);
        capacityNotifier.notifyChanged(worker, worker.getLogicalIpPort());
        long observed = capacityNotifier.sequence();
        worker.lock.lock();
        try {
            worker.stopUpdatesAfterAdmissionClosed();
        } finally {
            worker.lock.unlock();
        }
        capacityNotifier.removeClosedWorker(worker);
        assertTrue(capacityNotifier.lastChangedSequence(key) > observed);
        assertTrue(
                capacityNotifier.lastChangedSequence(CapacityKey.anyGroup(RoleType.PREFILL))
                        > observed);
    }

    @Test
    void delayedPublisherCannotRegressGroupOrRoleVersion() {
        CapacityNotifier capacityNotifier = new CapacityNotifier();
        CapacityKey group = new CapacityKey(RoleType.PREFILL, "g1", null);
        CapacityKey role = CapacityKey.anyGroup(RoleType.PREFILL);
        CapacityKey olderExact = CapacityKey.exact(RoleType.PREFILL, "g1", "127.0.0.1:8000");
        CapacityKey newerExact = CapacityKey.exact(RoleType.PREFILL, "g1", "127.0.0.2:8000");
        var olderWorker = mock(org.flexlb.worker.status.WorkerStatus.class);
        var topology =
                org.flexlb.worker.status.WorkerStatus.createDiscovered(
                                RoleType.PREFILL, "g1", "127.0.0.1", 8000, 8001, null)
                        .topologySnapshot();
        when(olderWorker.getRole()).thenReturn(RoleType.PREFILL);
        when(olderWorker.topologySnapshot()).thenReturn(topology);
        var interleaved = new java.util.concurrent.atomic.AtomicBoolean();
        when(olderWorker.isActiveGeneration())
                .thenAnswer(
                        ignored -> {
                            if (interleaved.compareAndSet(false, true)) {
                                CapacityNotifierTestSupport.notifyChanged(
                                        capacityNotifier, newerExact);
                            }
                            assertEquals(2L, capacityNotifier.lastChangedSequence(group));
                            assertEquals(2L, capacityNotifier.lastChangedSequence(role));
                            return true;
                        });
        List<CapacityKey> events = new ArrayList<>();
        capacityNotifier.addListener(events::add);

        capacityNotifier.notifyChanged(olderWorker, olderExact.endpoint());

        assertAll(
                () ->
                        assertEquals(
                                2L,
                                capacityNotifier.lastChangedSequence(group),
                                "a delayed exact publication must preserve the newer group edge"),
                () ->
                        assertEquals(
                                2L,
                                capacityNotifier.lastChangedSequence(role),
                                "a delayed exact publication must preserve the newer role edge"));
        assertEquals(2L, capacityNotifier.lastChangedSequence(olderExact));
        assertEquals(2L, capacityNotifier.lastChangedSequence(newerExact));
        assertEquals(List.of(newerExact, olderExact), events);
    }
}
