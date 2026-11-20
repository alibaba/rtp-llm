package org.flexlb.worker.endpoint;

import org.flexlb.config.service.RoleType;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.flexlb.worker.status.WorkerStatusTestSupport.createDiscovered;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Verifies capacity notification delivery and listener removal without retaining capacity history.
 */
class CapacityNotifierTest {
    @Test
    void deliversChangesAndStopsAfterListenerRemoval() {
        var notifier = new CapacityNotifier();
        List<WorkerEndpoint> changes = new ArrayList<>();
        CapacityNotifier.Listener listener = changes::add;
        notifier.addListener(listener);
        var endpoint =
                new WorkerEndpoint(
                        createDiscovered(RoleType.PREFILL, "g1", "localhost", 8000, 8001, null));
        notifier.notifyChanged(endpoint);
        notifier.removeListener(listener);
        notifier.notifyChanged(endpoint);
        assertEquals(List.of(endpoint), changes);
    }

    @Test
    void failingListenerDoesNotPreventOtherListenersReceivingChange() {
        var notifier = new CapacityNotifier();
        notifier.addListener(
                key -> {
                    throw new IllegalStateException("listener failed");
                });
        List<WorkerEndpoint> changes = new ArrayList<>();
        notifier.addListener(changes::add);
        var endpoint =
                new WorkerEndpoint(
                        createDiscovered(RoleType.DECODE, "g1", "localhost", 9000, 9001, null));
        notifier.notifyChanged(endpoint);
        assertEquals(List.of(endpoint), changes);
    }
}
