package org.flexlb.worker.endpoint;

import static org.flexlb.worker.status.WorkerStatusTestSupport.createDiscovered;

/**
 * Publish a worker capacity edge through the public notification boundary.
 */
public final class CapacityNotifierTestSupport {
    private CapacityNotifierTestSupport() {}

    public static void notifyChanged(CapacityNotifier capacityNotifier, CapacityKey key) {
        var worker = createDiscovered(key.role(), key.group(), "fixture", 8000, 8001, null);
        capacityNotifier.notifyChanged(new WorkerEndpoint(worker));
    }
}
