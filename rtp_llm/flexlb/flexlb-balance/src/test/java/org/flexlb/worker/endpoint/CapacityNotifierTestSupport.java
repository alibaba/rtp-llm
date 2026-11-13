package org.flexlb.worker.endpoint;

import org.flexlb.worker.status.WorkerStatus;

/**
 * Publish a worker capacity edge through the public notification boundary.
 */
public final class CapacityNotifierTestSupport {
    private CapacityNotifierTestSupport() {}

    public static void notifyChanged(CapacityNotifier capacityNotifier, CapacityKey key) {
        var worker =
                WorkerStatus.createDiscovered(key.role(), key.group(), "fixture", 8000, 8001, null);
        capacityNotifier.notifyChanged(
                worker, key.endpoint() == null ? worker.getLogicalIpPort() : key.endpoint());
    }
}
