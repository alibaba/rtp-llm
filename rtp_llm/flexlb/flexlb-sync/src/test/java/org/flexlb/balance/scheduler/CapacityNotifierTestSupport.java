package org.flexlb.balance.scheduler;

import org.flexlb.dao.master.WorkerStatus;

/**
 * Publish a worker capacity edge through the public notification boundary.
 */
final class CapacityNotifierTestSupport {
    private CapacityNotifierTestSupport() {}

    static void notifyChanged(CapacityNotifier capacityNotifier, CapacityKey key) {
        var worker =
                WorkerStatus.createDiscovered(key.role(), key.group(), "fixture", 8000, 8001, null);
        capacityNotifier.notifyChanged(
                worker, key.endpoint() == null ? worker.getLogicalIpPort() : key.endpoint());
    }
}
