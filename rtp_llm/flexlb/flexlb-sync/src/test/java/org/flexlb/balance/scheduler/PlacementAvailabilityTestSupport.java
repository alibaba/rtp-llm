package org.flexlb.balance.scheduler;

import org.flexlb.dao.master.WorkerStatus;

/**
 * Publish a worker capacity edge through the public notification boundary.
 */
final class PlacementAvailabilityTestSupport {
    private PlacementAvailabilityTestSupport() {}

    static void changed(PlacementAvailability availability, PlacementKey key) {
        var worker =
                WorkerStatus.createDiscovered(key.role(), key.group(), "fixture", 8000, 8001, null);
        availability.changed(
                worker, key.endpoint() == null ? worker.getLogicalIpPort() : key.endpoint());
    }
}
