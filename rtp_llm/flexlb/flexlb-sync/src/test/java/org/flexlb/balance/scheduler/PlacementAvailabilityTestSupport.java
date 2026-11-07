package org.flexlb.balance.scheduler;

import org.flexlb.dao.master.WorkerStatus;
import org.springframework.test.util.ReflectionTestUtils;

/** Test-only capacity edge driver for cases without a concrete WorkerStatus. */
final class PlacementAvailabilityTestSupport {

    private PlacementAvailabilityTestSupport() {}

    static void changed(PlacementAvailability availability, PlacementKey key) {
        ReflectionTestUtils.invokeMethod(availability, "changed", key, (WorkerStatus) null);
    }
}
