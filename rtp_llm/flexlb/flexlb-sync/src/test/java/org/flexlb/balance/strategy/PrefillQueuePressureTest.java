package org.flexlb.balance.strategy;

import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PrefillQueuePressureTest {
    @Test
    void strictMajorityMustPersistForTenSeconds() {
        var pressure = new PrefillQueuePressure();
        for (long ms = 0; ms < 10_000; ms += 100) {
            assertFalse(pressure.observe(RoleType.PREFILL, null, 4, 3, ms));
        }
        assertFalse(pressure.observe(RoleType.PREFILL, null, 4, 3, 9_999));
        assertTrue(pressure.observe(RoleType.PREFILL, null, 4, 3, 10_000));
        assertFalse(pressure.observe(RoleType.PREFILL, null, 4, 2, 10_001));
        for (long ms = 10_002; ms < 20_002; ms += 100) {
            assertFalse(pressure.observe(RoleType.PREFILL, null, 4, 3, ms));
        }
        assertTrue(pressure.observe(RoleType.PREFILL, null, 4, 3, 20_002));
    }

    @Test
    void groupsAndRolesHaveIndependentPressureWindows() {
        var pressure = new PrefillQueuePressure();
        for (long ms = 0; ms <= 10_000; ms += 100) {
            pressure.observe(RoleType.PREFILL, null, 3, 2, ms);
            assertFalse(pressure.observe(RoleType.PREFILL, "idle", 3, 1, ms));
        }
        assertTrue(pressure.observe(RoleType.PREFILL, null, 3, 2, 10_000));
        assertFalse(pressure.observe(RoleType.PREFILL, "other", 3, 2, 10_000));
        assertFalse(pressure.observe(RoleType.PDFUSION, null, 3, 2, 10_000));
    }

    @Test
    void emptyPoolsResetPressureAndOneWorkerStillNeedsTenSeconds() {
        var pressure = new PrefillQueuePressure();
        assertFalse(pressure.observe(RoleType.PREFILL, null, 1, 1, 0));
        assertTrue(pressure.observe(RoleType.PREFILL, null, 1, 1, 10_000));
        assertFalse(pressure.observe(RoleType.PREFILL, null, 0, 0, 10_001));
        assertFalse(pressure.observe(RoleType.PREFILL, null, 1, 1, 10_002));
        assertFalse(pressure.observe(RoleType.PREFILL, null, 1, 1, 20_001));
        assertTrue(pressure.observe(RoleType.PREFILL, null, 1, 1, 20_002));
    }
}
