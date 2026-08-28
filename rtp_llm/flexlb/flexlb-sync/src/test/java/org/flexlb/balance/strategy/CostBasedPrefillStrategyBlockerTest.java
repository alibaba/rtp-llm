package org.flexlb.balance.strategy;

import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

/** Regression contracts for Decode capacity projected through Prefill. */
class CostBasedPrefillStrategyBlockerTest {

    @Test
    void everyRegisteredEndpointMustProveTheSameBlocker() {
        assertNull(CostBasedPrefillStrategy.provenPoolWideBlocker(
                Map.of(RoleType.DECODE, 3), 4));
        assertNull(CostBasedPrefillStrategy.provenPoolWideBlocker(
                Map.of(RoleType.DECODE, 2, RoleType.PREFILL, 2), 4));
        assertEquals(
                RoleType.DECODE,
                CostBasedPrefillStrategy.provenPoolWideBlocker(
                        Map.of(RoleType.DECODE, 4), 4));
    }
}
