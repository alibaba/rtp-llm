package org.flexlb.dao.loadbalance;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class StrategyErrorTypeTest {

    @Test
    void admissionUnavailableIsRetainedForLegacyErrorCodeDecoding() {
        assertEquals(StrategyErrorType.ADMISSION_UNAVAILABLE,
                StrategyErrorType.fromErrorCode(8432));
    }
}
