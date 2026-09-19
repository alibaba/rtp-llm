package org.flexlb.dao.loadbalance;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

class StrategyErrorTypeTest {

    @Test
    void admissionUnavailableIsRetainedForLegacyErrorCodeDecoding() {
        assertEquals(StrategyErrorType.ADMISSION_UNAVAILABLE,
                StrategyErrorType.fromErrorCode(8432));
    }

    @Test
    void currentCancellationCodeIsDecodedWithoutReplacingLegacyCodes() {
        assertEquals(StrategyErrorType.REQUEST_CANCELLED,
                StrategyErrorType.fromErrorCode(8504));
    }

    @Test
    void unknownErrorCodeRetainsNullResult() {
        assertNull(StrategyErrorType.fromErrorCode(-1));
        assertNull(StrategyErrorType.fromErrorCode(Integer.MAX_VALUE));
    }
}
