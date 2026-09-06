package org.flexlb.dao.loadbalance;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class StrategyErrorTypeTest {

    @Test
    void admissionUnavailableIsRetainedForLegacyErrorCodeDecoding() {
        assertEquals(StrategyErrorType.ADMISSION_UNAVAILABLE,
                StrategyErrorType.fromErrorCode(8432));
    }

    @Test
    void fallbackIsRetainedForClientDomainRouting() {
        assertEquals(StrategyErrorType.FALLBACK,
                StrategyErrorType.fromErrorCode(8600));
        assertEquals("FALLBACK", StrategyErrorType.FALLBACK.buildErrorMessage(null));
    }
}
