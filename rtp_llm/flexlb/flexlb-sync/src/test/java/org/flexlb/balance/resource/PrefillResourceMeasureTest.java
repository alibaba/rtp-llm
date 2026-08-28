package org.flexlb.balance.resource;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Tests for {@link PrefillResourceMeasure}.
 *
 * <p>The generic availability path consumes the endpoint's churn-safe admission
 * count. Projection-aware callers use the explicit-count overload so their
 * availability decision and TTFT share one coherent observation.
 */
@ExtendWith(MockitoExtension.class)
class PrefillResourceMeasureTest {

    @Mock
    private ConfigService configService;

    private FlexlbConfig config;

    @BeforeEach
    void setUp() {
        config = new FlexlbConfig();
        config.getRouter().getRoles().getPrefill().getAvailability()
                .setMaxPendingRequests(2);
        config.getRouter().setAvailabilityHysteresisPercent(50);
        when(configService.loadBalanceConfig()).thenReturn(config);
    }

    @Test
    void coherentPendingCountDrivesAvailabilityWithoutReadingEndpointAgain() {
        PrefillResourceMeasure measure = new PrefillResourceMeasure(configService);

        assertFalse(measure.isResourceAvailable(2L),
                "the coherent count at the upper threshold must close admission");
        assertTrue(measure.isResourceAvailable(1L),
                "the coherent count at the lower threshold must reopen admission");
        assertThrows(IllegalArgumentException.class,
                () -> measure.isResourceAvailable(-1L));
    }

    @Test
    void callerPassesTheEndpointChurnSafeAdmissionCount() {
        PrefillResourceMeasure measure = new PrefillResourceMeasure(configService);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.admissionPendingRequestCount()).thenReturn(1L);

        assertTrue(measure.isResourceAvailable(
                endpoint.admissionPendingRequestCount()));
        verify(endpoint).admissionPendingRequestCount();
    }

}
