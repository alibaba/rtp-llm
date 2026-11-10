package org.flexlb.config;

import org.flexlb.dao.route.RequestPhase;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Checks phase-specific delivery and grouping without changing legacy configuration defaults.
 */
class PhasePolicyConfigTest {
    @Test
    void legacyConfigurationResolvesEncoderAndGenerationIndependently() {
        var config = ConfigTestFixtures.parse("{}");
        assertEquals(DispatcherConfig.Type.NON_BATCH, config.deliveryType(RequestPhase.ENCODER));
        assertEquals(
                DecisionPolicyConfig.Type.SINGLE,
                config.decisionPolicy(RequestPhase.ENCODER).getType());
        assertEquals(
                config.getDispatcher().getType(), config.deliveryType(RequestPhase.GENERATION));
    }

    @Test
    void generationCanPublishRoutesWithAnExplicitWindow() {
        var config =
                ConfigTestFixtures.parse(
                        """
                {"phases":{"generation":{"decision":{"type":"FIXED_WINDOW","maxRequests":3},
                 "delivery":{"type":"ROUTE"}}}}
                """);
        assertEquals(DispatcherConfig.Type.NON_BATCH, config.deliveryType(RequestPhase.GENERATION));
        assertEquals(3, config.decisionPolicy().getMaxRequests());
        assertEquals(
                DecisionPolicyConfig.Type.SINGLE,
                config.decisionPolicy(RequestPhase.ENCODER).getType());
    }

    @Test
    void unsupportedEncoderTransportFailsDuringConfigurationLoading() {
        assertThrows(
                ConfigValidationException.class,
                () ->
                        ConfigTestFixtures.parse(
                                """
                {"phases":{"encoder":{"delivery":{"type":"ENQUEUE_BATCH"}}}}
                """));
    }

    @Test
    void directSchedulingUsesTheExplicitRouteDeliveryPolicy() {
        var config =
                ConfigTestFixtures.parse(
                        """
                {"scheduler":{"type":"DIRECT"},
                 "phases":{"generation":{"delivery":{"type":"ROUTE"}}}}
                """);
        assertEquals(DispatcherConfig.Type.NON_BATCH, config.deliveryType(RequestPhase.GENERATION));
    }
}
