package org.flexlb.config;

import org.flexlb.config.loading.ConfigValidationException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Checks request admission defaults and rejects removed compute-token and batch controls.
 */
class FlexlbConfigV4Test {
    @Test
    void currentVersionHasRequestAdmissionAndLifecycleDefaults() {
        var config = ConfigTestFixtures.parse("{}");
        assertEquals(5, config.getSchemaVersion());
        assertEquals(
                128,
                config.getRouter().getRoles().getPrefill().getAdmission().getMaxInflightRequests());
        assertNull(config.getRouter().getRoles().getPrefill().getSelection().getCacheAffinity());
        assertEquals(60_000L, config.getRequestLifecycle().getDispatchTimeoutMs());
        assertEquals(60_000L, config.getRequestLifecycle().getInactivityTimeoutMs());
    }

    @Test
    void engineQueueDefaultsBelongToTokenBalanceAndRemovedDurationIsRejected() {
        var config = ConfigTestFixtures.parse("{}");
        assertNull(config.getRouter().getRoles().getPrefill().engineQueueAdmission());
        var token =
                ConfigTestFixtures.parse(
                        """
                {"router":{"roles":{"prefill":{"selection":{"type":"TOKEN_BALANCE"}}}}}
                """);
        assertEquals(
                10,
                token.getRouter()
                        .getRoles()
                        .getPrefill()
                        .engineQueueAdmission()
                        .getWaitingThreshold());
        assertThrows(
                ConfigValidationException.class,
                () ->
                        ConfigTestFixtures.parse(
                                """
                {"router":{"roles":{"prefill":{"selection":{"type":"TOKEN_BALANCE"},
                "admission":{"engineQueue":{"overloadDurationMs":5000}}}}}}
                """));
    }

    @Test
    void orderingAndPriorityComposeIndependently() {
        for (boolean priority : new boolean[] {true, false})
            for (String policy : new String[] {"FIFO", "PREFILL_COST"}) {
                var config =
                        ConfigTestFixtures.parse(
                                """
                    {"scheduler":{"type":"QUEUE","ordering":{"priority":{"enabled":%s},"policy":{"type":"%s"}}}}
                    """
                                        .formatted(priority, policy));
                assertEquals(
                        priority, config.queueScheduler().getOrdering().getPriority().isEnabled());
                assertEquals(
                        policy, config.queueScheduler().getOrdering().getPolicy().getType().name());
            }
    }

    @Test
    void selectionRejectsParametersOfAnotherPolicy() {
        for (String selection :
                new String[] {
                    "\"type\":\"TOKEN_BALANCE\",\"costExpression\":\"computeTokens\"",
                    "\"type\":\"TOKEN_BALANCE\",\"cacheAffinity\":{}",
                    "\"type\":\"COST_BASED\",\"minPrefixHitPercent\":5"
                }) {
            assertThrows(
                    ConfigValidationException.class,
                    () ->
                            ConfigTestFixtures.parse(
                                    "{\"router\":{\"roles\":{\"prefill\":{\"selection\":{"
                                            + selection
                                            + "}}}}}"));
        }
        assertThrows(
                ConfigValidationException.class,
                () ->
                        ConfigTestFixtures.parse(
                                """
                {"router":{"roles":{"prefill":{"admission":{"engineQueue":{}}}}}}
                """));
        var config =
                ConfigTestFixtures.parse(
                        """
                {"scheduler":{"type":"DIRECT"},"router":{"roles":{"prefill":{
                  "selection":{"type":"TOKEN_BALANCE"},"admission":{"maxInflightRequests":64,"engineQueue":{"waitingThreshold":3}}}}}}
                """);
        assertEquals(
                3,
                config.getRouter()
                        .getRoles()
                        .getPrefill()
                        .getAdmission()
                        .getEngineQueue()
                        .getWaitingThreshold());
    }

    @Test
    void parsesRequestAdmissionAndRejectsFormerTokenCaps() {
        var config =
                ConfigTestFixtures.parse(
                        """
                {"router":{"roles":{"prefill":{"admission":{"maxInflightRequests":32}}}}}
                """);
        assertEquals(
                32,
                config.getRouter().getRoles().getPrefill().getAdmission().getMaxInflightRequests());
        assertThrows(
                ConfigValidationException.class,
                () ->
                        ConfigTestFixtures.parse(
                                """
                {"router":{"roles":{"prefill":{"maxComputeTokens":1000}}}}
                """));
    }

    @Test
    void rejectsNonpositiveComputeTokenLimits() {
        for (String value : new String[] {"0", "-1"}) {
            assertThrows(
                    ConfigValidationException.class,
                    () ->
                            ConfigTestFixtures.parse(
                                    "{\"router\":{\"roles\":{\"prefill\":{\"maxComputeTokens\":"
                                            + value
                                            + "}}}}"));
            assertThrows(
                    ConfigValidationException.class,
                    () ->
                            ConfigTestFixtures.parse(
                                    "{\"router\":{\"roles\":{\"prefill\":{\"cacheAffinity\":{\"maxComputeTokens\":"
                                            + value
                                            + "}}}}}"));
        }
    }

    @Test
    void rejectsRemovedConfigurationInsteadOfSilentlyIgnoringIt() {
        for (String patch :
                new String[] {
                    "{\"dispatcher\":{\"type\":\"BATCH\"}}",
                    "{\"scheduler\":{\"type\":\"QUEUE\",\"decision\":{\"type\":\"SINGLE\"}}}",
                    "{\"requestLifecycle\":{\"decision\":{\"lifetime\":2}}}",
                    "{\"router\":{\"roles\":{\"prefill\":{\"executionTimeEstimator\":{\"type\":\"LEARNING\"}}}}}",
                    "{\"schemaVersion\":3}"
                }) {
            assertThrows(ConfigValidationException.class, () -> ConfigTestFixtures.parse(patch));
        }
    }
}
