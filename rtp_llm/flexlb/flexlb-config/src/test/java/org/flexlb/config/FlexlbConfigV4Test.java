package org.flexlb.config;

import org.flexlb.config.loading.ConfigValidationException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Checks V4 compute-token configuration and rejects removed batch and timing controls.
 */
class FlexlbConfigV4Test {
    @Test
    void currentVersionAndComputeTokenDefaultsAreV4() {
        var config = ConfigTestFixtures.parse("{}");
        assertEquals(4, config.getSchemaVersion());
        assertEquals(1048576L, config.getRouter().getRoles().getPrefill().getMaxComputeTokens());
        assertNull(config.getRouter().getRoles().getPrefill().getCacheAffinity());
    }

    @Test
    void parsesWorkerAndCacheFirstComputeTokenLimits() {
        var config =
                ConfigTestFixtures.parse(
                        """
                {"router":{"roles":{"prefill":{"maxComputeTokens":1000,
                "cacheAffinity":{"maxComputeTokens":500}}}}}
                """);
        assertEquals(1000L, config.getRouter().getRoles().getPrefill().getMaxComputeTokens());
        assertEquals(
                500L,
                config.getRouter()
                        .getRoles()
                        .getPrefill()
                        .getCacheAffinity()
                        .getMaxComputeTokens());
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
