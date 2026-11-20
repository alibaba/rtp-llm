package org.flexlb.config;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PrefillLeaseConfigTest {
    @Test
    void batchStepsDefaultToOneAndRejectNonPositiveValues() {
        var defaults = ConfigTestFixtures.parse("{}").getRouter().getRoles().getPrefill().getChunkLease();
        assertEquals(1, defaults.getScaleUpStep());
        assertEquals(1, defaults.getScaleDownStep());
        String template = """
                {"dispatcher":{"type":"NON_BATCH"},"router":{"roles":{"prefill":{
                  "chunkBatchingEnabled":true,"chunkLease":{"enabled":true,"groups":["test"],
                  "scaleUpStep":%d,"scaleDownStep":%d}}}}}
                """;
        var configured =
                ConfigTestFixtures.parse(template.formatted(3, 2)).getRouter().getRoles().getPrefill().getChunkLease();
        assertEquals(3, configured.getScaleUpStep());
        assertEquals(2, configured.getScaleDownStep());
        assertThrows(ConfigValidationException.class, () -> ConfigTestFixtures.parse(template.formatted(0, 1)));
        assertThrows(ConfigValidationException.class, () -> ConfigTestFixtures.parse(template.formatted(1, -1)));
    }

    @Test
    void disabledByDefaultAndManualModeNeedsNoMetricCredentials() {
        assertFalse(ConfigTestFixtures.parse("{}").getRouter().getRoles().getPrefill().getChunkLease().isEnabled());
        var config = ConfigTestFixtures.parse("""
                {"dispatcher":{"type":"NON_BATCH"},"router":{"roles":{"prefill":{
                  "chunkBatchingEnabled":true,"chunkLease":{"enabled":true,"groups":["test"]}
                }}}}
                """);
        assertTrue(config.getRouter().getRoles().getPrefill().getChunkLease().isEnabled());
    }

    @Test
    void automaticControlUsesWorkerStatusWithoutMetricCredentials() {
        var config = ConfigTestFixtures.parse("""
                {"dispatcher":{"type":"NON_BATCH"},"router":{"roles":{"prefill":{
                  "chunkBatchingEnabled":true,"chunkLease":{"enabled":true,"automatic":true,"groups":["test"]}
                }}}}
                """);
        assertTrue(config.getRouter().getRoles().getPrefill().getChunkLease().isAutomatic());
    }

    @Test
    void rejectsWindowsLargerThanRetainedWorkerStatusHistory() {
        assertThrows(ConfigValidationException.class, () -> ConfigTestFixtures.parse("""
                {"dispatcher":{"type":"NON_BATCH"},"router":{"roles":{"prefill":{
                  "chunkBatchingEnabled":true,"chunkLease":{"enabled":true,"automatic":true,"groups":["test"],
                    "windowSeconds":601}
                }}}}
                """));
    }

    @Test
    void requiresChunkModeAndOrderedPositiveBounds() {
        assertThrows(ConfigValidationException.class, () -> ConfigTestFixtures.parse("""
                {"router":{"roles":{"prefill":{"chunkLease":{"enabled":true,"groups":["test"]}}}}}
                """));
        assertThrows(ConfigValidationException.class, () -> ConfigTestFixtures.parse("""
                {"dispatcher":{"type":"NON_BATCH"},"router":{"roles":{"prefill":{
                  "chunkBatchingEnabled":true,"chunkLease":{"enabled":true,"groups":["test"],
                    "ttftLowerMs":200,"ttftUpperMs":100}
                }}}}
                """));
    }
}
