package org.flexlb.config;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NettyMemoryReporterConfigTest {
    @Test
    void diagnosticsAreDisabledByDefault() {
        var memory = ConfigTestFixtures.parse("{}").getScheduler().getNettyMemoryReporter();
        assertFalse(memory.isEnabled());
        assertFalse(memory.getAutoCapture().isEnabled());
        assertEquals(10_000L, memory.getIntervalMs());
        assertEquals(5L * 1024 * 1024 * 1024, memory.getAutoCapture().getThresholdBytes());
        assertEquals(300_000L, memory.getAutoCapture().getCooldownMs());
    }

    @Test
    void strictConfigSupportsBothSchedulerModesAndIndependentCaptureOptIn() {
        for (String type : new String[] {"QUEUE", "DIRECT"}) {
            var config =
                    ConfigTestFixtures.parse(
                            """
                    {"scheduler":{"type":"%s","nettyMemoryReporter":{
                      "enabled":true,"intervalMs":20000}},"dispatcher":{"type":"NON_BATCH"}}
                    """
                                    .formatted(type));
            var memory = config.getScheduler().getNettyMemoryReporter();
            assertTrue(memory.isEnabled());
            assertEquals(20_000L, memory.getIntervalMs());
            assertFalse(memory.getAutoCapture().isEnabled());
        }
        var memory =
                ConfigTestFixtures.parse(
                                """
                {"scheduler":{"nettyMemoryReporter":{"enabled":true,"autoCapture":{
                  "enabled":true,"thresholdBytes":123456,"cooldownMs":60000}}}}
                """)
                        .getScheduler()
                        .getNettyMemoryReporter();
        assertTrue(memory.getAutoCapture().isEnabled());
        assertEquals(123456L, memory.getAutoCapture().getThresholdBytes());
        assertEquals(60_000L, memory.getAutoCapture().getCooldownMs());
    }

    @Test
    void invalidDiagnosticsFailAtStartupInsteadOfStartingUnexpectedWork() {
        for (String patch :
                new String[] {
                    "{\"autoCapture\":{\"enabled\":true}}",
                    "{\"intervalMs\":0}",
                    "{\"intervalMs\":999}",
                    "{\"enabled\":\"true\"}",
                    "{\"enabled\":null}",
                    "{\"autoCapture\":{\"thresholdBytes\":0}}",
                    "{\"autoCapture\":{\"cooldownMs\":999}}",
                    "{\"autoCapture\":{\"heapDump\":true}}"
                }) {
            var error =
                    assertThrows(
                            ConfigValidationException.class,
                            () ->
                                    ConfigTestFixtures.parse(
                                            "{\"scheduler\":{\"nettyMemoryReporter\":"
                                                    + patch
                                                    + "}}"));
            assertTrue(error.getMessage().contains("nettyMemoryReporter"), error.getMessage());
        }
    }
}
