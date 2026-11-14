package org.flexlb.config;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class DispatchTimeoutConfigTest {
    @Test
    void queueAndDirectShareOneAbsoluteBudget() {
        for (String mode : new String[]{"QUEUE", "DIRECT"}) {
            var config = ConfigService.parse("{\"scheduler\":{\"type\":\"" + mode
                    + "\"},\"dispatchTimeoutMs\":120000}");
            assertEquals(121000L, config.resolveExpiresAtMs(1000L));
            assertEquals(Long.MAX_VALUE, config.resolveExpiresAtMs(Long.MAX_VALUE - 10L));
        }
        assertEquals(60000L, ConfigService.parse("{}").getDispatchTimeoutMs());
    }

    @Test
    void timeoutRequiresPositiveIntegerAndOldConfigurationIsRejected() {
        for (String value : new String[]{"0", "-1", "null", "1.5", "true", "\"120000\""}) {
            assertThrows(ConfigValidationException.class,
                    () -> ConfigService.parse("{\"dispatchTimeoutMs\":" + value + "}"));
        }
        for (String legacy : new String[]{
                "{\"scheduler\":{\"queueTimeoutMs\":120000}}",
                "{\"scheduler\":{\"prefillQueueShedding\":{\"maxWaitMs\":120000}}}",
                "{\"requestLifecycle\":{\"request\":{\"timeoutMs\":120000}}}",
                "{\"internalRuntime\":{\"requestInactivityTimeoutMs\":120000}}"}) {
            assertThrows(ConfigValidationException.class, () -> ConfigService.parse(legacy));
        }
    }
}
