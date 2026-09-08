package org.flexlb.util;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import uk.org.webcompere.systemstubs.environment.EnvironmentVariables;
import uk.org.webcompere.systemstubs.jupiter.SystemStub;
import uk.org.webcompere.systemstubs.jupiter.SystemStubsExtension;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

@ExtendWith(SystemStubsExtension.class)
class EnvUtilsTest {

    @SystemStub
    private EnvironmentVariables environmentVariables = new EnvironmentVariables();

    @Test
    void shouldParsePositiveLongOrFallback() {
        assertEquals(2500L, EnvUtils.parsePositiveLong("TIMEOUT", " 2500 ", 1000L));
        assertEquals(1000L, EnvUtils.parsePositiveLong("TIMEOUT", null, 1000L));
        assertEquals(1000L, EnvUtils.parsePositiveLong("TIMEOUT", "invalid", 1000L));
        assertEquals(1000L, EnvUtils.parsePositiveLong("TIMEOUT", "0", 1000L));
        assertEquals(1000L, EnvUtils.parsePositiveLong("TIMEOUT", "-1", 1000L));
        assertEquals(1000L, EnvUtils.parsePositiveLong("TIMEOUT", "", 1000L));
        assertEquals(1000L, EnvUtils.parsePositiveLong("TIMEOUT", "   ", 1000L));
        assertEquals(1000L,
                EnvUtils.parsePositiveLong("TIMEOUT", "9223372036854775808", 1000L));
    }

    @Test
    void shouldParseBooleanValuesAndFallbackForInvalidInput() {
        assertTrue(EnvUtils.parseBoolean("FLAG", " true ", false));
        assertTrue(EnvUtils.parseBoolean("FLAG", "1", false));
        assertTrue(EnvUtils.parseBoolean("FLAG", "YES", false));
        assertTrue(EnvUtils.parseBoolean("FLAG", "on", false));
        assertFalse(EnvUtils.parseBoolean("FLAG", "FALSE", true));
        assertFalse(EnvUtils.parseBoolean("FLAG", "0", true));
        assertFalse(EnvUtils.parseBoolean("FLAG", "No", true));
        assertFalse(EnvUtils.parseBoolean("FLAG", "OFF", true));
        assertTrue(EnvUtils.parseBoolean("FLAG", null, true));
        assertFalse(EnvUtils.parseBoolean("FLAG", "invalid", false));
    }

    @Test
    void shouldReadBooleanFromEnvironment() {
        environmentVariables.set("RTP_LLM_TEST_BOOLEAN_FLAG", "0");

        assertFalse(EnvUtils.readBoolean("RTP_LLM_TEST_BOOLEAN_FLAG", true));
        assertTrue(EnvUtils.readBoolean("RTP_LLM_TEST_MISSING_BOOLEAN_FLAG", true));
    }
}
