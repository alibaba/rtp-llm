package org.flexlb.dispatcher;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DispatchConfigTest {

    @Test
    void parsesFullJson() {
        DispatchConfig c = DispatchConfig.fromJson(
                "{\"enabled\":true,\"dispatchPort\":7005,\"subBatchSize\":5,"
                        + "\"feRequestTimeoutMs\":3000,\"fePoolAddresses\":[\"http://a:8088\"]}");
        assertTrue(c.isEnabled());
        assertEquals(7005, c.getDispatchPort());
        assertEquals(5, c.getSubBatchSize());
        assertEquals(3000, c.getFeRequestTimeoutMs());
        assertEquals(1, c.getFePoolAddresses().size());
    }

    @Test
    void disabledWhenEnvNullOrBlank() {
        assertFalse(DispatchConfig.fromJson(null).isEnabled());
        assertFalse(DispatchConfig.fromJson("  ").isEnabled());
    }

    @Test
    void rejectsEnabledWithoutFePool() {
        assertThrows(IllegalArgumentException.class,
                () -> DispatchConfig.fromJson("{\"enabled\":true,\"dispatchPort\":7005}"));
    }

    @Test
    void sameJvmIsolationKnobsDefaultSensibly() {
        DispatchConfig c = DispatchConfig.fromJson(null);
        assertEquals(200, c.getFeMaxConnections());
        assertEquals(1000, c.getFeMaxPendingAcquire());
        assertEquals(16 * 1024 * 1024, c.getFeMaxResponseBytes());
    }

    @Test
    void sameJvmIsolationKnobsOverridableViaJson() {
        DispatchConfig c = DispatchConfig.fromJson(
                "{\"enabled\":true,\"fePoolAddresses\":[\"http://a:8088\"],"
                        + "\"feMaxConnections\":64,\"feMaxPendingAcquire\":256,"
                        + "\"feMaxResponseBytes\":1048576}");
        assertEquals(64, c.getFeMaxConnections());
        assertEquals(256, c.getFeMaxPendingAcquire());
        assertEquals(1048576, c.getFeMaxResponseBytes());
    }
}
