package org.flexlb.dispatcher;

import org.flexlb.exception.FlexLBException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DispatchConfigTest {

    @Test
    void parsesFullJson() {
        DispatchConfig c = DispatchConfig.fromJson(
                "{\"enabled\":true,\"subBatchSize\":5,"
                        + "\"feRequestTimeoutMs\":3000,\"fePoolServiceId\":\"com.rtp_llm.fe\"}");
        assertTrue(c.isEnabled());
        assertEquals(5, c.getSubBatchSize());
        assertEquals(3000, c.getFeRequestTimeoutMs());
        assertEquals("com.rtp_llm.fe", c.getFePoolServiceId());
    }

    @Test
    void disabledWhenEnvNullOrBlank() {
        assertFalse(DispatchConfig.fromJson(null).isEnabled());
        assertFalse(DispatchConfig.fromJson("  ").isEnabled());
    }

    @Test
    void rejectsEnabledWithoutFePoolServiceId() {
        assertThrows(IllegalArgumentException.class,
                () -> DispatchConfig.fromJson("{\"enabled\":true}"));
    }

    @Test
    void rejectsEnabledWithBlankFePoolServiceId() {
        assertThrows(IllegalArgumentException.class,
                () -> DispatchConfig.fromJson("{\"enabled\":true,\"fePoolServiceId\":\"  \"}"));
    }

    @Test
    void rejectsNonPositiveSubBatchSize() {
        assertThrows(IllegalArgumentException.class,
                () -> DispatchConfig.fromJson(
                        "{\"enabled\":true,\"fePoolServiceId\":\"com.rtp_llm.fe\",\"subBatchSize\":0}"));
    }

    @Test
    void rejectsMalformedJson() {
        assertThrows(FlexLBException.class, () -> DispatchConfig.fromJson("{not json}"));
    }

    @Test
    void sameJvmIsolationKnobsDefaultSensibly() {
        DispatchConfig c = DispatchConfig.fromJson(null);
        assertEquals(200, c.getFeMaxConnections());
        assertEquals(1000, c.getFeMaxPendingAcquire());
        assertEquals(16 * 1024 * 1024, c.getFeMaxResponseBytes());
    }

    @Test
    void parsesNewTimeoutFields() {
        DispatchConfig c = DispatchConfig.fromJson(
                "{\"enabled\":true,\"fePoolServiceId\":\"com.rtp_llm.fe\","
                        + "\"feConnectTimeoutMs\":1500,\"feResponseTimeoutMs\":7500,"
                        + "\"feMaxStreamDurationMs\":300000}");
        assertEquals(1500, c.getFeConnectTimeoutMs());
        assertEquals(7500, c.getFeResponseTimeoutMs());
        assertEquals(300_000, c.getFeMaxStreamDurationMs());
    }

    @Test
    void newTimeoutFieldsHaveSensibleDefaults() {
        DispatchConfig c = DispatchConfig.fromJson(null);
        assertEquals(2000, c.getFeConnectTimeoutMs());
        assertEquals(5000, c.getFeResponseTimeoutMs());
        assertEquals(600_000, c.getFeMaxStreamDurationMs());
    }

    @Test
    void sameJvmIsolationKnobsOverridableViaJson() {
        DispatchConfig c = DispatchConfig.fromJson(
                "{\"enabled\":true,\"fePoolServiceId\":\"com.rtp_llm.fe\","
                        + "\"feMaxConnections\":64,\"feMaxPendingAcquire\":256,"
                        + "\"feMaxResponseBytes\":1048576}");
        assertEquals(64, c.getFeMaxConnections());
        assertEquals(256, c.getFeMaxPendingAcquire());
        assertEquals(1048576, c.getFeMaxResponseBytes());
    }
}
