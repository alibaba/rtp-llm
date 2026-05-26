package org.flexlb.dispatcher;

import org.flexlb.exception.FlexLBException;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DispatchConfigTest {

    @Test
    void defaultsAreSensible() {
        DispatchConfig c = DispatchConfig.fromJson(null);
        assertFalse(c.isEnabled());
        assertEquals("count:5", c.getSubBatch());
        assertEquals("", c.getFePoolServiceId());
        assertEquals(5000, c.getBatchTimeoutMs());
        assertEquals(200, c.getFeMaxConnectionsPerHost());
        assertEquals(1000, c.getFeMaxPendingAcquirePerHost());
    }

    @Test
    void parsesFullJson() {
        DispatchConfig c = DispatchConfig.fromJson(
                "{\"enabled\":true,\"subBatch\":\"size:10\","
                        + "\"fePoolServiceId\":\"com.rtp_llm.fe\","
                        + "\"batchTimeoutMs\":7500,"
                        + "\"feMaxConnectionsPerHost\":64,"
                        + "\"feMaxPendingAcquirePerHost\":256}");
        assertTrue(c.isEnabled());
        assertEquals("size:10", c.getSubBatch());
        assertEquals("com.rtp_llm.fe", c.getFePoolServiceId());
        assertEquals(7500, c.getBatchTimeoutMs());
        assertEquals(64, c.getFeMaxConnectionsPerHost());
        assertEquals(256, c.getFeMaxPendingAcquirePerHost());
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
    void rejectsInvalidSubBatchDsl() {
        // subBatch is parsed eagerly during validate() so a malformed DSL fails fast at boot
        assertThrows(IllegalArgumentException.class,
                () -> DispatchConfig.fromJson(
                        "{\"enabled\":true,\"fePoolServiceId\":\"x\",\"subBatch\":\"foo:5\"}"));
        assertThrows(IllegalArgumentException.class,
                () -> DispatchConfig.fromJson(
                        "{\"enabled\":true,\"fePoolServiceId\":\"x\",\"subBatch\":\"count:0\"}"));
    }

    @Test
    void rejectsMalformedJson() {
        assertThrows(FlexLBException.class, () -> DispatchConfig.fromJson("{not json}"));
    }

    @Test
    void unknownJsonFieldsIgnoredSoOldConfigsDoNotBreak() {
        // Old field names that have been deleted (subBatchSize, feRequestTimeoutMs,
        // feConnectTimeoutMs, feResponseTimeoutMs, feMaxStreamDurationMs, feMaxResponseBytes)
        // must not crash — operators with stale DISPATCH_CONFIG should keep starting up.
        DispatchConfig c = DispatchConfig.fromJson(
                "{\"enabled\":true,\"fePoolServiceId\":\"x\","
                        + "\"subBatchSize\":7,\"feRequestTimeoutMs\":4000,"
                        + "\"feConnectTimeoutMs\":1234,\"feResponseTimeoutMs\":6000,"
                        + "\"feMaxStreamDurationMs\":120000,\"feMaxResponseBytes\":2097152}");
        assertTrue(c.isEnabled());
        assertEquals("count:5", c.getSubBatch(), "subBatchSize is now unknown — default kept");
        assertEquals(5000, c.getBatchTimeoutMs(), "feResponseTimeoutMs is unknown — default kept");
    }

    @Test
    void envOverridesJsonForEachField() {
        Map<String, String> env = Map.of(
                "DISPATCH_BATCH_TIMEOUT_MS", "8000",
                "DISPATCH_FE_MAX_CONNECTIONS_PER_HOST", "300",
                "DISPATCH_FE_MAX_PENDING_ACQUIRE_PER_HOST", "1500");
        DispatchConfig c = DispatchConfig.fromJsonWithEnv(
                "{\"enabled\":true,\"fePoolServiceId\":\"x\",\"batchTimeoutMs\":5000}", env);
        assertEquals(8000, c.getBatchTimeoutMs(), "env wins over JSON");
        assertEquals(300, c.getFeMaxConnectionsPerHost());
        assertEquals(1500, c.getFeMaxPendingAcquirePerHost());
    }

    @Test
    void envOverridesDefaultsWhenNoJson() {
        Map<String, String> env = Map.of(
                "DISPATCH_ENABLED", "true",
                "DISPATCH_FE_POOL_SERVICE_ID", "from.env",
                "DISPATCH_SUB_BATCH", "count:10");
        DispatchConfig c = DispatchConfig.fromJsonWithEnv(null, env);
        assertTrue(c.isEnabled());
        assertEquals("from.env", c.getFePoolServiceId());
        assertEquals("count:10", c.getSubBatch());
    }

    @Test
    void subBatchSpecAccessorParsesEagerly() {
        DispatchConfig c = DispatchConfig.fromJson(
                "{\"enabled\":true,\"fePoolServiceId\":\"x\",\"subBatch\":\"count:7\"}");
        SubBatchSpec spec = c.subBatchSpec();
        assertEquals(SubBatchSpec.Mode.COUNT, spec.mode());
        assertEquals(7, spec.value());
    }
}
