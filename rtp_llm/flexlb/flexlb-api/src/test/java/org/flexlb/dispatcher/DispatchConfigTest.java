package org.flexlb.dispatcher;

import org.flexlb.exception.FlexLBException;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Exercises the loading + validation pipeline now homed in
 * {@link DispatcherConfiguration#loadAndValidate}. Mirrors the {@code FlexlbConfig} /
 * {@code ConfigService} convention: data class is a dumb POJO, loading lives in the loader.
 */
class DispatchConfigTest {

    @Test
    void defaultsAreSensible() {
        DispatchConfig c = load(null);
        assertFalse(c.isEnabled());
        assertEquals("count:5", c.getSubBatch());
        assertEquals("", c.getFePoolServiceId());
        assertEquals(5000, c.getBatchTimeoutMs());
        assertEquals(200, c.getFeMaxConnectionsPerHost());
        assertEquals(1000, c.getFeMaxPendingAcquirePerHost());
    }

    @Test
    void parsesFullJson() {
        DispatchConfig c = load("{\"enabled\":true,\"subBatch\":\"size:10\","
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
        assertFalse(load(null).isEnabled());
        assertFalse(load("  ").isEnabled());
    }

    @Test
    void rejectsEnabledWithoutFePoolServiceId() {
        assertThrows(IllegalArgumentException.class,
                () -> load("{\"enabled\":true}"));
    }

    @Test
    void rejectsEnabledWithBlankFePoolServiceId() {
        assertThrows(IllegalArgumentException.class,
                () -> load("{\"enabled\":true,\"fePoolServiceId\":\"  \"}"));
    }

    @Test
    void rejectsInvalidSubBatchDsl() {
        // subBatch is parsed eagerly during validate() so a malformed DSL fails fast at boot
        assertThrows(IllegalArgumentException.class,
                () -> load("{\"enabled\":true,\"fePoolServiceId\":\"x\",\"subBatch\":\"foo:5\"}"));
        assertThrows(IllegalArgumentException.class,
                () -> load("{\"enabled\":true,\"fePoolServiceId\":\"x\",\"subBatch\":\"count:0\"}"));
    }

    @Test
    void rejectsMalformedJson() {
        assertThrows(FlexLBException.class, () -> load("{not json}"));
    }

    @Test
    void unknownJsonFieldsIgnoredSoOldConfigsDoNotBreak() {
        // Old field names that have been deleted (subBatchSize, feRequestTimeoutMs,
        // feConnectTimeoutMs, feResponseTimeoutMs, feMaxStreamDurationMs, feMaxResponseBytes)
        // must not crash — operators with stale DISPATCH_CONFIG should keep starting up.
        DispatchConfig c = load("{\"enabled\":true,\"fePoolServiceId\":\"x\","
                + "\"subBatchSize\":7,\"feRequestTimeoutMs\":4000,"
                + "\"feConnectTimeoutMs\":1234,\"feResponseTimeoutMs\":6000,"
                + "\"feMaxStreamDurationMs\":120000,\"feMaxResponseBytes\":2097152}");
        assertTrue(c.isEnabled());
        assertEquals("count:5", c.getSubBatch(), "subBatchSize is now unknown — default kept");
        assertEquals(5000, c.getBatchTimeoutMs(), "feResponseTimeoutMs is unknown — default kept");
    }

    @Test
    void envOverridesJsonForEachField() {
        Map<String, String> env = mutableEnv(
                "DISPATCH_CONFIG", "{\"enabled\":true,\"fePoolServiceId\":\"x\",\"batchTimeoutMs\":5000}",
                "DISPATCH_BATCH_TIMEOUT_MS", "8000",
                "DISPATCH_FE_MAX_CONNECTIONS_PER_HOST", "300",
                "DISPATCH_FE_MAX_PENDING_ACQUIRE_PER_HOST", "1500");
        DispatchConfig c = DispatcherConfiguration.loadAndValidate(env);
        assertEquals(8000, c.getBatchTimeoutMs(), "env wins over JSON");
        assertEquals(300, c.getFeMaxConnectionsPerHost());
        assertEquals(1500, c.getFeMaxPendingAcquirePerHost());
    }

    @Test
    void envOverridesDefaultsWhenNoJson() {
        Map<String, String> env = mutableEnv(
                "DISPATCH_ENABLED", "true",
                "DISPATCH_FE_POOL_SERVICE_ID", "from.env",
                "DISPATCH_SUB_BATCH", "count:10");
        DispatchConfig c = DispatcherConfiguration.loadAndValidate(env);
        assertTrue(c.isEnabled());
        assertEquals("from.env", c.getFePoolServiceId());
        assertEquals("count:10", c.getSubBatch());
    }

    @Test
    void subBatchSpecAccessorParsesEagerly() {
        DispatchConfig c = load("{\"enabled\":true,\"fePoolServiceId\":\"x\",\"subBatch\":\"count:7\"}");
        SubBatchSpec spec = c.subBatchSpec();
        assertEquals(SubBatchSpec.Mode.COUNT, spec.mode());
        assertEquals(7, spec.value());
    }

    @Test
    void probePathDefaultsToFrontendHealth() {
        DispatchConfig c = load(null);
        assertEquals("/frontend_health", c.getProbePath(),
                "default targets rtp_llm FE; vLLM users override via DISPATCH_PROBE_PATH");
    }

    @Test
    void probePathFromJson() {
        DispatchConfig c = load("{\"enabled\":true,\"fePoolServiceId\":\"x\",\"probePath\":\"/health\"}");
        assertEquals("/health", c.getProbePath());
    }

    @Test
    void envOverridesProbePath() {
        Map<String, String> env = mutableEnv(
                "DISPATCH_CONFIG",
                "{\"enabled\":true,\"fePoolServiceId\":\"x\",\"probePath\":\"/frontend_health\"}",
                "DISPATCH_PROBE_PATH", "/health");
        DispatchConfig c = DispatcherConfiguration.loadAndValidate(env);
        assertEquals("/health", c.getProbePath(), "DISPATCH_PROBE_PATH must beat the JSON value");
    }

    @Test
    void preAssignBeDefaultsTrue() {
        DispatchConfig c = load(null);
        assertTrue(c.isPreAssignBe(),
                "preAssignBe defaults to true: stamping uses generate_config.role_addrs which "
                        + "FE already supports natively (backend_rpc_server_visitor.route_ips), "
                        + "so there's no FE-side prerequisite to gate this on");
    }

    @Test
    void preAssignBeJsonExplicitFalseDisables() {
        DispatchConfig c = load("{\"enabled\":true,\"fePoolServiceId\":\"x\",\"preAssignBe\":false}");
        assertFalse(c.isPreAssignBe(),
                "operator opt-out via JSON must be honored — explicit false beats default true");
    }

    @Test
    void envOverridesPreAssignBeOff() {
        Map<String, String> env = mutableEnv(
                "DISPATCH_CONFIG", "{\"enabled\":true,\"fePoolServiceId\":\"x\"}",
                "DISPATCH_PRE_ASSIGN_BE", "false");
        DispatchConfig c = DispatcherConfiguration.loadAndValidate(env);
        assertFalse(c.isPreAssignBe(),
                "DISPATCH_PRE_ASSIGN_BE=false must turn the optimization off without code change");
    }

    @Test
    void blankProbePathFailsValidation() {
        Map<String, String> env = mutableEnv("DISPATCH_PROBE_PATH", "");
        // Blank env is treated as "not set" by EnvConfigOverrides, so default sticks here.
        DispatchConfig c = DispatcherConfiguration.loadAndValidate(env);
        assertEquals("/frontend_health", c.getProbePath(),
                "blank env value is ignored; default kept (matches existing FLEXLB_CONFIG semantics)");

        // But an explicit blank in JSON is a config error — must throw to surface the typo.
        assertThrows(IllegalArgumentException.class, () -> load("{\"probePath\":\"  \"}"));
    }

    /** Test seam: load with the given JSON as DISPATCH_CONFIG, no other env overrides. */
    private static DispatchConfig load(String json) {
        Map<String, String> env = new HashMap<>();
        if (json != null) {
            env.put("DISPATCH_CONFIG", json);
        }
        return DispatcherConfiguration.loadAndValidate(env);
    }

    /** Mutable env map (Map.of is immutable; tests need to add multiple entries flexibly). */
    private static Map<String, String> mutableEnv(String... keysAndValues) {
        Map<String, String> map = new HashMap<>();
        for (int i = 0; i + 1 < keysAndValues.length; i += 2) {
            map.put(keysAndValues[i], keysAndValues[i + 1]);
        }
        return map;
    }
}
