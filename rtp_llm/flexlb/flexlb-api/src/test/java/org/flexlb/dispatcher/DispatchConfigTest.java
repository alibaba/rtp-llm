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
 *
 * <p>Note these tests bypass Spring's {@code @ConditionalOnProperty} entirely — they call
 * {@code loadAndValidate} directly with a synthetic env map, so they exercise the validation
 * branch as if the dispatcher bean had already been activated. The "what activates the bean"
 * question is owned by Spring's {@code OnPropertyCondition} and is verified separately by
 * the integration-style smoke checks (start/no-start with/without the env).
 */
class DispatchConfigTest {

    @Test
    void defaultsAreSensible() {
        // load() with a non-blank fePoolServiceId so validate() doesn't throw — the defaults we
        // care about asserting here are the other fields, not the enable signal itself.
        DispatchConfig c = load("{\"fePoolServiceId\":\"x\"}");
        assertEquals("count:5", c.getSubBatch());
        assertEquals(30_000, c.getBatchTimeoutMs(),
                "non-streaming generation endpoints return headers only after the full "
                        + "generation completes — 5s killed legitimate large chunks");
    }

    @Test
    void parsesFullJson() {
        DispatchConfig c = load("{\"subBatch\":\"size:10\","
                + "\"fePoolServiceId\":\"com.rtp_llm.fe\","
                + "\"batchTimeoutMs\":7500}");
        assertEquals("size:10", c.getSubBatch());
        assertEquals("com.rtp_llm.fe", c.getFePoolServiceId());
        assertEquals(7500, c.getBatchTimeoutMs());
    }

    @Test
    void rejectsBlankFePoolServiceId() {
        // No env at all → defaults give blank fePoolServiceId → must throw. This branch only
        // fires in practice if Spring activated the bean on a whitespace-only env value, but
        // the test exercises it directly so the precise error message stays asserted.
        assertThrows(IllegalArgumentException.class, () -> load(null));
        assertThrows(IllegalArgumentException.class, () -> load("{\"fePoolServiceId\":\"  \"}"));
        assertThrows(IllegalArgumentException.class, () -> load("{}"));
    }

    @Test
    void rejectsInvalidSubBatchDsl() {
        // subBatch is parsed eagerly during validate() so a malformed DSL fails fast at boot
        assertThrows(IllegalArgumentException.class,
                () -> load("{\"fePoolServiceId\":\"x\",\"subBatch\":\"foo:5\"}"));
        assertThrows(IllegalArgumentException.class,
                () -> load("{\"fePoolServiceId\":\"x\",\"subBatch\":\"count:0\"}"));
    }

    @Test
    void rejectsMalformedJson() {
        assertThrows(FlexLBException.class, () -> load("{not json}"));
    }

    @Test
    void unknownJsonFieldsIgnoredSoOldConfigsDoNotBreak() {
        // Old field names that have been deleted (subBatchSize, feRequestTimeoutMs,
        // feConnectTimeoutMs, feResponseTimeoutMs, feMaxStreamDurationMs, feMaxResponseBytes)
        // must not crash — keeps the POJO contract stable for any future field that gets retired.
        DispatchConfig c = load("{\"fePoolServiceId\":\"x\","
                + "\"subBatchSize\":7,\"feRequestTimeoutMs\":4000,"
                + "\"feConnectTimeoutMs\":1234,\"feResponseTimeoutMs\":6000,"
                + "\"feMaxStreamDurationMs\":120000,\"feMaxResponseBytes\":2097152}");
        assertEquals("count:5", c.getSubBatch(), "subBatchSize is now unknown — default kept");
        assertEquals(30_000, c.getBatchTimeoutMs(), "feResponseTimeoutMs is unknown — default kept");
        assertEquals("x", c.getFePoolServiceId());
    }

    @Test
    void envOverridesJsonForEachField() {
        Map<String, String> env = mutableEnv(
                "DISPATCH_CONFIG", "{\"fePoolServiceId\":\"x\",\"batchTimeoutMs\":5000}",
                "DISPATCH_BATCH_TIMEOUT_MS", "8000");
        DispatchConfig c = DispatcherConfiguration.loadAndValidate(env);
        assertEquals(8000, c.getBatchTimeoutMs(), "env wins over JSON");
    }

    @Test
    void bodyReadMarginMsFlowsFromJsonAndEnv() {
        DispatchConfig fromJson = load("{\"fePoolServiceId\":\"x\",\"bodyReadMarginMs\":12000}");
        assertEquals(12000L, fromJson.getBodyReadMarginMs(), "JSON value must reach the field");

        Map<String, String> env = mutableEnv(
                "DISPATCH_CONFIG", "{\"fePoolServiceId\":\"x\",\"bodyReadMarginMs\":12000}",
                "DISPATCH_BODY_READ_MARGIN_MS", "45000");
        DispatchConfig c = DispatcherConfiguration.loadAndValidate(env);
        assertEquals(45000L, c.getBodyReadMarginMs(), "DISPATCH_BODY_READ_MARGIN_MS env must win over JSON");
    }

    @Test
    void bodyReadMarginMsDefault() {
        DispatchConfig c = load("{\"fePoolServiceId\":\"x\"}");
        assertEquals(30000L, c.getBodyReadMarginMs(), "default whole-call body margin is 30s");
    }

    @Test
    void envOverridesDefaultsWhenNoJson() {
        Map<String, String> env = mutableEnv(
                "DISPATCH_FE_POOL_SERVICE_ID", "from.env",
                "DISPATCH_SUB_BATCH", "count:10");
        DispatchConfig c = DispatcherConfiguration.loadAndValidate(env);
        assertEquals("from.env", c.getFePoolServiceId());
        assertEquals("count:10", c.getSubBatch());
    }

    @Test
    void subBatchSpecAccessorParsesEagerly() {
        DispatchConfig c = load("{\"fePoolServiceId\":\"x\",\"subBatch\":\"count:7\"}");
        SubBatchSpec spec = c.getSubBatchSpec();
        assertEquals(SubBatchSpec.Mode.COUNT, spec.mode());
        assertEquals(7, spec.value());
    }

    @Test
    void probePathDefaultsToFrontendHealth() {
        DispatchConfig c = load("{\"fePoolServiceId\":\"x\"}");
        assertEquals("/frontend_health", c.getProbePath(),
                "default targets rtp_llm FE; vLLM users override via DISPATCH_PROBE_PATH");
    }

    @Test
    void probePathFromJson() {
        DispatchConfig c = load("{\"fePoolServiceId\":\"x\",\"probePath\":\"/health\"}");
        assertEquals("/health", c.getProbePath());
    }

    @Test
    void envOverridesProbePath() {
        Map<String, String> env = mutableEnv(
                "DISPATCH_CONFIG",
                "{\"fePoolServiceId\":\"x\",\"probePath\":\"/frontend_health\"}",
                "DISPATCH_PROBE_PATH", "/health");
        DispatchConfig c = DispatcherConfiguration.loadAndValidate(env);
        assertEquals("/health", c.getProbePath(), "DISPATCH_PROBE_PATH must beat the JSON value");
    }

    @Test
    void preAssignBeDefaultsTrue() {
        DispatchConfig c = load("{\"fePoolServiceId\":\"x\"}");
        assertTrue(c.isPreAssignBe(),
                "preAssignBe defaults to true: stamping uses generate_config.role_addrs which "
                        + "FE already supports natively (backend_rpc_server_visitor.route_ips), "
                        + "so there's no FE-side prerequisite to gate this on");
    }

    @Test
    void preAssignBeJsonExplicitFalseDisables() {
        DispatchConfig c = load("{\"fePoolServiceId\":\"x\",\"preAssignBe\":false}");
        assertFalse(c.isPreAssignBe(),
                "operator opt-out via JSON must be honored — explicit false beats default true");
    }

    @Test
    void envOverridesPreAssignBeOff() {
        Map<String, String> env = mutableEnv(
                "DISPATCH_CONFIG", "{\"fePoolServiceId\":\"x\"}",
                "DISPATCH_PRE_ASSIGN_BE", "false");
        DispatchConfig c = DispatcherConfiguration.loadAndValidate(env);
        assertFalse(c.isPreAssignBe(),
                "DISPATCH_PRE_ASSIGN_BE=false must turn the optimization off without code change");
    }

    @Test
    void blankProbePathFailsValidation() {
        // Blank env is treated as "not set" by EnvConfigOverrides, so default sticks here.
        Map<String, String> env = mutableEnv(
                "DISPATCH_FE_POOL_SERVICE_ID", "x",
                "DISPATCH_PROBE_PATH", "");
        DispatchConfig c = DispatcherConfiguration.loadAndValidate(env);
        assertEquals("/frontend_health", c.getProbePath(),
                "blank env value is ignored; default kept (matches existing FLEXLB_CONFIG semantics)");

        // But an explicit blank in JSON is a config error — must throw to surface the typo.
        assertThrows(IllegalArgumentException.class,
                () -> load("{\"fePoolServiceId\":\"x\",\"probePath\":\"  \"}"));
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
