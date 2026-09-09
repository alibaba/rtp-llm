package org.flexlb.config;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ConfigServiceTest {
    @Test
    void grpcExecutorDefaultsAndOverridesAreValidated() {
        FlexlbConfig.GrpcServerConfig defaults = ConfigTestFixtures.parse("{}").getGrpcServer();
        assertEquals(1000, defaults.getExecutorCoreSize());
        assertEquals(1000, defaults.getExecutorMaxSize());
        assertEquals(1000, defaults.getExecutorQueueSize());
        FlexlbConfig.GrpcServerConfig grpc = ConfigTestFixtures.parse("""
                {"grpcServer":{"executorCoreSize":16,"executorMaxSize":32,"executorQueueSize":256}}
                """).getGrpcServer();
        assertEquals(16, grpc.getExecutorCoreSize());
        assertEquals(32, grpc.getExecutorMaxSize());
        assertEquals(256, grpc.getExecutorQueueSize());
        for (String patch : new String[]{
                "{\"grpcServer\":{\"executorCoreSize\":-1}}",
                "{\"grpcServer\":{\"executorMaxSize\":0}}",
                "{\"grpcServer\":{\"executorCoreSize\":32,\"executorMaxSize\":16}}",
                "{\"grpcServer\":{\"executorQueueSize\":0}}",
                "{\"grpcServer\":{\"executorQueueSize\":-1}}",
                "{\"grpcServer\":{\"executorCoreSize\":\"16\"}}"}) {
            assertThrows(ConfigValidationException.class, () -> ConfigTestFixtures.parse(patch), patch);
        }
    }

    @Test
    void requires_explicit_request_timeout_and_defaults_decision_lifetime() {
        assertThrows(ConfigValidationException.class, () -> new ConfigService((String) null));
        assertThrows(ConfigValidationException.class,
                () -> new ConfigService("   "));
        for (String patch : new String[]{
                "{}",
                "{\"requestLifecycle\":{\"decision\":{\"lifetime\":1}}}"}) {
            assertThrows(ConfigValidationException.class, () -> ConfigService.parse(patch));
        }
        FlexlbConfig config = new ConfigService(ConfigTestFixtures.REQUIRED).loadBalanceConfig();
        assertEquals(60000L, config.getRequestLifecycle().getRequest().getTimeoutMs());
        assertEquals(2.0, config.getRequestLifecycle().getDecision().getLifetime());
        FlexlbConfig defaults = ConfigService.parse("{\"requestLifecycle\":{\"request\":{\"timeoutMs\":1}}}");
        assertEquals(2.0, defaults.getRequestLifecycle().getDecision().getLifetime());
    }

    @Test
    void retained_defaults_have_one_owner() {
        FlexlbConfig config = ConfigTestFixtures.parse("{}");
        assertEquals(3, config.getSchemaVersion());
        assertTrue(config.isQueue());
        assertFalse(config.isPriorityOrdering());
        assertTrue(config.isFixedWindowDecision());
        assertEquals(3600000L, config.queueScheduler().getQueueTimeoutMs());
        assertEquals(8, config.fixedWindowDecision().getMaxRequests());
        assertEquals(300L, config.fixedWindowDecision().getMaxCollectionWaitMs());
        assertNull(config.fixedWindowDecision().getMaxPredictedExecutionMs());
        assertEquals(2, config.getDispatcher().getMaxInflightPerPrefillWorker());
        var prefill = config.getRouter().getRoles().getPrefill();
        assertEquals(RoutingConfig.EstimatorType.FORMULA, prefill.getExecutionTimeEstimator().getType());
        assertEquals("sum(computeTokens) + 0.3*sum(hitCacheTokens)",
                prefill.getExecutionTimeEstimator().getExpression());
        assertNull(prefill.getCacheAffinity());
        assertNull(config.getRouter().getGroupSelector());
        var decode = config.getRouter().getRoles().getDecode().getAvailability();
        assertEquals(90, decode.getMaxKvUsagePercent());
        assertNull(decode.getMaxEngineRequests());
        var health = config.getWorkerRegistry().getHealth();
        assertEquals(20L, health.getStatusPollIntervalMs());
        assertEquals(5000L, health.getStatusRpcTimeoutMs());
        assertEquals(10000L, health.getStatusStaleAfterMs());
        assertEquals(3000L, health.getCleanupIntervalMs());
        var cache = config.getWorkerRegistry().getCacheStatus();
        assertEquals(30, cache.getTargetDiffSize());
        assertEquals(50L, cache.getMinRefreshIntervalMs());
        assertEquals(3000L, cache.getMaxRefreshIntervalMs());
        assertFalse(cache.isFullSnapshotDebugMode());
        var observability = config.getObservability().getCacheHit();
        assertTrue(observability.getRecentKeyWindow().isWriteEnabled());
        assertEquals(1800000L, observability.getRecentKeyWindow().getDurationMs());
        assertEquals(10000000L, observability.getRecentKeyWindow().getMaxKeyOccurrences());
        assertTrue(observability.isMetricsEnabled());
        assertFalse(observability.isRequestTraceLogEnabled());
        assertNull(observability.getTheoryLog());
    }

    @Test
    void readme_flexlb_config_examples_parse_strictly() throws Exception {
        Path readme = Path.of("README.md");
        if (!Files.exists(readme)) {
            readme = Path.of("..", "README.md");
        }
        String content = Files.readString(readme);
        Matcher examples = Pattern.compile(
                "export FLEXLB_CONFIG='(\\{.*?})'", Pattern.DOTALL)
                .matcher(content);
        int parsed = 0;
        while (examples.find()) {
            ConfigService.parse(examples.group(1));
            parsed++;
        }
        assertEquals(2, parsed, "README FLEXLB_CONFIG example count changed");
    }

    @Test
    void removed_legacy_environment_fails_fast_with_migration_guidance() {
        ConfigValidationException failure = assertThrows(
                ConfigValidationException.class,
                () -> new ConfigService(Map.of(
                        ConfigService.FLEXLB_CONFIG_ENV, ConfigTestFixtures.REQUIRED,
                        "CACHE_STATUS_MAX_INTERVAL_MS", "100",
                        "DEFAULT_SCHEDULE_MODE", "QUEUE",
                        "FLEXLB_MONITOR_MODE", "all",
                        "FLEXLB_MONITOR_METRIC_WHITELIST", "flexlb_")));

        assertTrue(failure.getMessage().contains("CACHE_STATUS_MAX_INTERVAL_MS"));
        assertTrue(failure.getMessage().contains("DEFAULT_SCHEDULE_MODE"));
        assertTrue(failure.getMessage().contains("FLEXLB_MONITOR_MODE"));
        assertTrue(failure.getMessage().contains("FLEXLB_MONITOR_METRIC_WHITELIST"));
        assertTrue(failure.getMessage().contains("schemaVersion 3"));
        assertTrue(failure.getMessage().contains("--flexlb.monitor.metric-whitelist"));
    }

    @Test
    void retained_environment_controls_are_not_treated_as_legacy() {
        FlexlbConfig config = new ConfigService(Map.of(
                ConfigService.FLEXLB_CONFIG_ENV, ConfigTestFixtures.REQUIRED,
                "LOG_LEVEL", "DEBUG",
                "HIPPO_ROLE", "flexlb-test"))
                .loadBalanceConfig();

        assertEquals(FlexlbConfig.CURRENT_SCHEMA_VERSION, config.getSchemaVersion());
        assertEquals(60000L, config.getRequestLifecycle().getRequest().getTimeoutMs());
        assertThrows(ConfigValidationException.class, () -> new ConfigService(Map.of()));
        assertThrows(ConfigValidationException.class,
                () -> new ConfigService(Map.of(ConfigService.FLEXLB_CONFIG_ENV, "   ")));
    }

    @Test
    void rejects_inactive_estimator_variant_fields() {
        assertInvalid("""
                {"router":{"roles":{"prefill":{"executionTimeEstimator":{
                  "type":"LEARNING","expression":"sum(computeTokens)"
                }}}}}
                """, "active mode");
    }

    @Test
    void accepts_retained_policies_and_operational_controls() {
        FlexlbConfig config = ConfigTestFixtures.parse("""
                {
                  "scheduler":{"ordering":{"type":"PRIORITY","defaultPriority":60,
                    "preemption":{"allowedVictimStages":["DECODE_RESERVED","DECODE_ENGINE_OWNED"]}},
                    "decision":{"maxRequests":12,"maxCollectionWaitMs":40,"maxPredictedExecutionMs":90}},
                  "dispatcher":{"maxInflightPerPrefillWorker":4},
                  "router":{
                    "groupSelector":{"defaultTargets":[{"group":"blue"}],"rules":[{
                      "name":"long-context","match":{"inputTokens":{"min":4096}},"targets":[{"group":"long"}]}]},
                    "roles":{"prefill":{"executionTimeEstimator":{"expression":"sum(computeTokens)"},
                      "cacheAffinity":{"maxExtraTtftMs":25,"minPrefixHitPercent":10}},
                      "decode":{"availability":{"maxKvUsagePercent":85,"maxEngineRequests":128}}}},
                  "workerRegistry":{"health":{"statusPollIntervalMs":25},
                    "cacheStatus":{"minRefreshIntervalMs":100,"maxRefreshIntervalMs":2000}},
                  "observability":{"cacheHit":{"recentKeyWindow":{"durationMs":60000,"maxKeyOccurrences":100000},
                    "requestTraceLogEnabled":true,"theoryLog":{}}}
                }
                """);
        assertTrue(config.isPriorityOrdering());
        assertEquals(60, config.priorityOrdering().getDefaultPriority());
        assertTrue(config.priorityOrdering().getPreemption().allows(VictimStage.DECODE_ENGINE_OWNED));
        assertEquals(12, config.fixedWindowDecision().getMaxRequests());
        assertEquals(90L, config.fixedWindowDecision().getMaxPredictedExecutionMs());
        assertEquals(4, config.getDispatcher().getMaxInflightPerPrefillWorker());
        assertEquals(25L, config.getRouter().getRoles().getPrefill().getCacheAffinity().getMaxExtraTtftMs());
        assertEquals(128L, config.getRouter().getRoles().getDecode().getAvailability().getMaxEngineRequests());
        assertEquals(1, config.getRouter().getGroupSelector().getRules().size());
        assertEquals("/home/admin/ai-whale/logs/master_theory_hit.log",
                config.getObservability().getCacheHit().getTheoryLog().getPath());
    }

    @Test
    void engine_owned_preemption_timeout_has_one_default_and_accepts_positive_overrides() {
        assertEquals(1000L, new PreemptionConfig().getTimeoutMs());
        FlexlbConfig config = ConfigTestFixtures.parse("""
                {"scheduler":{"ordering":{"type":"PRIORITY","preemption":{
                  "allowedVictimStages":["DECODE_ENGINE_OWNED"]}}}}
                """);
        assertEquals(1000L, config.priorityOrdering().getPreemption().getTimeoutMs());
        for (long timeoutMs : new long[]{1L, 2500L, Long.MAX_VALUE}) {
            config = ConfigTestFixtures.parse("""
                    {"scheduler":{"ordering":{"type":"PRIORITY","preemption":{
                      "allowedVictimStages":["DECODE_RESERVED","DECODE_ENGINE_OWNED"],
                      "timeoutMs":%d}}}}
                    """.formatted(timeoutMs));
            assertEquals(timeoutMs, config.priorityOrdering().getPreemption().getTimeoutMs());
        }
    }

    @Test
    void priority_defaults_to_all_preemption_stages_and_preserves_explicit_subsets() {
        var allStages = java.util.EnumSet.allOf(VictimStage.class);
        assertEquals(allStages, new PreemptionConfig().getAllowedVictimStages());
        assertEquals(allStages, QueueOrderingConfig.priority().getPreemption().getAllowedVictimStages());
        for (String ordering : new String[]{
                "{\"type\":\"PRIORITY\"}",
                "{\"type\":\"PRIORITY\",\"preemption\":{}}",
                "{\"type\":\"PRIORITY\",\"preemption\":{\"timeoutMs\":2500}}"}) {
            FlexlbConfig config = ConfigTestFixtures.parse("{\"scheduler\":{\"ordering\":" + ordering + "}}");
            assertEquals(allStages, config.priorityOrdering().getPreemption().getAllowedVictimStages());
            for (VictimStage stage : allStages) { assertTrue(config.allowsPreemption(stage)); }
        }
        for (String ordering : new String[]{
                "{\"type\":\"PRIORITY\",\"preemption\":{\"allowedVictimStages\":[\"PREFILL_QUEUED\"]}}",
                "{\"preemption\":{\"allowedVictimStages\":[\"PREFILL_QUEUED\"]},\"type\":\"PRIORITY\"}"}) {
            FlexlbConfig config = ConfigTestFixtures.parse("{\"scheduler\":{\"ordering\":" + ordering + "}}");
            assertEquals(java.util.Set.of(VictimStage.PREFILL_QUEUED),
                    config.priorityOrdering().getPreemption().getAllowedVictimStages());
            assertFalse(config.allowsPreemption(VictimStage.DECODE_ENGINE_OWNED));
        }
        FlexlbConfig fifo = ConfigTestFixtures.parse("{}");
        assertTrue(fifo.queueScheduler().getOrdering().preemptionPolicy().isEmpty());
        for (VictimStage stage : allStages) { assertFalse(fifo.allowsPreemption(stage)); }
        assertInvalid("""
                {"scheduler":{"ordering":{"type":"PRIORITY","preemption":{"allowedVictimStages":[]}}}}
                """, "allowedVictimStages");
    }

    @Test
    void preemption_timeout_rejects_non_positive_values_and_json_coercion() {
        for (String value : new String[]{"0", "-1", "1.5", "\"1000\"", "true", "null", "[]", "{}",
                "9223372036854775808"}) {
            assertInvalid("""
                    {"scheduler":{"ordering":{"type":"PRIORITY","preemption":{
                      "allowedVictimStages":["DECODE_ENGINE_OWNED"],"timeoutMs":%s}}}}
                    """.formatted(value), "timeoutMs");
        }
        FlexlbConfig config = ConfigTestFixtures.parse("""
                {"scheduler":{"ordering":{"type":"PRIORITY","preemption":{
                  "allowedVictimStages":["DECODE_ENGINE_OWNED"]}}}}
                """);
        config.priorityOrdering().getPreemption().setTimeoutMs(0);
        assertThrows(ConfigValidationException.class, () -> FlexlbConfigValidator.validate(config));
    }

    @Test
    void explicit_preemption_timeout_requires_engine_owned_decode_stage() {
        FlexlbConfig config = ConfigTestFixtures.parse("""
                {"scheduler":{"ordering":{"type":"PRIORITY","preemption":{
                  "allowedVictimStages":["DECODE_RESERVED"]}}}}
                """);
        assertTrue(config.priorityOrdering().getPreemption().allows(VictimStage.DECODE_RESERVED));
        assertInvalid("""
                {"scheduler":{"ordering":{"type":"PRIORITY","preemption":{
                  "allowedVictimStages":["DECODE_RESERVED"],"timeoutMs":1000}}}}
                """, "timeoutMs");
        assertInvalid("""
                {"scheduler":{"ordering":{"type":"FIFO","preemption":{
                  "allowedVictimStages":["DECODE_ENGINE_OWNED"],"timeoutMs":1000}}}}
                """, "preemption");
    }

    @Test
    void queued_prefill_preemption_is_valid_alone_or_with_decode_preemption() {
        FlexlbConfig config = ConfigTestFixtures.parse("""
                {"scheduler":{"ordering":{"type":"PRIORITY","preemption":{
                  "allowedVictimStages":["PREFILL_QUEUED"]}}}}
                """);
        assertTrue(config.allowsPreemption(VictimStage.PREFILL_QUEUED));
        config = ConfigTestFixtures.parse("""
                {"scheduler":{"ordering":{"type":"PRIORITY","preemption":{
                  "allowedVictimStages":["DECODE_RESERVED","PREFILL_QUEUED"]}}}}
                """);
        assertTrue(config.allowsPreemption(VictimStage.PREFILL_QUEUED));
        assertTrue(config.allowsPreemption(VictimStage.DECODE_RESERVED));
        assertInvalid("""
                {"scheduler":{"ordering":{"type":"PRIORITY","preemption":{
                  "allowedVictimStages":["PREFILL_QUEUED"],"timeoutMs":1000}}}}
                """, "timeoutMs");
    }

    @Test
    void removed_control_fields_fail_instead_of_becoming_hidden_settings() throws Exception {
        Map<String, String> removed = Map.ofEntries(
                Map.entry("autoTpmEnabled", "true"),
                Map.entry("autoTpmSloLengthBuckets", "\"*:100\""),
                Map.entry("router.availabilityHysteresisPercent", "15"),
                Map.entry("router.roles.prefill.availability.maxPendingRequests", "32"),
                Map.entry("router.roles.prefill.availability.maxUncachedTokens", "65536"),
                Map.entry("router.roles.prefill.availability.maxPredictedQueueWaitMs", "1000"),
                Map.entry("scheduler.capacity.maxOutstandingRequestsGlobal", "100000"),
                Map.entry("scheduler.capacity.maxWaitingRequestsPerPrefillWorker", "1024"),
                Map.entry("scheduler.lifecycle.staleInflightTimeoutMs", "300000"),
                Map.entry("scheduler.lifecycle.deliveredNotAcceptedTimeoutMs", "30000"),
                Map.entry("scheduler.lifecycle.maxDeliveredNotAcceptedRequestsGlobal", "200"),
                Map.entry("scheduler.ordering.preemption.engineCancellation.ackTimeoutMs", "50"),
                Map.entry("scheduler.ordering.preemption.engineCancellation.completionTimeoutMs", "1000"),
                Map.entry("dispatcher.maxInflightRequestsPerPrefillWorker", "8"),
                Map.entry("dispatcher.enqueueRpcTimeoutMs", "5000"),
                Map.entry("router.roles.prefill.candidateChoice.type", "\"BEST_ONLY\""),
                Map.entry("router.roles.prefill.candidateChoice.relativeTolerance", "0.1"),
                Map.entry("router.roles.prefill.candidateChoice.minimumToleranceMs", "20"),
                Map.entry("router.roles.prefill.candidateChoice.pool.type", "\"RATIO\""),
                Map.entry("router.roles.prefill.candidateChoice.pool.ratio", "0.3"),
                Map.entry("router.roles.prefill.candidateChoice.pool.minimumWorkers", "1"),
                Map.entry("router.roles.prefill.candidateChoice.pool.workers", "1"),
                Map.entry("router.roles.prefill.candidateChoice.outlierRejection.maxPendingVsAverageMultiplier", "3"),
                Map.entry("router.roles.prefill.candidateChoice.outlierRejection.maxProjectedDrainVsAverageMultiplier", "3"),
                Map.entry("router.roles.decode.kvReservation.maxOutputTokensForEstimate", "1000"),
                Map.entry("router.roles.decode.decayPerToken", "0.001"),
                Map.entry("router.roles.decode.loadDecayPerRequest", "1"),
                Map.entry("router.roles.decode.outlierRejection.maxEngineLoadVsAverageMultiplier", "3"),
                Map.entry("router.roles.decode.outlierRejection.maxKvUsedVsAverageMultiplier", "3"));
        ObjectMapper json = new ObjectMapper();
        for (var entry : removed.entrySet()) {
            ObjectNode root = (ObjectNode) json.readTree(ConfigTestFixtures.document(
                    "{\"scheduler\":{\"ordering\":{\"type\":\"PRIORITY\",\"preemption\":{\"allowedVictimStages\":[\"DECODE_ENGINE_OWNED\"]}}}}"));
            String[] names = entry.getKey().split("\\.");
            ObjectNode node = root;
            for (int i = 0; i < names.length - 1; i++) {
                node = node.with(names[i]);
            }
            node.set(names[names.length - 1], json.readTree(entry.getValue()));
            assertThrows(ConfigValidationException.class, () -> ConfigService.parse(root.toString()), entry.getKey());
        }
    }

    @Test
    void rejects_json_ambiguity_and_type_coercion() {
        for (String document : new String[]{
                ConfigTestFixtures.REQUIRED.replace("\"schemaVersion\": 3", "\"schemaVersion\": 3, \"schemaVersion\": 3"),
                ConfigTestFixtures.REQUIRED + " {}",
                ConfigTestFixtures.document("{\"router\":null}"),
                ConfigTestFixtures.document("{\"schemaVersion\":\"3\"}"),
                ConfigTestFixtures.document("{\"schemaVersion\":3.5}"),
                ConfigTestFixtures.document("{\"internalRuntime\":{}}"),
                ConfigTestFixtures.document("{\"requestLifecycle\":{\"request\":{\"timeoutMs\":\"60000\"}}}"),
                ConfigTestFixtures.document("{\"requestLifecycle\":{\"decision\":{\"lifetime\":\"2\"}}}"),
                ConfigTestFixtures.document("{\"scheduler\":{\"type\":0}}")}) {
            assertThrows(ConfigValidationException.class, () -> ConfigService.parse(document), document);
        }
        for (int version : new int[]{1, 2, 4}) {
            assertInvalid("{\"schemaVersion\":" + version + "}", "schemaVersion");
        }
    }

    @Test
    void validates_required_values_and_retained_bounds() {
        for (String patch : new String[]{
                "{\"requestLifecycle\":{\"request\":{\"timeoutMs\":0}}}",
                "{\"requestLifecycle\":{\"request\":{\"timeoutMs\":-1}}}",
                "{\"requestLifecycle\":{\"decision\":{\"lifetime\":0.99}}}",
                "{\"router\":{\"roles\":{\"decode\":{\"availability\":{\"maxEngineRequests\":0}}}}}",
                "{\"router\":{\"roles\":{\"decode\":{\"availability\":{\"maxKvUsagePercent\":101}}}}}",
                "{\"workerRegistry\":{\"health\":{\"statusStaleAfterMs\":9999}}}",
                "{\"workerRegistry\":{\"health\":{\"cleanupIntervalMs\":0}}}",
                "{\"workerRegistry\":{\"health\":{\"cleanupIntervalMs\":-1}}}",
                "{\"workerRegistry\":{\"cacheStatus\":{\"minRefreshIntervalMs\":100,\"maxRefreshIntervalMs\":99}}}"}) {
            assertThrows(ConfigValidationException.class, () -> ConfigTestFixtures.parse(patch), patch);
        }
        assertInvalid("{\"router\":{\"roles\":{\"prefill\":{\"executionTimeEstimator\":{\"expression\":\"sum(unknownTokens)\"}}}}}", "expression");
        assertInvalid("{\"router\":{\"roles\":{\"prefill\":{\"cacheAffinity\":{\"maxExtraTtftMs\":-1}}}}}", "maxExtraTtftMs");
        assertInvalid("{\"router\":{\"roles\":{\"prefill\":{\"cacheAffinity\":{\"minPrefixHitPercent\":100.1}}}}}", "minPrefixHitPercent");
        FlexlbConfig config = ConfigTestFixtures.parse("{}");
        for (double invalid : new double[]{Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY}) {
            config.getRequestLifecycle().getDecision().setLifetime(invalid);
            assertThrows(ConfigValidationException.class, () -> FlexlbConfigValidator.validate(config));
        }
    }

    @Test
    void shipped_examples_parse_using_the_runtime_contract() throws Exception {
        for (String name : new String[]{"flexlb-queue-priority-batch.json", "flexlb-queue-priority-non-batch.json"}) {
            String document = java.nio.file.Files.readString(java.nio.file.Path.of("../docs/config-examples", name));
            assertEquals(3, ConfigService.parse(document).getSchemaVersion());
        }
    }

    private static void assertInvalid(String patch, String field) {
        var error = assertThrows(ConfigValidationException.class, () -> ConfigTestFixtures.parse(patch));
        assertTrue(error.getMessage().contains(field), error.getMessage());
    }
}
