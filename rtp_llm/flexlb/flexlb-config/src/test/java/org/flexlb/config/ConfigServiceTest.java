package org.flexlb.config;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.flexlb.config.loading.ConfigValidationException;
import org.flexlb.config.loading.FlexlbConfigValidator;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Checks configuration loading and updates exposed by the common configuration service.
 */
class ConfigServiceTest {

    @Test
    void prefillCostFormulaDefaultsToTokensAndAcceptsScalarExpressions() {
        assertEquals(
                "computeTokens",
                ConfigTestFixtures.parse("{}")
                        .getRouter()
                        .getRoles()
                        .getPrefill()
                        .getCostExpression());
        var config =
                ConfigTestFixtures.parse(
                        """
                {"schemaVersion":4,"router":{"roles":{"prefill":{"costExpression":"10 + computeTokens * 0.5"}}}}
                """);
        var predictor =
                new org.flexlb.util.formula.PrefillTimePredictor(
                        config.getRouter().getRoles().getPrefill().getCostExpression());
        assertEquals(35.0, predictor.score(50));
        for (String expression :
                new String[] {"", "unknown", "sum(computeTokens)", "computeTokens +"}) {
            var failure =
                    assertThrows(
                            ConfigValidationException.class,
                            () ->
                                    ConfigTestFixtures.parse(
                                            "{\"router\":{\"roles\":{\"prefill\":{\"costExpression\":\""
                                                    + expression
                                                    + "\"}}}}"));
            assertTrue(failure.getMessage().contains("costExpression"));
        }
    }

    @Test
    void localStandbyTtlReductionRatioAllowsEndpoints() {
        for (double ratio : new double[] {0.0, 1.0}) {
            String patch =
                    "{\"cacheMatching\":{\"type\":\"KVCM\",\"localStandby\":{\"ttlReductionStartRatio\":"
                            + ratio
                            + "}}}";
            assertEquals(
                    ratio,
                    ConfigTestFixtures.parse(patch)
                            .kvcmCacheMatching()
                            .getLocalStandby()
                            .getTtlReductionStartRatio());
        }
        for (double ratio : new double[] {-0.01, 1.01}) {
            String patch =
                    "{\"cacheMatching\":{\"type\":\"KVCM\",\"localStandby\":{\"ttlReductionStartRatio\":"
                            + ratio
                            + "}}}";
            assertThrows(ConfigValidationException.class, () -> ConfigTestFixtures.parse(patch));
        }
    }

    @Test
    void receivingCountsDefaultIndependentlyOfExecutorAndAcceptExplicitOverrides() {
        var defaults = ConfigTestFixtures.parse("{}").getGrpcServer();
        assertEquals(1024, defaults.getMaxReceivingRequests());
        assertEquals(128, defaults.getMaxReceivingControlRequests());
        var configured =
                ConfigTestFixtures.parse(
                                """
                {"grpcServer":{"executorCoreSize":64,"executorMaxSize":64,
                 "maxReceivingRequests":4096,"maxReceivingControlRequests":16}}
                """)
                        .getGrpcServer();
        assertEquals(4096, configured.getMaxReceivingRequests());
        assertEquals(16, configured.getMaxReceivingControlRequests());
        for (String field : new String[] {"maxReceivingRequests", "maxReceivingControlRequests"}) {
            assertThrows(
                    ConfigValidationException.class,
                    () -> ConfigTestFixtures.parse("{\"grpcServer\":{\"" + field + "\":-1}}"));
        }
    }

    @Test
    void shutdownQuietPeriodUsesConfigDefaultsOverridesAndValidation() {
        assertEquals(
                5000L, ConfigTestFixtures.parse("{}").getGrpcServer().getShutdownQuietPeriodMs());
        assertEquals(
                7000L,
                ConfigTestFixtures.parse(
                                """
                {"grpcServer":{"shutdownQuietPeriodMs":7000}}
                """)
                        .getGrpcServer()
                        .getShutdownQuietPeriodMs());
        for (String value : new String[] {"0", "-1", "null", "1.5", "\"7000\""}) {
            ConfigValidationException error =
                    assertThrows(
                            ConfigValidationException.class,
                            () ->
                                    ConfigTestFixtures.parse(
                                            "{\"grpcServer\":{\"shutdownQuietPeriodMs\":"
                                                    + value
                                                    + "}}"));
            assertTrue(error.getMessage().contains("shutdownQuietPeriodMs"), error.getMessage());
        }
    }

    @Test
    void grpcExecutorDefaultsAndOverridesAreValidated() {
        FlexlbConfig.GrpcServerConfig defaults = ConfigTestFixtures.parse("{}").getGrpcServer();
        assertEquals(1000, defaults.getExecutorCoreSize());
        assertEquals(1000, defaults.getExecutorMaxSize());
        assertEquals(1000, defaults.getExecutorQueueSize());
        FlexlbConfig.GrpcServerConfig grpc =
                ConfigTestFixtures.parse(
                                """
                {"grpcServer":{"executorCoreSize":16,"executorMaxSize":32,"executorQueueSize":256}}
                """)
                        .getGrpcServer();
        assertEquals(16, grpc.getExecutorCoreSize());
        assertEquals(32, grpc.getExecutorMaxSize());
        assertEquals(256, grpc.getExecutorQueueSize());
        for (String patch :
                new String[] {
                    "{\"grpcServer\":{\"executorCoreSize\":-1}}",
                    "{\"grpcServer\":{\"executorMaxSize\":0}}",
                    "{\"grpcServer\":{\"executorCoreSize\":32,\"executorMaxSize\":16}}",
                    "{\"grpcServer\":{\"executorQueueSize\":0}}",
                    "{\"grpcServer\":{\"executorQueueSize\":-1}}",
                    "{\"grpcServer\":{\"executorCoreSize\":\"16\"}}"
                }) {
            assertThrows(
                    ConfigValidationException.class, () -> ConfigTestFixtures.parse(patch), patch);
        }
    }

    @Test
    void readme_flexlb_config_examples_parse_strictly() throws Exception {
        Path readme = Path.of("README.md");
        if (!Files.exists(readme)) {
            readme = Path.of("..", "README.md");
        }
        String content = Files.readString(readme);
        Matcher examples =
                Pattern.compile("export FLEXLB_CONFIG='(\\{.*?})'", Pattern.DOTALL)
                        .matcher(content);
        int parsed = 0;
        while (examples.find()) {
            ConfigService.parse(examples.group(1));
            parsed++;
        }
        assertEquals(1, parsed, "README FLEXLB_CONFIG example count changed");
    }

    @Test
    void rejects_removed_legacy_environment_variables() {
        ConfigValidationException error =
                assertThrows(
                        ConfigValidationException.class,
                        () ->
                                new ConfigService(
                                        Map.of(
                                                ConfigService.FLEXLB_CONFIG_ENV,
                                                ConfigTestFixtures.REQUIRED,
                                                "CACHE_STATUS_MAX_INTERVAL_MS",
                                                "100",
                                                "DEFAULT_SCHEDULE_MODE",
                                                "QUEUE",
                                                "FLEXLB_MONITOR_MODE",
                                                "all",
                                                "FLEXLB_MONITOR_METRIC_WHITELIST",
                                                "app.flexlb")));
        assertTrue(
                error.getMessage().contains("Removed legacy FlexLB environment variables"),
                error.getMessage());
    }

    @Test
    void retained_environment_controls_are_not_treated_as_legacy() {
        FlexlbConfig config =
                new ConfigService(
                                Map.of(
                                        ConfigService.FLEXLB_CONFIG_ENV,
                                        ConfigTestFixtures.REQUIRED,
                                        "LOG_LEVEL",
                                        "DEBUG",
                                        "HIPPO_ROLE",
                                        "flexlb-test"))
                        .loadBalanceConfig();

        assertEquals(FlexlbConfig.CURRENT_SCHEMA_VERSION, config.getSchemaVersion());
        assertEquals(60000L, config.getRequestLifecycle().getRequest().getTimeoutMs());
        assertThrows(ConfigValidationException.class, () -> new ConfigService(Map.of()));
        assertThrows(
                ConfigValidationException.class,
                () -> new ConfigService(Map.of(ConfigService.FLEXLB_CONFIG_ENV, "   ")));
    }

    @Test
    void decode_cost_estimator_accepts_custom_formulas() {
        for (String expression :
                new String[] {
                    "kvcache_used_ratio",
                    "running_size",
                    "sqrt(running_size) + pow(kvcache_used_ratio, 2)",
                    "0.3 * running_size / max_running_size + 0.7 * kvcache_used_ratio",
                    "-2"
                }) {
            FlexlbConfig config =
                    ConfigTestFixtures.parse(
                            """
                    {"router":{"roles":{"decode":{"costEstimator":{"expression":"%s"},
                      "availability":{"maxEngineRequests":128}}}}}
                    """
                                    .formatted(expression));
            assertEquals(
                    expression,
                    config.getRouter().getRoles().getDecode().getCostEstimator().getExpression());
            assertEquals(
                    90,
                    config.getRouter()
                            .getRoles()
                            .getDecode()
                            .getAvailability()
                            .getMaxKvUsagePercent());
        }
        FlexlbConfig config =
                ConfigTestFixtures.parse(
                        """
                {"router":{"roles":{"decode":{"costEstimator":{"expression":"running_size"}}}}}
                """);
        assertNull(
                config.getRouter().getRoles().getDecode().getAvailability().getMaxEngineRequests());
    }

    @Test
    void decode_max_running_size_variable_requires_positive_max_engine_requests() {
        assertInvalid(
                """
                {"router":{"roles":{"decode":{"costEstimator":{"expression":"running_size / max_running_size"}}}}}
                """,
                "maxEngineRequests");
        for (int maxRequests : new int[] {0, -1}) {
            assertInvalid(
                    """
                    {"router":{"roles":{"decode":{"costEstimator":{"expression":"max_running_size"},
                      "availability":{"maxEngineRequests":%s}}}}}
                    """
                            .formatted(maxRequests),
                    "maxEngineRequests");
        }
        FlexlbConfig config = ConfigTestFixtures.parse("{}");
        assertNull(
                config.getRouter().getRoles().getDecode().getAvailability().getMaxEngineRequests());
        config.getRouter()
                .getRoles()
                .getDecode()
                .getCostEstimator()
                .setExpression("max_running_size");
        var error =
                assertThrows(
                        ConfigValidationException.class,
                        () -> FlexlbConfigValidator.validate(config));
        assertTrue(error.getMessage().contains("maxEngineRequests"), error.getMessage());
    }

    @Test
    void decode_cost_estimator_rejects_invalid_formulas_and_json_coercion() {
        for (String expression :
                new String[] {
                    "", "   ", "running_size +", "unknown_variable", "sum(running_size)", "sqrt()"
                }) {
            assertInvalid(
                    """
                    {"router":{"roles":{"decode":{"costEstimator":{"expression":"%s"}}}}}
                    """
                            .formatted(expression),
                    "expression");
        }
        for (String value : new String[] {"0", "1.5", "true", "null", "[]", "{}"}) {
            assertInvalid(
                    """
                    {"router":{"roles":{"decode":{"costEstimator":{"expression":%s}}}}}
                    """
                            .formatted(value),
                    "expression");
        }
        assertInvalid(
                "{\"router\":{\"roles\":{\"decode\":{\"costEstimator\":null}}}}", "costEstimator");
    }

    @Test
    void decode_cost_estimator_validates_programmatic_values() {
        FlexlbConfig config = ConfigTestFixtures.parse("{}");
        var estimator = config.getRouter().getRoles().getDecode().getCostEstimator();
        for (String invalid :
                new String[] {null, "", "   ", "running_size +", "unknown_variable"}) {
            estimator.setExpression(invalid);
            assertThrows(
                    ConfigValidationException.class, () -> FlexlbConfigValidator.validate(config));
        }
        config.getRouter().getRoles().getDecode().setCostEstimator(null);
        assertThrows(ConfigValidationException.class, () -> FlexlbConfigValidator.validate(config));
    }

    @Test
    void decode_cost_estimator_reuses_compiled_expression_without_exposing_its_cache() {
        FlexlbConfig config = ConfigTestFixtures.parse("{}");
        var estimator = config.getRouter().getRoles().getDecode().getCostEstimator();
        var original = estimator.compiledFormula();
        assertSame(original, estimator.compiledFormula());
        FlexlbConfigValidator.validate(config);
        assertSame(original, estimator.compiledFormula());
        estimator.setExpression("running_size");
        var changed = estimator.compiledFormula();
        assertNotSame(original, changed);
        assertSame(changed, estimator.compiledFormula());
        assertFalse(new ObjectMapper().valueToTree(estimator).has("compiledCost"));
        assertInvalid(
                """
                {"router":{"roles":{"decode":{"costEstimator":{"compiledCost":{}}}}}}
                """,
                "compiledCost");
    }

    @Test
    void preemption_timeout_rejects_non_positive_values_and_json_coercion() {
        for (String value :
                new String[] {
                    "0", "-1", "1.5", "\"1000\"", "true", "null", "[]", "{}", "9223372036854775808"
                }) {
            assertInvalid(
                    """
                    {"scheduler":{"ordering":{"type":"PRIORITY","preemption":{
                      "allowedVictimStages":["DECODE_ENGINE_OWNED"],"timeoutMs":%s}}}}
                    """
                            .formatted(value),
                    "timeoutMs");
        }
        FlexlbConfig config =
                ConfigTestFixtures.parse(
                        """
                {"scheduler":{"ordering":{"type":"PRIORITY","preemption":{
                  "allowedVictimStages":["DECODE_ENGINE_OWNED"]}}}}
                """);
        config.priorityOrdering().getPreemption().setTimeoutMs(0);
        assertThrows(ConfigValidationException.class, () -> FlexlbConfigValidator.validate(config));
    }

    @Test
    void removed_control_fields_fail_instead_of_becoming_hidden_settings() throws Exception {
        Map<String, String> removed =
                Map.ofEntries(
                        Map.entry("autoTpmEnabled", "true"),
                        Map.entry("autoTpmSloLengthBuckets", "\"*:100\""),
                        Map.entry("router.availabilityHysteresisPercent", "15"),
                        Map.entry("router.roles.prefill.availability.maxPendingRequests", "32"),
                        Map.entry("router.roles.prefill.availability.maxUncachedTokens", "65536"),
                        Map.entry(
                                "router.roles.prefill.availability.maxPredictedQueueWaitMs",
                                "1000"),
                        Map.entry("scheduler.capacity.maxOutstandingRequestsGlobal", "100000"),
                        Map.entry("scheduler.capacity.maxWaitingRequestsPerPrefillWorker", "1024"),
                        Map.entry("scheduler.lifecycle.staleInflightTimeoutMs", "300000"),
                        Map.entry("scheduler.lifecycle.deliveredNotAcceptedTimeoutMs", "30000"),
                        Map.entry(
                                "scheduler.lifecycle.maxDeliveredNotAcceptedRequestsGlobal", "200"),
                        Map.entry(
                                "scheduler.ordering.preemption.engineCancellation.ackTimeoutMs",
                                "50"),
                        Map.entry(
                                "scheduler.ordering.preemption.engineCancellation.completionTimeoutMs",
                                "1000"),
                        Map.entry("dispatcher.maxInflightRequestsPerPrefillWorker", "8"),
                        Map.entry("dispatcher.enqueueRpcTimeoutMs", "5000"),
                        Map.entry("router.roles.prefill.candidateChoice.type", "\"BEST_ONLY\""),
                        Map.entry("router.roles.prefill.candidateChoice.relativeTolerance", "0.1"),
                        Map.entry("router.roles.prefill.candidateChoice.minimumToleranceMs", "20"),
                        Map.entry("router.roles.prefill.candidateChoice.pool.type", "\"RATIO\""),
                        Map.entry("router.roles.prefill.candidateChoice.pool.ratio", "0.3"),
                        Map.entry("router.roles.prefill.candidateChoice.pool.minimumWorkers", "1"),
                        Map.entry("router.roles.prefill.candidateChoice.pool.workers", "1"),
                        Map.entry(
                                "router.roles.prefill.candidateChoice.outlierRejection.maxPendingVsAverageMultiplier",
                                "3"),
                        Map.entry(
                                "router.roles.prefill.candidateChoice.outlierRejection.maxProjectedDrainVsAverageMultiplier",
                                "3"),
                        Map.entry(
                                "router.roles.decode.kvReservation.maxOutputTokensForEstimate",
                                "1000"),
                        Map.entry("router.roles.decode.decayPerToken", "0.001"),
                        Map.entry("router.roles.decode.loadDecayPerRequest", "1"),
                        Map.entry(
                                "router.roles.decode.outlierRejection.maxEngineLoadVsAverageMultiplier",
                                "3"),
                        Map.entry(
                                "router.roles.decode.outlierRejection.maxKvUsedVsAverageMultiplier",
                                "3"));
        ObjectMapper json = new ObjectMapper();
        for (var entry : removed.entrySet()) {
            ObjectNode root =
                    (ObjectNode)
                            json.readTree(
                                    ConfigTestFixtures.document(
                                            "{\"scheduler\":{\"ordering\":{\"type\":\"PRIORITY\",\"preemption\":{\"allowedVictimStages\":[\"DECODE_ENGINE_OWNED\"]}}}}"));
            String[] names = entry.getKey().split("\\.");
            ObjectNode node = root;
            for (int i = 0; i < names.length - 1; i++) {
                node = node.with(names[i]);
            }
            node.set(names[names.length - 1], json.readTree(entry.getValue()));
            assertThrows(
                    ConfigValidationException.class,
                    () -> ConfigService.parse(root.toString()),
                    entry.getKey());
        }
    }

    @Test
    void rejects_json_ambiguity_and_type_coercion() {
        for (String document :
                new String[] {
                    ConfigTestFixtures.REQUIRED.replace(
                            "\"schemaVersion\": 4", "\"schemaVersion\": 4, \"schemaVersion\": 4"),
                    ConfigTestFixtures.REQUIRED + " {}",
                    ConfigTestFixtures.document("{\"router\":null}"),
                    ConfigTestFixtures.document("{\"schemaVersion\":\"3\"}"),
                    ConfigTestFixtures.document("{\"schemaVersion\":4.5}"),
                    ConfigTestFixtures.document("{\"internalRuntime\":{}}"),
                    ConfigTestFixtures.document(
                            "{\"requestLifecycle\":{\"request\":{\"timeoutMs\":\"60000\"}}}"),
                    ConfigTestFixtures.document(
                            "{\"requestLifecycle\":{\"decision\":{\"lifetime\":\"2\"}}}"),
                    ConfigTestFixtures.document("{\"scheduler\":{\"type\":0}}")
                }) {
            assertThrows(
                    ConfigValidationException.class, () -> ConfigService.parse(document), document);
        }
        for (int version : new int[] {1, 2, 3, 5}) {
            assertInvalid("{\"schemaVersion\":" + version + "}", "schemaVersion");
        }
    }

    @Test
    void decodeKvUsagePercentageRequiresOneThroughOneHundred() {
        String field = "router.roles.decode.availability.maxKvUsagePercent";
        assertEquals(
                90L,
                ConfigTestFixtures.parse("{}")
                        .getRouter()
                        .getRoles()
                        .getDecode()
                        .getAvailability()
                        .getMaxKvUsagePercent());
        for (long percent : new long[] {-1L, 0L, 101L}) {
            ConfigValidationException error =
                    assertThrows(
                            ConfigValidationException.class,
                            () ->
                                    ConfigTestFixtures.parse(
                                            "{\"router\":{\"roles\":{\"decode\":{\"availability\":{\"maxKvUsagePercent\":"
                                                    + percent
                                                    + "}}}}}"));
            assertTrue(error.getMessage().contains(field), error.getMessage());
            assertTrue(error.getMessage().contains("[1.0, 100.0]"), error.getMessage());
        }
        for (long percent : new long[] {1L, 90L, 100L}) {
            var config =
                    ConfigTestFixtures.parse(
                            "{\"router\":{\"roles\":{\"decode\":{\"availability\":{\"maxKvUsagePercent\":"
                                    + percent
                                    + "}}}}}");
            assertEquals(
                    percent,
                    config.getRouter()
                            .getRoles()
                            .getDecode()
                            .getAvailability()
                            .getMaxKvUsagePercent());
        }
    }

    @Test
    void shipped_examples_parse_using_the_runtime_contract() throws Exception {
        for (String name : new String[] {"flexlb-queue-priority.json"}) {
            String document =
                    java.nio.file.Files.readString(
                            java.nio.file.Path.of("../docs/config-examples", name));
            assertEquals(4, ConfigService.parse(document).getSchemaVersion());
        }
    }

    private static void assertInvalid(String patch, String field) {
        var error =
                assertThrows(
                        ConfigValidationException.class, () -> ConfigTestFixtures.parse(patch));
        assertTrue(error.getMessage().contains(field), error.getMessage());
    }
}
