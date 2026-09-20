package org.flexlb.config;

import org.flexlb.enums.EngineType;
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
    void empty_environment_uses_valid_defaults() {
        FlexlbConfig config = new ConfigService(Map.of()).loadBalanceConfig();

        assertTrue(config.isQueue());
        assertFalse(config.isPriorityOrdering());
        assertEquals(DispatcherConfig.Type.BATCH,
                config.getDispatcher().getType());
        assertTrue(config.isFixedWindowDecision());
        assertEquals(2, config.getSchemaVersion());
        assertEquals(1000, config.getRouter().getBatchScheduleMaxCount());
        assertEquals(EngineType.LLM, config.getWorkerRegistry().getEngineType());

        // An omitted estimator keeps the upstream default expression
        // (the legacy 1 ms/token sum). Test lines that need the production
        // DSv4 prefill fit inject it explicitly in their FLEXLB_CONFIG
        // documents (harness.py / master_fixed_window.json).
        RoutingConfig.ExecutionTimeEstimatorConfig estimator =
                config.getRouter().getRoles().getPrefill()
                        .getExecutionTimeEstimator();
        assertEquals(RoutingConfig.EstimatorType.FORMULA, estimator.getType());
        assertEquals("sum(computeTokens) + 0.3*sum(hitCacheTokens)",
                estimator.getExpression());
    }

    @Test
    void configured_document_must_not_be_blank() {
        assertThrows(ConfigValidationException.class,
                () -> new ConfigService(Map.of(ConfigService.FLEXLB_CONFIG_ENV, "   ")));
    }

    @Test
    void rejects_unsupported_schema_instead_of_migrating_it() {
        ConfigValidationException failure = assertThrows(
                ConfigValidationException.class,
                () -> ConfigService.parse("{\"schemaVersion\":1}"));

        assertTrue(failure.getMessage().contains("schemaVersion"));
    }

    @Test
    void removed_legacy_environment_fails_fast_with_migration_guidance() {
        ConfigValidationException failure = assertThrows(
                ConfigValidationException.class,
                () -> new ConfigService(Map.of(
                        ConfigService.FLEXLB_CONFIG_ENV, "{}",
                        "CACHE_STATUS_MAX_INTERVAL_MS", "100",
                        "DEFAULT_SCHEDULE_MODE", "QUEUE",
                        "FLEXLB_MONITOR_MODE", "all",
                        "ENGINE_TYPE", "EMBEDDING", "FLEXLB_ENGINE_TYPE", "EMBEDDING",
                        "BATCH_SCHEDULE_MAX_COUNT", "16", "BATCH_LOAD_BALANCE_STRATEGY", "ROUND_ROBIN")));

        for (String key : new String[]{"ENGINE_TYPE", "FLEXLB_ENGINE_TYPE", "BATCH_SCHEDULE_MAX_COUNT", "BATCH_LOAD_BALANCE_STRATEGY"}) {
            assertTrue(failure.getMessage().contains(key));
        }
        assertTrue(failure.getMessage().contains("CACHE_STATUS_MAX_INTERVAL_MS"));
        assertTrue(failure.getMessage().contains("DEFAULT_SCHEDULE_MODE"));
        assertTrue(failure.getMessage().contains("FLEXLB_MONITOR_MODE"));
        assertTrue(failure.getMessage().contains("schemaVersion 2"));
        assertTrue(failure.getMessage().contains("FLEXLB_MONITOR_METRIC_WHITELIST"));
    }

    @Test
    void supported_monitor_environment_is_not_treated_as_legacy() {
        FlexlbConfig config = new ConfigService(Map.of(
                "FLEXLB_MONITOR_METRIC_WHITELIST", "flexlb_"))
                .loadBalanceConfig();

        assertEquals(FlexlbConfig.CURRENT_SCHEMA_VERSION, config.getSchemaVersion());
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
    void parses_complete_responsibility_oriented_document() {
        FlexlbConfig config = ConfigService.parse("""
                {
                  "schemaVersion": 2,
                  "scheduler": {
                    "type": "QUEUE",
                    "ordering": {
                      "type": "PRIORITY",
                      "defaultPriority": 60,
                      "preemption": {
                        "allowedVictimStages": ["PREFILL_QUEUED", "DECODE_ENGINE_OWNED"],
                        "engineCancellation": {
                          "ackTimeoutMs": 75,
                          "completionTimeoutMs": 1200
                        }
                      }
                    },
                    "decision": {
                      "type": "FIXED_WINDOW",
                      "maxRequests": 12,
                      "maxCollectionWaitMs": 40,
                      "maxPredictedExecutionMs": 90
                    },
                    "capacity": {
                      "maxOutstandingRequestsGlobal": 2000,
                      "maxWaitingRequestsPerPrefillWorker": 192
                    },
                    "lifecycle": {
                      "staleInflightTimeoutMs": 300000,
                      "deliveredNotAcceptedTimeoutMs": 30000,
                      "maxDeliveredNotAcceptedRequestsGlobal": 200
                    }
                  },
                  "dispatcher": {
                    "type": "BATCH",
                    "maxInflightBatchesPerPrefillWorker": 2,
                    "enqueueRpcTimeoutMs": 4000
                  },
                  "router": {
                    "batchScheduleMaxCount": 32,
                    "groupSelector": {
                      "defaultTargets": [{"group": "blue", "weight": 1}],
                      "rules": [{
                        "name": "long-context",
                        "match": {"inputTokens": {"min": 4096}},
                        "targets": [{"group": "long", "weight": 1}]
                      }]
                    },
                    "roles": {
                      "prefill": {
                        "executionTimeEstimator": {
                          "type": "FORMULA",
                          "expression": "sum(computeTokens)"
                        },
                        "candidateChoice": {
                          "type": "RANDOM_WITHIN_TOLERANCE",
                          "relativeTolerance": 0.2,
                          "minimumToleranceMs": 10,
                          "outlierRejection": {
                            "maxPendingVsAverageMultiplier": 2.0,
                            "maxProjectedDrainVsAverageMultiplier": 2.5
                          }
                        },
                        "cacheAffinity": {
                          "maxExtraTtftMs": 25,
                          "minPrefixHitPercent": 10
                        }
                      },
                      "decode": {
                        "availability": {
                          "maxKvUsagePercent": 85,
                          "maxEngineRequests": 128
                        },
                        "kvReservation": {"maxOutputTokensForEstimate": 2048},
                        "decayPerToken": 0.002,
                        "outlierRejection": {
                          "maxEngineLoadVsAverageMultiplier": 2.0,
                          "maxKvUsedVsAverageMultiplier": 2.0
                        }
                      }
                    }
                  },
                  "workerRegistry": {
                    "engineType": "EMBEDDING",
                    "health": {
                      "statusPollIntervalMs": 25,
                      "statusRpcTimeoutMs": 5000,
                      "statusStaleAfterMs": 10000
                    },
                    "cacheStatus": {
                      "minRefreshIntervalMs": 100,
                      "maxRefreshIntervalMs": 2000
                    }
                  },
                  "observability": {
                    "cacheHit": {
                      "recentKeyWindow": {
                        "writeEnabled": true,
                        "durationMs": 60000,
                        "maxKeyOccurrences": 100000
                      },
                      "metricsEnabled": true,
                      "requestTraceLogEnabled": false,
                      "theoryLog": {"path": "/tmp/flexlb-theory.log"}
                    }
                  }
                }
                """);

        assertTrue(config.isPriorityOrdering());
        DispatcherConfig dispatcher = config.getDispatcher();
        assertEquals(32, config.getRouter().getBatchScheduleMaxCount());
        assertEquals(EngineType.EMBEDDING, config.getWorkerRegistry().getEngineType());
        assertEquals(DispatcherConfig.Type.BATCH, dispatcher.getType());
        assertEquals(60, config.priorityOrdering().getDefaultPriority());
        assertEquals(75, config.priorityOrdering().getPreemption()
                .getEngineCancellation().getAckTimeoutMs());
        assertEquals(1200, config.priorityOrdering().getPreemption()
                .getEngineCancellation().getCompletionTimeoutMs());
        assertEquals(12, config.fixedWindowDecision().getMaxRequests());
        assertEquals(40L, config.fixedWindowDecision().getMaxCollectionWaitMs());
        assertEquals(90L, config.fixedWindowDecision()
                .getMaxPredictedExecutionMs().longValue());
        assertEquals(192, config.queueScheduler().getCapacity()
                .getMaxWaitingRequestsPerPrefillWorker());
        assertEquals(2, dispatcher
                .getMaxInflightBatchesPerPrefillWorker().intValue());
        RoutingConfig.ExecutionTimeEstimatorConfig estimator =
                config.getRouter().getRoles().getPrefill()
                        .getExecutionTimeEstimator();
        assertEquals(RoutingConfig.EstimatorType.FORMULA, estimator.getType());
        assertEquals("sum(computeTokens)", estimator.getExpression());
        RoutingConfig.CandidateChoiceConfig candidateChoice =
                config.getRouter().getRoles().getPrefill()
                        .getCandidateChoice();
        assertEquals(RoutingConfig.CandidateChoiceType.RANDOM_WITHIN_TOLERANCE,
                candidateChoice.getType());
        assertEquals(2.0, candidateChoice.getOutlierRejection()
                .getMaxPendingVsAverageMultiplier());
        assertEquals(2.5, candidateChoice.getOutlierRejection()
                .getMaxProjectedDrainVsAverageMultiplier());
        RoutingConfig.CacheAffinityConfig cacheAffinity =
                config.getRouter().getRoles().getPrefill().getCacheAffinity();
        assertEquals(25L, cacheAffinity.getMaxExtraTtftMs());
        assertEquals(10.0, cacheAffinity.getMinPrefixHitPercent());
        assertEquals(128L, config.getRouter().getRoles().getDecode()
                .getAvailability().getMaxEngineRequests());
        assertEquals(1.0, config.getRouter().getRoles().getDecode()
                .getLoadDecayPerRequest());
        assertEquals(1, config.getRouter().getGroupSelector().getRules().size());
    }

    @Test
    void rejects_unknown_removed_and_inactive_fields() {
        assertThrows(ConfigValidationException.class,
                () -> ConfigService.parse("{\"autoTpmEnabled\":true}"));
        assertThrows(ConfigValidationException.class,
                () -> ConfigService.parse("{\"autoTpmSloLengthBuckets\":\"*:100\"}"));
        assertThrows(ConfigValidationException.class, () -> ConfigService.parse("""
                {
                  "scheduler":{"type":"DIRECT","ordering":{"type":"FIFO"}},
                  "dispatcher":{"type":"NON_BATCH"}
                }
                """));
        assertThrows(ConfigValidationException.class, () -> ConfigService.parse("""
                {
                  "scheduler":{"type":"DIRECT"},
                  "dispatcher":{"type":"NON_BATCH","maxRequests":8}
                }
                """));
        ConfigValidationException removedPrefillAvailability = assertThrows(
                ConfigValidationException.class, () -> ConfigService.parse("""
                {
                  "router":{"roles":{"prefill":{
                    "availability":{"maxPendingRequests":32}
                  }}}
                }
                """));
        assertTrue(removedPrefillAvailability.getMessage()
                .contains("availability"));
        ConfigValidationException removedHysteresis = assertThrows(
                ConfigValidationException.class, () -> ConfigService.parse("""
                {"router":{"availabilityHysteresisPercent":15}}
                """));
        assertTrue(removedHysteresis.getMessage()
                .contains("availabilityHysteresisPercent"));
    }

    @Test
    void rejects_inactive_routing_variant_fields() {
        for (String document : new String[]{
                """
                {"router":{"roles":{"prefill":{"executionTimeEstimator":{
                  "type":"LEARNING","expression":"sum(computeTokens)"
                }}}}}
                """,
                """
                {"router":{"roles":{"prefill":{"candidateChoice":{
                  "type":"BEST_ONLY","relativeTolerance":0.1
                }}}}}
                """,
                """
                {"router":{"roles":{"prefill":{"candidateChoice":{
                  "type":"RANDOM_WITHIN_TOLERANCE","pool":{"type":"RATIO"}
                }}}}}
                """,
                """
                {"router":{"roles":{"prefill":{"candidateChoice":{
                  "type":"LEAST_RECENTLY_USED_IN_POOL","minimumToleranceMs":20,
                  "pool":{"type":"RATIO"}
                }}}}}
                """,
                """
                {"router":{"roles":{"prefill":{"candidateChoice":{
                  "type":"LEAST_RECENTLY_USED_IN_POOL",
                  "pool":{"type":"RATIO","workers":2}
                }}}}}
                """,
                """
                {"router":{"roles":{"prefill":{"candidateChoice":{
                  "type":"LEAST_RECENTLY_USED_IN_POOL",
                  "pool":{"type":"FIXED","ratio":0.5,"minimumWorkers":1}
                }}}}}
                """}) {
            ConfigValidationException failure = assertThrows(
                    ConfigValidationException.class,
                    () -> ConfigService.parse(document), document);
            assertTrue(failure.getMessage().contains("active mode"), document);
        }
    }

    @Test
    void rejects_duplicate_keys_nulls_and_scalar_coercion() {
        for (String json : new String[]{"{\"router\":{\"batchScheduleMaxCount\":0}}",
                "{\"workerRegistry\":{\"engineType\":\"UNKNOWN\"}}", "{\"batchScheduleMaxCount\":32}"}) {
            assertThrows(ConfigValidationException.class, () -> ConfigService.parse(json));
        }
        assertThrows(ConfigValidationException.class,
                () -> ConfigService.parse("{\"schemaVersion\":1,\"schemaVersion\":1}"));
        assertThrows(ConfigValidationException.class,
                () -> ConfigService.parse("{} {}"));
        assertThrows(ConfigValidationException.class,
                () -> ConfigService.parse("{\"router\":null}"));
        assertThrows(ConfigValidationException.class,
                () -> ConfigService.parse("{\"schemaVersion\":\"1\"}"));
        assertThrows(ConfigValidationException.class,
                () -> ConfigService.parse("{\"schemaVersion\":1.5}"));
        assertThrows(ConfigValidationException.class,
                () -> ConfigService.parse("{\"internalRuntime\":{}}"));
    }

    @Test
    void validates_cross_component_semantics() {
        assertThrows(ConfigValidationException.class, () -> ConfigService.parse("""
                {
                  "scheduler":{"type":"DIRECT"},
                  "dispatcher":{"type":"BATCH"}
                }
                """));
        assertThrows(ConfigValidationException.class, () -> ConfigService.parse("""
                {
                  "scheduler":{"type":"QUEUE","ordering":{"type":"PRIORITY","defaultPriority":101}},
                  "dispatcher":{"type":"NON_BATCH"}
                }
                """));
        assertThrows(ConfigValidationException.class, () -> ConfigService.parse("""
                {
                  "scheduler":{
                    "type":"QUEUE",
                    "ordering":{"type":"PRIORITY","preemption":{}}
                  },
                  "dispatcher":{"type":"NON_BATCH"}
                }
                """));
        assertThrows(ConfigValidationException.class, () -> ConfigService.parse("""
                {
                  "scheduler":{"type":"QUEUE","ordering":{"type":"PRIORITY",
                    "preemption":{"allowedVictimStages":["DECODE_ENGINE_OWNED"]}}},
                  "dispatcher":{"type":"NON_BATCH"}
                }
                """));
        assertThrows(ConfigValidationException.class, () -> ConfigService.parse("""
                {
                  "scheduler":{"type":"QUEUE","ordering":{"type":"PRIORITY",
                    "preemption":{
                      "allowedVictimStages":["PREFILL_QUEUED"],
                      "engineCancellation":{"ackTimeoutMs":50,"completionTimeoutMs":1000}
                    }}},
                  "dispatcher":{"type":"NON_BATCH"}
                }
                """));
        assertThrows(ConfigValidationException.class, () -> ConfigService.parse("""
                {
                  "scheduler":{
                    "type":"QUEUE","ordering":{"type":"PRIORITY"},
                    "maxRoutingRetries":3
                  },
                  "dispatcher":{"type":"NON_BATCH"}
                }
                """));
        assertThrows(ConfigValidationException.class, () -> ConfigService.parse("""
                {
                  "workerRegistry": {
                    "health": {
                      "statusPollIntervalMs":20,
                      "statusRpcTimeoutMs":5000,
                      "statusStaleAfterMs":9999
                    },
                    "cacheStatus":{"minRefreshIntervalMs":50,"maxRefreshIntervalMs":3000}
                  }
                }
                """));
        assertThrows(ConfigValidationException.class, () -> ConfigService.parse("""
                {
                  "router":{"roles":{"prefill":{"executionTimeEstimator":{
                    "type":"FORMULA","expression":"sum(unknownTokens)"
                  }}}}
                }
                """));
    }

    @Test
    void validates_cache_affinity_bounds_from_json() {
        assertInvalidCacheAffinity(-1, 5);
        assertInvalidCacheAffinity(0, -0.1);
        assertInvalidCacheAffinity(0, 100.1);
    }

    private static void assertInvalidCacheAffinity(
            long maxExtraTtftMs, double minPrefixHitPercent) {
        assertThrows(ConfigValidationException.class, () -> ConfigService.parse("""
                {
                  "router":{"roles":{"prefill":{
                    "cacheAffinity":{
                      "maxExtraTtftMs":%d,
                      "minPrefixHitPercent":%s
                    }
                  }}}
                }
                """.formatted(maxExtraTtftMs, minPrefixHitPercent)));
    }

    @Test
    void omission_is_the_only_unbounded_representation() {
        FlexlbConfig config = ConfigService.parse("""
                {
                  "scheduler":{"type":"QUEUE","ordering":{"type":"FIFO"}},
                  "dispatcher":{"type":"NON_BATCH"},
                  "router":{"roles":{"decode":{"availability":{"maxKvUsagePercent":90}}}}
                }
                """);

        DispatcherConfig dispatcher = config.getDispatcher();
        assertEquals(DispatcherConfig.Type.NON_BATCH, dispatcher.getType());
        assertNull(dispatcher.getMaxInflightRequestsPerPrefillWorker());
        assertNull(config.getRouter().getRoles().getDecode()
                .getAvailability().getMaxEngineRequests());
    }
}
