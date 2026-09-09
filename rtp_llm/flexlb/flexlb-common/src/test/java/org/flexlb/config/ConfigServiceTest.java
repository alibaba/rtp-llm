package org.flexlb.config;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.config.RoutingConfig.EstimatedTtftSelectorConfig;
import org.flexlb.config.RoutingConfig.FormulaEstimatorConfig;
import org.flexlb.config.RoutingConfig.RandomWithinToleranceConfig;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ConfigServiceTest {
    private static void assertInvalid(String document, String field) {
        var error = assertThrows(ConfigValidationException.class, () -> ConfigService.parse(document));
        assertTrue(error.getMessage().contains(field), error.getMessage());
    }

    @Test
    void decode_cost_estimator_accepts_custom_formulas() {
        for (String expression : new String[]{"kvcache_used_ratio", "running_size",
                "sqrt(running_size) + pow(kvcache_used_ratio, 2)",
                "0.3 * running_size / max_running_size + 0.7 * kvcache_used_ratio", "-2"}) {
            FlexlbConfig config = ConfigService.parse("""
                    {"router":{"roles":{"decode":{"costEstimator":{"expression":"%s"},
                      "availability":{"maxEngineRequests":128}}}}}
                    """.formatted(expression));
            assertEquals(expression, config.getRouter().getRoles().getDecode().getCostEstimator().getExpression());
            assertEquals(90, config.getRouter().getRoles().getDecode().getAvailability().getMaxKvUsagePercent());
        }
        FlexlbConfig config = ConfigService.parse("""
                {"router":{"roles":{"decode":{"costEstimator":{"expression":"running_size"}}}}}
                """);
        assertNull(config.getRouter().getRoles().getDecode().getAvailability().getMaxEngineRequests());
    }

    @Test
    void decode_max_running_size_variable_requires_positive_max_engine_requests() {
        assertInvalid("""
                {"router":{"roles":{"decode":{"costEstimator":{"expression":"running_size / max_running_size"}}}}}
                """, "maxEngineRequests");
        for (int maxRequests : new int[]{0, -1}) {
            assertInvalid("""
                    {"router":{"roles":{"decode":{"costEstimator":{"expression":"max_running_size"},
                      "availability":{"maxEngineRequests":%s}}}}}
                    """.formatted(maxRequests), "maxEngineRequests");
        }
        FlexlbConfig config = ConfigService.parse("{}");
        assertNull(config.getRouter().getRoles().getDecode().getAvailability().getMaxEngineRequests());
        config.getRouter().getRoles().getDecode().getCostEstimator().setExpression("max_running_size");
        var error = assertThrows(ConfigValidationException.class, () -> FlexlbConfigValidator.validate(config));
        assertTrue(error.getMessage().contains("maxEngineRequests"), error.getMessage());
    }

    @Test
    void decode_cost_estimator_rejects_invalid_formulas_and_json_coercion() {
        for (String expression : new String[]{"", "   ", "running_size +", "unknown_variable",
                "sum(running_size)", "sqrt()"}) {
            assertInvalid("""
                    {"router":{"roles":{"decode":{"costEstimator":{"expression":"%s"}}}}}
                    """.formatted(expression), "expression");
        }
        for (String value : new String[]{"0", "1.5", "true", "null", "[]", "{}"}) {
            assertInvalid("""
                    {"router":{"roles":{"decode":{"costEstimator":{"expression":%s}}}}}
                    """.formatted(value), "expression");
        }
        assertInvalid("{\"router\":{\"roles\":{\"decode\":{\"costEstimator\":null}}}}", "costEstimator");
    }

    @Test
    void decode_cost_estimator_validates_programmatic_values() {
        FlexlbConfig config = ConfigService.parse("{}");
        var estimator = config.getRouter().getRoles().getDecode().getCostEstimator();
        for (String invalid : new String[]{null, "", "   ", "running_size +", "unknown_variable"}) {
            estimator.setExpression(invalid);
            assertThrows(ConfigValidationException.class, () -> FlexlbConfigValidator.validate(config));
        }
        config.getRouter().getRoles().getDecode().setCostEstimator(null);
        assertThrows(ConfigValidationException.class, () -> FlexlbConfigValidator.validate(config));
    }

    @Test
    void decode_cost_estimator_reuses_compiled_expression_without_exposing_its_cache() {
        FlexlbConfig config = ConfigService.parse("{}");
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
        assertInvalid("""
                {"router":{"roles":{"decode":{"costEstimator":{"compiledCost":{}}}}}}
                """, "compiledCost");
    }



    @Test
    void empty_environment_uses_valid_defaults() {
        FlexlbConfig config = new ConfigService(Map.of()).loadBalanceConfig();

        assertTrue(config.isQueue());
        assertFalse(config.isPriorityOrdering());
        assertTrue(config.isBatchDispatch());
        assertEquals(1, config.getSchemaVersion());
    }

    @Test
    void configured_document_must_not_be_blank() {
        assertThrows(ConfigValidationException.class,
                () -> new ConfigService(Map.of(ConfigService.FLEXLB_CONFIG_ENV, "   ")));
    }

    @Test
    void parses_complete_responsibility_oriented_document() {
        FlexlbConfig config = ConfigService.parse("""
                {
                  "schemaVersion": 1,
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
                    "capacity": {
                      "maxOutstandingRequestsGlobal": 2000
                    },
                    "lifecycle": {
                      "staleInflightTimeoutMs": 300000,
                      "deliveredNotAcceptedTimeoutMs": 30000,
                      "maxDeliveredNotAcceptedRequestsGlobal": 200
                    }
                  },
                  "dispatcher": {
                    "type": "BATCH",
                    "maxRequests": 16,
                    "maxCollectionWaitMs": 50,
                    "maxWaitingRequestsPerPrefillWorker": 256,
                    "earlyDispatchPredictedExecutionMs": 100,
                    "maxInflightBatchesPerPrefillWorker": 2,
                    "enqueueRpcTimeoutMs": 4000
                  },
                  "router": {
                    "availabilityHysteresisPercent": 10,
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
                        "availability": {"maxPendingRequests": 32},
                        "executionTimeEstimator": {
                          "type": "FORMULA",
                          "expression": "sum(computeTokens)"
                        },
                        "selector": {
                          "type": "ESTIMATED_TTFT",
                          "candidateChoice": {
                            "type": "RANDOM_WITHIN_TOLERANCE",
                            "relativeTolerance": 0.2,
                            "minimumToleranceMs": 10,
                            "outlierRejection": {
                              "maxPendingVsAverageMultiplier": 2.0,
                              "maxWaitVsAverageMultiplier": 2.5
                            }
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
                        "selector": {
                          "type": "KV_USAGE_WEIGHTED_RANDOM",
                          "decayPerToken": 0.002,
                          "outlierRejection": {
                            "maxEngineLoadVsAverageMultiplier": 2.0,
                            "maxKvUsedVsAverageMultiplier": 2.0
                          }
                        }
                      },
                      "vit": {"selector": {"type": "RANDOM"}}
                    }
                  },
                  "workerRegistry": {
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
        assertTrue(config.isBatchDispatch());
        assertEquals(60, config.priorityOrdering().getDefaultPriority());
        assertEquals(75, config.priorityOrdering().getPreemption()
                .getEngineCancellation().getAckTimeoutMs());
        assertEquals(1200, config.priorityOrdering().getPreemption()
                .getEngineCancellation().getCompletionTimeoutMs());
        assertEquals(16, config.batchDispatcher().getMaxRequests());
        FormulaEstimatorConfig estimator = assertInstanceOf(FormulaEstimatorConfig.class,
                config.getRouter().getRoles().getPrefill().getExecutionTimeEstimator());
        assertEquals("sum(computeTokens)", estimator.getExpression());
        EstimatedTtftSelectorConfig selector = assertInstanceOf(
                EstimatedTtftSelectorConfig.class,
                config.getRouter().getRoles().getPrefill().getSelector());
        RandomWithinToleranceConfig candidateChoice = assertInstanceOf(
                RandomWithinToleranceConfig.class, selector.getCandidateChoice());
        assertEquals(2.0, candidateChoice.getOutlierRejection()
                .getMaxPendingVsAverageMultiplier());
        assertEquals(2.5, candidateChoice.getOutlierRejection()
                .getMaxWaitVsAverageMultiplier());
        assertEquals(128L, config.getRouter().getRoles().getDecode()
                .getAvailability().getMaxEngineRequests());
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
        assertThrows(ConfigValidationException.class, () -> ConfigService.parse("""
                {
                  "router":{"roles":{"prefill":{"selector":{
                    "type":"ESTIMATED_TTFT",
                    "candidateChoice":{
                      "type":"LEAST_RECENTLY_USED_IN_POOL",
                      "outlierRejection":{
                        "maxPendingVsAverageMultiplier":2.0,
                        "maxWaitVsAverageMultiplier":2.0
                      }
                    }
                  }}}}
                }
                """));
        assertThrows(ConfigValidationException.class, () -> ConfigService.parse("""
                {
                  "router":{"roles":{"prefill":{
                    "selector":{"type":"RANDOM"},
                    "cacheAffinity":{"maxExtraTtftMs":10,"minPrefixHitPercent":5}
                  }}},
                  "scheduler":{"type":"DIRECT"},
                  "dispatcher":{"type":"NON_BATCH"}
                }
                """));
    }

    @Test
    void rejects_duplicate_keys_nulls_and_scalar_coercion() {
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
    void omission_is_the_only_unbounded_representation() {
        FlexlbConfig config = ConfigService.parse("""
                {
                  "scheduler":{"type":"QUEUE","ordering":{"type":"FIFO"}},
                  "dispatcher":{"type":"NON_BATCH"},
                  "router":{"roles":{"decode":{"availability":{"maxKvUsagePercent":90}}}}
                }
                """);

        assertNull(config.nonBatchDispatcher().getMaxInflightRequestsPerPrefillWorker());
        assertNull(config.getRouter().getRoles().getDecode()
                .getAvailability().getMaxEngineRequests());
    }
}
