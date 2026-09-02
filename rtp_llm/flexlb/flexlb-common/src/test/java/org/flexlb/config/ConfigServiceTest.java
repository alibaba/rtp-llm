package org.flexlb.config;

import org.flexlb.config.RoutingConfig.EstimatedTtftSelectorConfig;
import org.flexlb.config.RoutingConfig.FormulaEstimatorConfig;
import org.flexlb.config.RoutingConfig.RandomWithinToleranceConfig;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
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

        // An omitted estimator keeps the upstream default expression
        // (the legacy 1 ms/token sum). Test lines that need the production
        // DSv4 prefill fit inject it explicitly in their FLEXLB_CONFIG
        // documents (harness.py / master_fixed_window.json).
        FormulaEstimatorConfig estimator = assertInstanceOf(
                FormulaEstimatorConfig.class,
                config.getRouter().getRoles().getPrefill()
                        .getExecutionTimeEstimator());
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
                              "maxProjectedDrainVsAverageMultiplier": 2.5
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
        DispatcherConfig dispatcher = config.getDispatcher();
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
                .getMaxProjectedDrainVsAverageMultiplier());
        RoutingConfig.CacheAffinityConfig cacheAffinity =
                config.getRouter().getRoles().getPrefill().getCacheAffinity();
        assertEquals(25L, cacheAffinity.getMaxExtraTtftMs());
        assertEquals(10.0, cacheAffinity.getMinPrefixHitPercent());
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
                        "maxProjectedDrainVsAverageMultiplier":2.0
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
