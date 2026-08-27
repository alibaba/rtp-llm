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
        ConfigService configService = new ConfigService(Map.of());
        FlexlbConfig config = configService.loadBalanceConfig();

        assertTrue(config.isQueue());
        assertFalse(config.isPriorityOrdering());
        assertTrue(config.isBatchDispatch());
        assertFalse(config.isEnableFallback());
        assertEquals(1, config.getSchemaVersion());
        assertNull(configService.loadModelServiceConfig());
    }

    @Test
    void parses_explicit_fallback_switch() {
        FlexlbConfig config = ConfigService.parse("{\"enableFallback\":true}");

        assertTrue(config.isEnableFallback());
    }

    @Test
    void loads_model_topology_independently_from_scheduling_config() {
        ConfigService configService = new ConfigService(Map.of(
                ConfigService.FLEXLB_CONFIG_ENV, """
                        {
                          "scheduler":{"type":"DIRECT"},
                          "dispatcher":{"type":"NON_BATCH"}
                        }
                        """,
                ConfigService.MODEL_SERVICE_CONFIG_ENV, """
                        {
                          "service_id":"test-service",
                          "role_endpoints":[]
                        }
                        """));

        assertTrue(configService.loadBalanceConfig().isDirect());
        assertEquals("test-service", configService.loadModelServiceConfig().getServiceId());
        assertTrue(configService.loadModelServiceConfig().getRoleEndpoints().isEmpty());
    }

    @Test
    void rejects_invalid_model_service_config_document() {
        ConfigValidationException error = assertThrows(ConfigValidationException.class,
                () -> new ConfigService(Map.of(
                        ConfigService.MODEL_SERVICE_CONFIG_ENV, "not-json")));

        assertTrue(error.getMessage().contains("'MODEL_SERVICE_CONFIG'"));
    }

    @Test
    void rejects_flexlb_behavior_inside_model_service_config() {
        ConfigValidationException kvcmError = assertThrows(ConfigValidationException.class,
                () -> ConfigService.parseModelServiceConfig("""
                        {
                          "service_id":"test-service",
                          "role_endpoints":[],
                          "kvcm":{"enabled":true,"address":"127.0.0.1"}
                        }
                        """));
        assertTrue(kvcmError.getMessage().contains("FLEXLB_CONFIG"));
        assertTrue(kvcmError.getMessage().contains("$.kvcm.enabled"));

        ConfigValidationException legacyError = assertThrows(ConfigValidationException.class,
                () -> ConfigService.parseModelServiceConfig("""
                        {
                          "service_id":"test-service",
                          "load_balance":true,
                          "role_endpoints":[]
                        }
                        """));
        assertTrue(legacyError.getMessage().contains("FLEXLB_CONFIG"));
        assertTrue(legacyError.getMessage().contains("$.load_balance"));
    }

    @Test
    void blank_model_service_config_is_treated_as_missing() {
        ConfigService configService = new ConfigService(Map.of(
                ConfigService.MODEL_SERVICE_CONFIG_ENV, "   "));

        assertNull(configService.loadModelServiceConfig());
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
                  },
                  "serviceDiscovery": {
                    "connectTimeoutMs": 600,
                    "readTimeoutMs": 700,
                    "pollIntervalMs": 1200,
                    "maxIdleConnections": 6,
                    "keepAliveDurationMs": 240000
                  },
                  "cacheMatching": {
                    "type": "KVCM",
                    "requestTimeoutMs": 800,
                    "leaderRefreshIntervalMs": 15000,
                    "heartbeatFailureThreshold": 4,
                    "queryFailureThreshold": 12,
                    "maxQueryRetryCount": 2,
                    "recoverySuccessThreshold": 5,
                    "p2pHostCount": 3,
                    "localStandby": {
                      "autoSwitch": false,
                      "blockSize": 64,
                      "ttlMs": 400000,
                      "minimumTtlMs": 120000,
                      "ttlReductionStartRatio": 0.75,
                      "maximumEntries": 3000000,
                      "capacityMultiplier": 12,
                      "asyncQueueCapacity": 120000,
                      "hashThreadCount": 6,
                      "hashQueueCapacity": 130000
                    }
                  },
                  "optimizer": {
                    "enabled": true,
                    "discoveryPollIntervalMs": 1500
                  },
                  "consistency": {
                    "type": "ZOOKEEPER",
                    "connectString": "zk-1:2181,zk-2:2181",
                    "sessionTimeoutMs": 31000,
                    "connectionTimeoutMs": 32000,
                    "masterRefreshIntervalMs": 6000
                  },
                  "blockHashStrategy": "SGLANG",
                  "enableFallback": true
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
        assertEquals(600, config.getServiceDiscovery().getConnectTimeoutMs());
        KvcmCacheMatchingConfig kvcm = assertInstanceOf(
                KvcmCacheMatchingConfig.class, config.getCacheMatching());
        assertEquals(800, kvcm.getRequestTimeoutMs());
        assertEquals(64, kvcm.getLocalStandby().getBlockSize());
        assertTrue(config.getOptimizer().isEnabled());
        ZookeeperConsistencyConfig consistency = assertInstanceOf(
                ZookeeperConsistencyConfig.class, config.getConsistency());
        assertEquals("zk-1:2181,zk-2:2181", consistency.getConnectString());
        assertTrue(config.isEnableFallback());
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
