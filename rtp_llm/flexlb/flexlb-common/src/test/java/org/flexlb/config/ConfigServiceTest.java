package org.flexlb.config;

import org.flexlb.service.config.ConfigSource;
import org.flexlb.service.config.NormalizedConfig;
import org.flexlb.service.config.merger.FlexlbConfigMerger;
import org.flexlb.service.config.parser.StandardConfigDocumentParser;
import org.flexlb.service.config.parser.V0ConfigDocumentParser;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import uk.org.webcompere.systemstubs.environment.EnvironmentVariables;

import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ConfigServiceTest {

    private static final String FLEXLB_CONFIG_ENV = "FLEXLB_CONFIG";
    private static final String MODEL_SERVICE_CONFIG_ENV = "MODEL_SERVICE_CONFIG";

    private ConfigService configService;

    @AfterEach
    void tearDown() {
        if (configService != null) {
            configService.close();
        }
    }

    @Test
    void empty_environment_uses_valid_defaults() {
        ConfigService configService = createConfigService(Map.of());
        FlexlbConfig config = configService.loadBalanceConfig();

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
        RoutingConfig.ExecutionTimeEstimatorConfig estimator =
                config.getRouter().getRoles().getPrefill()
                        .getExecutionTimeEstimator();
        assertEquals(RoutingConfig.EstimatorType.FORMULA, estimator.getType());
        assertEquals("sum(computeTokens) + 0.3*sum(hitCacheTokens)",
                estimator.getExpression());
    }

    @Test
    void configured_document_must_not_be_blank() {
        assertThrows(IllegalStateException.class,
                () -> createConfigService(Map.of(FLEXLB_CONFIG_ENV, "   ")));
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
        FlexlbConfig config = FlexlbConfigMerger.mergeWithDefaults("""
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
                  "enableFallback": true,
                  "fallbackBatchTokenCapacity": 2097152
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
        assertEquals(2_097_152L, config.getFallbackBatchTokenCapacity());
    }

    @Test
    void rejects_unknown_removed_and_inactive_fields() {
        assertThrows(ConfigValidationException.class,
                () -> FlexlbConfigMerger.mergeWithDefaults("{\"autoTpmEnabled\":true}"));
        assertThrows(ConfigValidationException.class,
                () -> FlexlbConfigMerger.mergeWithDefaults("{\"autoTpmSloLengthBuckets\":\"*:100\"}"));
        assertThrows(ConfigValidationException.class, () -> FlexlbConfigMerger.mergeWithDefaults("""
                {
                  "scheduler":{"type":"DIRECT","ordering":{"type":"FIFO"}},
                  "dispatcher":{"type":"NON_BATCH"}
                }
                """));
        assertThrows(ConfigValidationException.class, () -> FlexlbConfigMerger.mergeWithDefaults("""
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
    void rejects_duplicate_keys_nulls_and_scalar_coercion() {
        assertThrows(ConfigValidationException.class,
                () -> FlexlbConfigMerger.mergeWithDefaults("{\"schemaVersion\":1,\"schemaVersion\":1}"));
        assertThrows(ConfigValidationException.class,
                () -> FlexlbConfigMerger.mergeWithDefaults("{} {}"));
        assertThrows(ConfigValidationException.class,
                () -> FlexlbConfigMerger.mergeWithDefaults("{\"router\":null}"));
        assertThrows(ConfigValidationException.class,
                () -> FlexlbConfigMerger.mergeWithDefaults("{\"schemaVersion\":\"1\"}"));
        assertThrows(ConfigValidationException.class,
                () -> FlexlbConfigMerger.mergeWithDefaults("{\"schemaVersion\":0}"));
        assertThrows(ConfigValidationException.class,
                () -> FlexlbConfigMerger.mergeWithDefaults("{\"schemaVersion\":1.5}"));
        assertThrows(ConfigValidationException.class,
                () -> FlexlbConfigMerger.mergeWithDefaults("{\"internalRuntime\":{}}"));
    }

    @Test
    void validates_cross_component_semantics() {
        assertThrows(ConfigValidationException.class,
                () -> FlexlbConfigMerger.mergeWithDefaults("{\"fallbackBatchTokenCapacity\":0}"));
        assertThrows(ConfigValidationException.class, () -> FlexlbConfigMerger.mergeWithDefaults("""
                {
                  "scheduler":{"type":"DIRECT"},
                  "dispatcher":{"type":"BATCH"}
                }
                """));
        assertThrows(ConfigValidationException.class, () -> FlexlbConfigMerger.mergeWithDefaults("""
                {
                  "scheduler":{"type":"QUEUE","ordering":{"type":"PRIORITY","defaultPriority":101}},
                  "dispatcher":{"type":"NON_BATCH"}
                }
                """));
        assertThrows(ConfigValidationException.class, () -> FlexlbConfigMerger.mergeWithDefaults("""
                {
                  "scheduler":{
                    "type":"QUEUE",
                    "ordering":{"type":"PRIORITY","preemption":{}}
                  },
                  "dispatcher":{"type":"NON_BATCH"}
                }
                """));
        assertThrows(ConfigValidationException.class, () -> FlexlbConfigMerger.mergeWithDefaults("""
                {
                  "scheduler":{"type":"QUEUE","ordering":{"type":"PRIORITY",
                    "preemption":{"allowedVictimStages":["DECODE_ENGINE_OWNED"]}}},
                  "dispatcher":{"type":"NON_BATCH"}
                }
                """));
        assertThrows(ConfigValidationException.class, () -> FlexlbConfigMerger.mergeWithDefaults("""
                {
                  "scheduler":{"type":"QUEUE","ordering":{"type":"PRIORITY",
                    "preemption":{
                      "allowedVictimStages":["PREFILL_QUEUED"],
                      "engineCancellation":{"ackTimeoutMs":50,"completionTimeoutMs":1000}
                    }}},
                  "dispatcher":{"type":"NON_BATCH"}
                }
                """));
        assertThrows(ConfigValidationException.class, () -> FlexlbConfigMerger.mergeWithDefaults("""
                {
                  "scheduler":{
                    "type":"QUEUE","ordering":{"type":"PRIORITY"},
                    "maxRoutingRetries":3
                  },
                  "dispatcher":{"type":"NON_BATCH"}
                }
                """));
        assertThrows(ConfigValidationException.class, () -> FlexlbConfigMerger.mergeWithDefaults("""
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
        assertThrows(ConfigValidationException.class, () -> FlexlbConfigMerger.mergeWithDefaults("""
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
        FlexlbConfig config = FlexlbConfigMerger.mergeWithDefaults("""
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

    private ConfigService createConfigService(Map<String, String> environment) {
        try {
            return new EnvironmentVariables(environment).execute(() -> {
                ConfigService.register(new ConfigSource() {
                    @Override
                    public String name() {
                        return "environment";
                    }

                    @Override
                    public int priority() {
                        return 1;
                    }

                    @Override
                    public void setUpdateListener(Consumer<String> listener) {}

                    @Override
                    public String load() {
                        String content = environment.get(FLEXLB_CONFIG_ENV);
                        if (content != null && content.isBlank()) {
                            throw new IllegalArgumentException(FLEXLB_CONFIG_ENV + " must not be blank when configured");
                        }
                        return content;
                    }

                    @Override
                    public String loadModelServiceConfig() {
                        return environment.get(MODEL_SERVICE_CONFIG_ENV);
                    }

                    @Override
                    public NormalizedConfig loadConfig() {
                        String rawFlexlbConfig = load();
                        return rawFlexlbConfig == null
                                ? new NormalizedConfig(null, loadModelServiceConfig(), ConfigSchemaVersion.V0_COMPATIBILITY)
                                : normalize(rawFlexlbConfig);
                    }
                });
                configService = new ConfigService(List.of(new StandardConfigDocumentParser(), new V0ConfigDocumentParser()));
                return configService;
            });
        } catch (RuntimeException error) {
            throw error;
        } catch (Exception error) {
            throw new IllegalStateException("Failed to construct ConfigService for test", error);
        }
    }
}
