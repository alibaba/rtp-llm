package org.flexlb.config;

import org.flexlb.config.RoutingConfig.FormulaEstimatorConfig;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FlexlbConfigStartupMigrationTest {

    @Test
    void live_non_batch_v1_shape_changes_schema_without_changing_configuration() {
        FlexlbConfig config = ConfigService.parse("""
                {
                  "schemaVersion":1,
                  "scheduler":{
                    "type":"QUEUE",
                    "ordering":{"type":"PRIORITY"},
                    "capacity":{"maxOutstandingRequestsGlobal":500000},
                    "lifecycle":{
                      "staleInflightTimeoutMs":300000,
                      "deliveredNotAcceptedTimeoutMs":30000,
                      "maxDeliveredNotAcceptedRequestsGlobal":500000
                    },
                    "queueTimeoutMs":60000,
                    "decision":{"type":"SINGLE"}
                  },
                  "dispatcher":{
                    "type":"NON_BATCH",
                    "maxInflightRequestsPerPrefillWorker":128
                  },
                  "router":{
                    "availabilityHysteresisPercent":11,
                    "roles":{
                      "prefill":{
                        "availability":{"maxPendingRequests":321},
                        "executionTimeEstimator":{
                          "type":"FORMULA",
                          "expression":"sum(computeTokens)"
                        },
                        "selector":{"type":"RANDOM"}
                      },
                      "decode":{
                        "availability":{
                          "maxKvUsagePercent":87,
                          "maxEngineRequests":123
                        },
                        "kvReservation":{"maxOutputTokensForEstimate":456},
                        "selector":{"type":"RANDOM"}
                      },
                      "vit":{"selector":{"type":"RANDOM"}}
                    }
                  },
                  "workerRegistry":{
                    "health":{
                      "statusPollIntervalMs":25,
                      "statusRpcTimeoutMs":4000,
                      "statusStaleAfterMs":9000
                    },
                    "cacheStatus":{
                      "targetDiffSize":256,
                      "minRefreshIntervalMs":100,
                      "maxRefreshIntervalMs":2000
                    }
                  },
                  "observability":{
                    "cacheHit":{
                      "recentKeyWindow":{
                        "writeEnabled":false,
                        "durationMs":12345,
                        "maxKeyOccurrences":67890
                      },
                      "metricsEnabled":false,
                      "requestTraceLogEnabled":true
                    }
                  }
                }
                """);

        assertEquals(2, config.getSchemaVersion());
        assertTrue(config.isPriorityOrdering());
        assertTrue(config.isSingleDecision());
        assertEquals(500_000, config.queueScheduler().getCapacity()
                .getMaxOutstandingRequestsGlobal());
        assertEquals(1024, config.queueScheduler().getCapacity()
                .getMaxWaitingRequestsPerPrefillWorker());
        assertEquals(60_000L, config.queueScheduler().getQueueTimeoutMs());
        assertEquals(500_000, config.queueScheduler().getLifecycle()
                .getMaxDeliveredNotAcceptedRequestsGlobal());
        NonBatchDispatcherConfig dispatcher = assertInstanceOf(
                NonBatchDispatcherConfig.class, config.getDispatcher());
        assertEquals(128, dispatcher
                .getMaxInflightRequestsPerPrefillWorker());
        assertEquals(11L, config.getRouter().getAvailabilityHysteresisPercent());
        assertEquals(321L, config.getRouter().getRoles().getPrefill()
                .getAvailability().getMaxPendingRequests());
        FormulaEstimatorConfig estimator = assertInstanceOf(
                FormulaEstimatorConfig.class,
                config.getRouter().getRoles().getPrefill()
                        .getExecutionTimeEstimator());
        assertEquals("sum(computeTokens)", estimator.getExpression());
        assertEquals(123L, config.getRouter().getRoles().getDecode()
                .getAvailability().getMaxEngineRequests());
        assertEquals(456L, config.getRouter().getRoles().getDecode()
                .getKvReservation().getMaxOutputTokensForEstimate());
        assertEquals(25L, config.getWorkerRegistry().getHealth()
                .getStatusPollIntervalMs());
        assertEquals(256, config.getWorkerRegistry().getCacheStatus()
                .getTargetDiffSize());
        assertEquals(12_345L, config.getObservability().getCacheHit()
                .getRecentKeyWindow().getDurationMs());
        assertTrue(config.getObservability().getCacheHit()
                .isRequestTraceLogEnabled());
    }

    @Test
    void omitted_v1_non_batch_decision_becomes_explicit_single() {
        FlexlbConfig config = ConfigService.parse("""
                {
                  "schemaVersion":1,
                  "scheduler":{"type":"QUEUE","ordering":{"type":"FIFO"}},
                  "dispatcher":{"type":"NON_BATCH"}
                }
                """);

        assertTrue(config.isSingleDecision());
        assertInstanceOf(SingleDecisionConfig.class, config.decisionPolicy());
        assertEquals(1024, config.queueScheduler().getCapacity()
                .getMaxWaitingRequestsPerPrefillWorker());
    }

    @Test
    void active_v1_batch_fields_move_to_their_canonical_owners() {
        FlexlbConfig config = ConfigService.parse("""
                {
                  "schemaVersion":1,
                  "scheduler":{"type":"QUEUE","ordering":{"type":"FIFO"}},
                  "dispatcher":{
                    "type":"BATCH",
                    "maxRequests":17,
                    "maxCollectionWaitMs":0,
                    "maxWaitingRequestsPerPrefillWorker":55,
                    "maxInflightBatchesPerPrefillWorker":3,
                    "enqueueRpcTimeoutMs":4321
                  }
                }
                """);

        assertTrue(config.isFixedWindowDecision());
        assertEquals(17, config.fixedWindowDecision().getMaxRequests());
        assertEquals(0L, config.fixedWindowDecision().getMaxCollectionWaitMs());
        assertEquals(55, config.queueScheduler().getCapacity()
                .getMaxWaitingRequestsPerPrefillWorker());
        BatchDispatcherConfig dispatcher = assertInstanceOf(
                BatchDispatcherConfig.class, config.getDispatcher());
        assertEquals(3, dispatcher
                .getMaxInflightBatchesPerPrefillWorker());
        assertEquals(4321L, dispatcher.getEnqueueRpcTimeoutMs());
    }

    @Test
    void explicit_v1_owners_keep_precedence_over_valid_legacy_fields() {
        FlexlbConfig config = ConfigService.parse("""
                {
                  "schemaVersion":1,
                  "scheduler":{
                    "type":"QUEUE",
                    "ordering":{"type":"FIFO"},
                    "decision":{
                      "type":"FIXED_WINDOW",
                      "maxRequests":4,
                      "maxCollectionWaitMs":25
                    },
                    "capacity":{"maxWaitingRequestsPerPrefillWorker":64}
                  },
                  "dispatcher":{
                    "type":"BATCH",
                    "maxRequests":99,
                    "maxCollectionWaitMs":999,
                    "earlyDispatchPredictedExecutionMs":777,
                    "maxWaitingRequestsPerPrefillWorker":888
                  }
                }
                """);

        assertEquals(4, config.fixedWindowDecision().getMaxRequests());
        assertEquals(25L, config.fixedWindowDecision().getMaxCollectionWaitMs());
        assertEquals(64, config.queueScheduler().getCapacity()
                .getMaxWaitingRequestsPerPrefillWorker());
    }

    @Test
    void active_v1_prediction_trigger_is_rejected_because_mapping_is_lossy() {
        ConfigValidationException error = assertThrows(
                ConfigValidationException.class, () -> ConfigService.parse("""
                        {
                          "schemaVersion":1,
                          "scheduler":{"type":"QUEUE"},
                          "dispatcher":{
                            "type":"BATCH",
                            "earlyDispatchPredictedExecutionMs":100
                          }
                        }
                        """));

        assertTrue(error.getMessage().contains(
                "cannot be migrated without changing its equality-boundary"));

        assertThrows(ConfigValidationException.class, () -> ConfigService.parse("""
                {
                  "schemaVersion":1,
                  "scheduler":{
                    "type":"QUEUE",
                    "decision":{
                      "type":"FIXED_WINDOW",
                      "maxPredictedExecutionMs":100
                    }
                  },
                  "dispatcher":{"type":"BATCH"}
                }
                """));
        assertThrows(ConfigValidationException.class, () -> ConfigService.parse("""
                {
                  "schemaVersion":1,
                  "scheduler":{
                    "type":"QUEUE",
                    "decision":{
                      "type":"FIXED_WINDOW",
                      "maxPredictedExecutionMs":100
                    }
                  },
                  "dispatcher":{"type":"NON_BATCH"}
                }
                """));
    }

    @Test
    void invalid_ignored_v1_fields_are_rejected_before_removal() {
        assertInvalidIgnoredLegacyField("\"maxRequests\":0");
        assertInvalidIgnoredLegacyField("\"maxCollectionWaitMs\":-1");
        assertInvalidIgnoredLegacyField(
                "\"maxWaitingRequestsPerPrefillWorker\":0");
        assertInvalidIgnoredLegacyField(
                "\"earlyDispatchPredictedExecutionMs\":0");
        assertInvalidIgnoredLegacyField("\"maxRequests\":1.5");
        assertInvalidIgnoredLegacyField(
                "\"maxWaitingRequestsPerPrefillWorker\":2147483648");
    }

    @Test
    void non_batch_v1_rejects_every_batch_only_legacy_field() {
        assertInvalidNonBatchLegacyField("\"maxRequests\":8");
        assertInvalidNonBatchLegacyField("\"maxCollectionWaitMs\":300");
        assertInvalidNonBatchLegacyField(
                "\"earlyDispatchPredictedExecutionMs\":100");
        assertInvalidNonBatchLegacyField(
                "\"maxWaitingRequestsPerPrefillWorker\":1024");
    }

    @Test
    void strict_json_and_numeric_checks_run_before_v1_fields_are_removed() {
        assertThrows(ConfigValidationException.class, () -> ConfigService.parse("""
                {
                  "schemaVersion":1,
                  "scheduler":{"type":"QUEUE","decision":{"type":"SINGLE"}},
                  "dispatcher":{
                    "type":"BATCH",
                    "maxRequests":8,
                    "maxRequests":9
                  }
                }
                """));
        assertInvalidIgnoredLegacyField("\"maxRequests\":null");
        assertInvalidIgnoredLegacyField("\"maxRequests\":\"8\"");
        assertInvalidIgnoredLegacyField("\"maxCollectionWaitMs\":true");
        assertInvalidIgnoredLegacyField(
                "\"maxCollectionWaitMs\":9223372036854775808");
        assertInvalidIgnoredLegacyField(
                "\"earlyDispatchPredictedExecutionMs\":9223372036854775808");
    }

    @Test
    void explicit_v1_decision_remains_independent_from_dispatcher_type() {
        FlexlbConfig defaultBatch = ConfigService.parse("""
                {
                  "schemaVersion":1,
                  "scheduler":{
                    "type":"QUEUE",
                    "decision":{"type":"SINGLE"}
                  }
                }
                """);
        assertTrue(defaultBatch.isSingleDecision());
        assertInstanceOf(BatchDispatcherConfig.class, defaultBatch.getDispatcher());

        FlexlbConfig singleBatch = ConfigService.parse("""
                {
                  "schemaVersion":1,
                  "scheduler":{
                    "type":"QUEUE",
                    "decision":{"type":"SINGLE"}
                  },
                  "dispatcher":{
                    "type":"BATCH",
                    "maxRequests":8,
                    "maxCollectionWaitMs":300,
                    "earlyDispatchPredictedExecutionMs":100
                  }
                }
                """);
        assertTrue(singleBatch.isSingleDecision());
        assertInstanceOf(BatchDispatcherConfig.class, singleBatch.getDispatcher());

        FlexlbConfig fixedWindowNonBatch = ConfigService.parse("""
                {
                  "schemaVersion":1,
                  "scheduler":{
                    "type":"QUEUE",
                    "decision":{
                      "type":"FIXED_WINDOW",
                      "maxRequests":5,
                      "maxCollectionWaitMs":20
                    }
                  },
                  "dispatcher":{"type":"NON_BATCH"}
                }
                """);
        assertTrue(fixedWindowNonBatch.isFixedWindowDecision());
        assertEquals(5, fixedWindowNonBatch.fixedWindowDecision()
                .getMaxRequests());
        assertEquals(20L, fixedWindowNonBatch.fixedWindowDecision()
                .getMaxCollectionWaitMs());
        assertInstanceOf(
                NonBatchDispatcherConfig.class,
                fixedWindowNonBatch.getDispatcher());
    }

    @Test
    void v1_migration_preserves_unknown_and_inactive_fields_for_strict_binding() {
        assertThrows(ConfigValidationException.class, () -> ConfigService.parse("""
                {
                  "schemaVersion":1,
                  "scheduler":{"type":"QUEUE"},
                  "dispatcher":{
                    "type":"BATCH",
                    "maxInflightRequestsPerPrefillWorker":2
                  }
                }
                """));
        assertThrows(ConfigValidationException.class, () -> ConfigService.parse("""
                {
                  "schemaVersion":1,
                  "scheduler":{"type":"QUEUE"},
                  "dispatcher":{"type":"BATCH","unknownDispatcherField":2}
                }
                """));
    }

    @Test
    void v2_documents_never_receive_v1_compatibility_behavior() {
        assertThrows(ConfigValidationException.class, () -> ConfigService.parse("""
                {
                  "schemaVersion":2,
                  "scheduler":{"type":"QUEUE"},
                  "dispatcher":{"type":"BATCH","maxRequests":9}
                }
                """));

        FlexlbConfig nonBatch = ConfigService.parse("""
                {
                  "schemaVersion":2,
                  "scheduler":{"type":"QUEUE"},
                  "dispatcher":{"type":"NON_BATCH"}
                }
                """);
        assertTrue(nonBatch.isFixedWindowDecision(),
                "v2 omission must retain the canonical v2 default");
    }

    @Test
    void unsupported_versions_and_strict_json_failures_remain_rejected() {
        assertThrows(ConfigValidationException.class,
                () -> ConfigService.parse("{\"schemaVersion\":0}"));
        assertThrows(ConfigValidationException.class,
                () -> ConfigService.parse("{\"schemaVersion\":3}"));
        assertThrows(ConfigValidationException.class,
                () -> ConfigService.parse("{\"schemaVersion\":\"1\"}"));
        assertThrows(ConfigValidationException.class,
                () -> ConfigService.parse("{\"schemaVersion\":1.0}"));
        assertThrows(ConfigValidationException.class,
                () -> ConfigService.parse(
                        "{\"schemaVersion\":1,\"schemaVersion\":1}"));
        assertThrows(ConfigValidationException.class,
                () -> ConfigService.parse(
                        "{\"schemaVersion\":1,\"unknownV1Field\":true}"));
        assertThrows(ConfigValidationException.class,
                () -> ConfigService.parse("{\"schemaVersion\":1} {}"));
    }

    @Test
    void direct_non_batch_v1_only_updates_the_schema_version() {
        FlexlbConfig config = ConfigService.parse("""
                {
                  "schemaVersion":1,
                  "scheduler":{"type":"DIRECT"},
                  "dispatcher":{"type":"NON_BATCH"}
                }
                """);

        assertEquals(2, config.getSchemaVersion());
        assertTrue(config.isDirect());
        assertFalse(config.isQueue());
        assertInstanceOf(NonBatchDispatcherConfig.class, config.getDispatcher());
    }

    private static void assertInvalidIgnoredLegacyField(String field) {
        assertThrows(ConfigValidationException.class, () -> ConfigService.parse("""
                {
                  "schemaVersion":1,
                  "scheduler":{
                    "type":"QUEUE",
                    "decision":{"type":"SINGLE"},
                    "capacity":{"maxWaitingRequestsPerPrefillWorker":64}
                  },
                  "dispatcher":{"type":"BATCH",%s}
                }
                """.formatted(field)));
    }

    private static void assertInvalidNonBatchLegacyField(String field) {
        assertThrows(ConfigValidationException.class, () -> ConfigService.parse("""
                {
                  "schemaVersion":1,
                  "scheduler":{"type":"QUEUE"},
                  "dispatcher":{"type":"NON_BATCH",%s}
                }
                """.formatted(field)));
    }
}
