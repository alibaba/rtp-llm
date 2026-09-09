package org.flexlb.config;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FlexlbConfigSchedulingModeTest {

    @Test
    void direct_is_explicit_and_only_supports_non_batch_delivery() {
        FlexlbConfig config = ConfigTestFixtures.parse("""
                {
                  "scheduler":{"type":"DIRECT"},
                  "dispatcher":{"type":"NON_BATCH"}
                }
                """);

        assertTrue(config.isDirect());
        assertFalse(config.isQueue());
        assertFalse(config.isPriorityOrdering());
        assertFalse(config.isSingleDecision());
        assertFalse(config.isFixedWindowDecision());
        assertEquals(DispatcherConfig.Type.NON_BATCH,
                config.getDispatcher().getType());
    }

    @Test
    void fifo_queue_supports_all_decision_by_dispatcher_combinations() {
        assertMode(parseQueue("FIFO", "SINGLE", "NON_BATCH"), false, true, false);
        assertMode(parseQueue("FIFO", "SINGLE", "BATCH"), false, true, true);
        assertMode(parseQueue("FIFO", "FIXED_WINDOW", "NON_BATCH"),
                false, false, false);
        assertMode(parseQueue("FIFO", "FIXED_WINDOW", "BATCH"),
                false, false, true);
    }

    @Test
    void ordering_is_independent_from_decision_and_dispatcher() {
        FlexlbConfig prioritySingleBatch = parseQueue("PRIORITY", "SINGLE", "BATCH");
        assertMode(prioritySingleBatch, true, true, true);

        FlexlbConfig priorityWindowNonBatch = parseQueue(
                "PRIORITY", "FIXED_WINDOW", "NON_BATCH");
        assertMode(priorityWindowNonBatch, true, false, false);
    }

    @Test
    void decision_and_capacity_have_single_configuration_owners() {
        FlexlbConfig config = ConfigTestFixtures.parse("""
                {
                  "scheduler":{
                    "type":"QUEUE",
                    "ordering":{"type":"FIFO"},
                    "decision":{
                      "type":"FIXED_WINDOW",
                      "maxRequests":4,
                      "maxCollectionWaitMs":25,
                      "maxPredictedExecutionMs":80
                    }
                  },
                  "dispatcher":{
                    "type":"BATCH",
                    "maxInflightPerPrefillWorker":2
                  }
                }
                """);

        DecisionPolicyConfig decision = config.fixedWindowDecision();
        assertEquals(4, decision.getMaxRequests());
        assertEquals(25L, decision.getMaxCollectionWaitMs());
        assertEquals(80L, decision.getMaxPredictedExecutionMs().longValue());
    }

    @Test
    void omitted_queue_fields_use_canonical_scheduler_defaults() {
        FlexlbConfig batch = ConfigTestFixtures.parse("""
                {
                  "scheduler":{"type":"QUEUE","ordering":{"type":"FIFO"}},
                  "dispatcher":{"type":"BATCH"}
                }
                """);
        assertTrue(batch.isFixedWindowDecision());
        assertEquals(8, batch.fixedWindowDecision().getMaxRequests());
        assertEquals(300L, batch.fixedWindowDecision().getMaxCollectionWaitMs());
        assertNull(batch.fixedWindowDecision().getMaxPredictedExecutionMs());
        assertEquals(2, batch.getDispatcher().getMaxInflightPerPrefillWorker());

        FlexlbConfig nonBatchDefault = ConfigTestFixtures.parse("""
                {
                  "scheduler":{"type":"QUEUE","ordering":{"type":"FIFO"}},
                  "dispatcher":{"type":"NON_BATCH"}
                }
                """);
        assertTrue(nonBatchDefault.isFixedWindowDecision(),
                "dispatcher type must not choose the decision policy");

        FlexlbConfig nonBatchSingle = ConfigTestFixtures.parse("""
                {
                  "scheduler":{"type":"QUEUE","ordering":{"type":"FIFO"},
                    "decision":{"type":"SINGLE"}},
                  "dispatcher":{"type":"NON_BATCH"}
                }
                """);
        assertTrue(nonBatchSingle.isSingleDecision());
        assertEquals(DecisionPolicyConfig.Type.SINGLE,
                nonBatchSingle.decisionPolicy().getType());
        DispatcherConfig dispatcher = nonBatchSingle.getDispatcher();
        assertEquals(2, dispatcher.getMaxInflightPerPrefillWorker());
    }

    @Test
    void fixed_window_group_size_has_no_configured_upper_bound() {
        FlexlbConfig config = ConfigTestFixtures.parse("""
                {
                  "scheduler":{"type":"QUEUE","ordering":{"type":"FIFO"},
                    "decision":{"type":"FIXED_WINDOW","maxRequests":4096}},
                  "dispatcher":{"type":"BATCH"}
                }
                """);

        assertEquals(4096, config.fixedWindowDecision().getMaxRequests());
        assertEquals(4096, config.fixedWindowDecision().resolveMaxRequests());
    }

    @Test
    void tagged_unions_reject_parameters_from_inactive_variants() {
        assertThrows(ConfigValidationException.class, () -> ConfigTestFixtures.parse("""
                {
                  "scheduler":{"type":"QUEUE","ordering":{"type":"FIFO"},
                    "decision":{"type":"SINGLE","maxRequests":2}},
                  "dispatcher":{"type":"NON_BATCH"}
                }
                """));
        assertThrows(ConfigValidationException.class, () -> ConfigTestFixtures.parse("""
                {
                  "scheduler":{"type":"QUEUE","ordering":{"type":"FIFO"},
                    "decision":{"type":"FIXED_WINDOW","maxRequests":0}},
                  "dispatcher":{"type":"NON_BATCH"}
                }
                """));
        assertThrows(ConfigValidationException.class, () -> ConfigTestFixtures.parse("""
                {
                  "scheduler":{"type":"QUEUE","ordering":{"type":"FIFO"},
                    "decision":{"type":"FIXED_WINDOW","maxPredictedExecutionMs":0}},
                  "dispatcher":{"type":"BATCH"}
                }
                """));
        assertThrows(ConfigValidationException.class, () -> ConfigTestFixtures.parse("""
                {
                  "scheduler":{"type":"QUEUE","ordering":{"type":"FIFO"},
                    "capacity":{"maxWaitingRequestsPerPrefillWorker":0}},
                  "dispatcher":{"type":"NON_BATCH"}
                }
                """));
        assertThrows(ConfigValidationException.class, () -> ConfigTestFixtures.parse("""
                {
                  "scheduler":{"type":"QUEUE","ordering":{"type":"FIFO","defaultPriority":50}},
                  "dispatcher":{"type":"NON_BATCH"}
                }
                """));
        assertThrows(ConfigValidationException.class, () -> ConfigTestFixtures.parse("""
                {
                  "scheduler":{"type":"QUEUE","ordering":{"type":"PRIORITY"}},
                  "dispatcher":{"type":"BATCH","maxInflightRequestsPerPrefillWorker":1}
                }
                """));
        assertThrows(ConfigValidationException.class, () -> ConfigTestFixtures.parse("""
                {
                  "scheduler":{"type":"QUEUE","ordering":{"type":"FIFO"}},
                  "dispatcher":{"type":"BATCH","maxRequests":8}
                }
                """));
        assertThrows(ConfigValidationException.class, () -> ConfigTestFixtures.parse("""
                {
                  "scheduler":{"type":"QUEUE","queueTimeoutMs":0,
                    "ordering":{"type":"FIFO"}},
                  "dispatcher":{"type":"NON_BATCH"}
                }
                """));
        assertThrows(ConfigValidationException.class, () -> ConfigTestFixtures.parse("""
                {
                  "scheduler":{"type":"DIRECT","decision":{"type":"SINGLE"}},
                  "dispatcher":{"type":"NON_BATCH"}
                }
                """));
    }

    @Test
    void all_delivery_modes_share_the_same_default_and_explicit_limit() {
        for (String mode : new String[]{"QUEUE/BATCH", "QUEUE/NON_BATCH", "DIRECT/NON_BATCH"}) {
            String[] parts = mode.split("/");
            FlexlbConfig config = parse(parts[0], parts[1], "");
            assertEquals(2, config.getDispatcher().getMaxInflightPerPrefillWorker(), mode);
            for (int limit : new int[]{1, 7, Integer.MAX_VALUE}) {
                config = parse(parts[0], parts[1], ",\"maxInflightPerPrefillWorker\":" + limit);
                assertEquals(limit, config.getDispatcher().getMaxInflightPerPrefillWorker(), mode);
            }
        }
    }

    @Test
    void limits_require_positive_integers_without_json_coercion() {
        for (String type : new String[]{"BATCH", "NON_BATCH"}) {
            for (String value : new String[]{"0", "-1", "1.5", "\"2\"", "true", "null", "[]", "{}", "2147483648"}) {
                assertThrows(ConfigValidationException.class,
                        () -> parse("QUEUE", type, ",\"maxInflightPerPrefillWorker\":" + value), value);
            }
            FlexlbConfig config = parse("QUEUE", type, "");
            config.getDispatcher().setMaxInflightPerPrefillWorker(0);
            assertThrows(ConfigValidationException.class, () -> FlexlbConfigValidator.validate(config));
        }
    }

    @Test
    void removed_concurrency_fields_and_direct_queue_settings_are_rejected() {
        for (String type : new String[]{"BATCH", "NON_BATCH"}) {
            for (String field : new String[]{"maxInflightBatchesPerPrefillWorker", "inflightMultiplier", "inflightRequestMultiplier"}) {
                assertThrows(ConfigValidationException.class,
                        () -> parse("QUEUE", type, ",\"" + field + "\":2"), field);
            }
        }
        assertThrows(ConfigValidationException.class, () -> ConfigTestFixtures.parse("""
                {"scheduler":{"type":"DIRECT","queueTimeoutMs":1000},
                 "dispatcher":{"type":"NON_BATCH"}}
                """));
    }

    private static FlexlbConfig parseQueue(
            String ordering, String decision, String dispatcher) {
        return ConfigTestFixtures.parse("""
                {
                  "scheduler":{
                    "type":"QUEUE",
                    "ordering":{"type":"%s"},
                    "decision":{"type":"%s"}
                  },
                  "dispatcher":{"type":"%s"}
                }
                """.formatted(ordering, decision, dispatcher));
    }

    private static void assertMode(FlexlbConfig config,
                                   boolean priority,
                                   boolean single,
                                   boolean batchDispatch) {
        assertTrue(config.isQueue());
        assertEquals(priority, config.isPriorityOrdering());
        assertEquals(single, config.isSingleDecision());
        assertEquals(!single, config.isFixedWindowDecision());
        assertEquals(batchDispatch ? DispatcherConfig.Type.BATCH
                        : DispatcherConfig.Type.NON_BATCH,
                config.getDispatcher().getType());
        assertEquals(3_600_000L, config.queueScheduler().getQueueTimeoutMs());
    }

    private static FlexlbConfig parse(String scheduler, String dispatcher, String fields) {
        return ConfigTestFixtures.parse("""
                {"scheduler":{"type":"%s"},"dispatcher":{"type":"%s"%s}}
                """.formatted(scheduler, dispatcher, fields));
    }
}
