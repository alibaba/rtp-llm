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
        FlexlbConfig config = ConfigService.parse("""
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
    void only_preemption_requires_decode_capacity_during_placement() {
        FlexlbConfig fifo = parseQueue("FIFO", "SINGLE", "BATCH");
        FlexlbConfig priority = parseQueue("PRIORITY", "SINGLE", "BATCH");
        FlexlbConfig preemptive = ConfigService.parse("""
                {
                  "scheduler":{"type":"QUEUE",
                    "ordering":{"type":"PRIORITY","preemption":{
                      "allowedVictimStages":["PREFILL_QUEUED"]}},
                    "decision":{"type":"SINGLE"}},
                  "dispatcher":{"type":"BATCH"}
                }
                """);
        FlexlbConfig direct = ConfigService.parse("""
                {
                  "scheduler":{"type":"DIRECT"},
                  "dispatcher":{"type":"NON_BATCH"}
                }
                """);

        assertTrue(fifo.defersDecodeCapacityUntilDispatch());
        assertTrue(priority.defersDecodeCapacityUntilDispatch());
        assertFalse(preemptive.defersDecodeCapacityUntilDispatch());
        assertFalse(direct.defersDecodeCapacityUntilDispatch());
    }

    @Test
    void decision_and_capacity_have_single_configuration_owners() {
        FlexlbConfig config = ConfigService.parse("""
                {
                  "scheduler":{
                    "type":"QUEUE",
                    "ordering":{"type":"FIFO"},
                    "decision":{
                      "type":"FIXED_WINDOW",
                      "maxRequests":4,
                      "maxCollectionWaitMs":25,
                      "maxPredictedExecutionMs":80
                    },
                    "capacity":{
                      "maxOutstandingRequestsGlobal":1000,
                      "maxWaitingRequestsPerPrefillWorker":64
                    }
                  },
                  "dispatcher":{
                    "type":"BATCH",
                    "maxInflightBatchesPerPrefillWorker":2
                  }
                }
                """);

        DecisionPolicyConfig decision = config.fixedWindowDecision();
        assertEquals(4, decision.getMaxRequests());
        assertEquals(25L, decision.getMaxCollectionWaitMs());
        assertEquals(80L, decision.getMaxPredictedExecutionMs().longValue());
        assertEquals(64, config.queueScheduler().getCapacity()
                .getMaxWaitingRequestsPerPrefillWorker());
    }

    @Test
    void omitted_queue_fields_use_canonical_scheduler_defaults() {
        FlexlbConfig batch = ConfigService.parse("""
                {
                  "scheduler":{"type":"QUEUE","ordering":{"type":"FIFO"}},
                  "dispatcher":{"type":"BATCH"}
                }
                """);
        assertTrue(batch.isFixedWindowDecision());
        assertEquals(8, batch.fixedWindowDecision().getMaxRequests());
        assertEquals(300L, batch.fixedWindowDecision().getMaxCollectionWaitMs());
        assertNull(batch.fixedWindowDecision().getMaxPredictedExecutionMs());
        assertEquals(1024, batch.queueScheduler().getCapacity()
                .getMaxWaitingRequestsPerPrefillWorker());

        FlexlbConfig nonBatchDefault = ConfigService.parse("""
                {
                  "scheduler":{"type":"QUEUE","ordering":{"type":"FIFO"}},
                  "dispatcher":{"type":"NON_BATCH"}
                }
                """);
        assertTrue(nonBatchDefault.isFixedWindowDecision(),
                "dispatcher type must not choose the decision policy");

        FlexlbConfig nonBatchSingle = ConfigService.parse("""
                {
                  "scheduler":{"type":"QUEUE","ordering":{"type":"FIFO"},
                    "decision":{"type":"SINGLE"}},
                  "dispatcher":{"type":"NON_BATCH","maxInflightRequestsPerPrefillWorker":1}
                }
                """);
        assertTrue(nonBatchSingle.isSingleDecision());
        assertEquals(DecisionPolicyConfig.Type.SINGLE,
                nonBatchSingle.decisionPolicy().getType());
        DispatcherConfig dispatcher = nonBatchSingle.getDispatcher();
        assertEquals(1, dispatcher
                .getMaxInflightRequestsPerPrefillWorker().intValue());
    }

    @Test
    void fixed_window_accepts_the_documented_maximum_group_size() {
        FlexlbConfig config = ConfigService.parse("""
                {
                  "scheduler":{"type":"QUEUE","ordering":{"type":"FIFO"},
                    "decision":{"type":"FIXED_WINDOW","maxRequests":1024}},
                  "dispatcher":{"type":"BATCH"}
                }
                """);

        assertEquals(DecisionPolicyConfig.MAX_REQUESTS,
                config.fixedWindowDecision().getMaxRequests());
    }

    @Test
    void tagged_unions_reject_parameters_from_inactive_variants() {
        assertThrows(ConfigValidationException.class, () -> ConfigService.parse("""
                {
                  "scheduler":{"type":"QUEUE","ordering":{"type":"FIFO"},
                    "decision":{"type":"SINGLE","maxRequests":2}},
                  "dispatcher":{"type":"NON_BATCH"}
                }
                """));
        assertThrows(ConfigValidationException.class, () -> ConfigService.parse("""
                {
                  "scheduler":{"type":"QUEUE","ordering":{"type":"FIFO"},
                    "decision":{"type":"FIXED_WINDOW","maxRequests":0}},
                  "dispatcher":{"type":"NON_BATCH"}
                }
                """));
        assertThrows(ConfigValidationException.class, () -> ConfigService.parse("""
                {
                  "scheduler":{"type":"QUEUE","ordering":{"type":"FIFO"},
                    "decision":{"type":"FIXED_WINDOW","maxRequests":1025}},
                  "dispatcher":{"type":"NON_BATCH"}
                }
                """));
        assertThrows(ConfigValidationException.class, () -> ConfigService.parse("""
                {
                  "scheduler":{"type":"QUEUE","ordering":{"type":"FIFO"},
                    "decision":{"type":"FIXED_WINDOW","maxPredictedExecutionMs":0}},
                  "dispatcher":{"type":"BATCH"}
                }
                """));
        assertThrows(ConfigValidationException.class, () -> ConfigService.parse("""
                {
                  "scheduler":{"type":"QUEUE","ordering":{"type":"FIFO"},
                    "capacity":{"maxWaitingRequestsPerPrefillWorker":0}},
                  "dispatcher":{"type":"NON_BATCH"}
                }
                """));
        assertThrows(ConfigValidationException.class, () -> ConfigService.parse("""
                {
                  "scheduler":{"type":"QUEUE","ordering":{"type":"FIFO","defaultPriority":50}},
                  "dispatcher":{"type":"NON_BATCH"}
                }
                """));
        assertThrows(ConfigValidationException.class, () -> ConfigService.parse("""
                {
                  "scheduler":{"type":"QUEUE","ordering":{"type":"PRIORITY"}},
                  "dispatcher":{"type":"BATCH","maxInflightRequestsPerPrefillWorker":1}
                }
                """));
        assertThrows(ConfigValidationException.class, () -> ConfigService.parse("""
                {
                  "scheduler":{"type":"QUEUE","ordering":{"type":"FIFO"}},
                  "dispatcher":{"type":"BATCH","maxRequests":8}
                }
                """));
        assertThrows(ConfigValidationException.class, () -> ConfigService.parse("""
                {
                  "scheduler":{"type":"QUEUE","queueTimeoutMs":0,
                    "ordering":{"type":"FIFO"}},
                  "dispatcher":{"type":"NON_BATCH"}
                }
                """));
        assertThrows(ConfigValidationException.class, () -> ConfigService.parse("""
                {
                  "scheduler":{"type":"DIRECT","decision":{"type":"SINGLE"}},
                  "dispatcher":{"type":"NON_BATCH"}
                }
                """));
    }

    private static FlexlbConfig parseQueue(
            String ordering, String decision, String dispatcher) {
        return ConfigService.parse("""
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
}
