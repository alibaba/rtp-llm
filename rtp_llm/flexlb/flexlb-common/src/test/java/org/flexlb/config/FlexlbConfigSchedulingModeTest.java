package org.flexlb.config;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
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
        assertFalse(config.isBatchDispatch());
        assertInstanceOf(NonBatchDispatcherConfig.class, config.getDispatcher());
    }

    @Test
    void queue_axes_are_independent() {
        FlexlbConfig fifoBatch = parseQueue("FIFO", "BATCH");
        assertTrue(fifoBatch.isQueue());
        assertFalse(fifoBatch.isPriorityOrdering());
        assertTrue(fifoBatch.isBatchDispatch());
        assertEquals(3_600_000L, fifoBatch.queueScheduler().getQueueTimeoutMs());

        FlexlbConfig fifoNonBatch = parseQueue("FIFO", "NON_BATCH");
        assertFalse(fifoNonBatch.isPriorityOrdering());
        assertFalse(fifoNonBatch.isBatchDispatch());
        fifoNonBatch = ConfigService.parse("""
                {
                  "scheduler":{"type":"QUEUE","ordering":{"type":"FIFO"}},
                  "dispatcher":{"type":"NON_BATCH","maxInflightRequestsPerPrefillWorker":1}
                }
                """);
        assertEquals(1, fifoNonBatch.nonBatchDispatcher()
                .getMaxInflightRequestsPerPrefillWorker().intValue());

        FlexlbConfig priorityBatch = parseQueue("PRIORITY", "BATCH");
        assertTrue(priorityBatch.isPriorityOrdering());
        assertTrue(priorityBatch.isBatchDispatch());

        FlexlbConfig priorityNonBatch = parseQueue("PRIORITY", "NON_BATCH");
        assertTrue(priorityNonBatch.isPriorityOrdering());
        assertFalse(priorityNonBatch.isBatchDispatch());
    }

    @Test
    void tagged_unions_reject_parameters_from_inactive_variants() {
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
                  "scheduler":{"type":"QUEUE","queueTimeoutMs":0,
                    "ordering":{"type":"FIFO"}},
                  "dispatcher":{"type":"NON_BATCH"}
                }
                """));
        assertThrows(ConfigValidationException.class, () -> ConfigService.parse("""
                {
                  "scheduler":{"type":"DIRECT","queueTimeoutMs":1000},
                  "dispatcher":{"type":"NON_BATCH"}
                }
                """));
    }

    private static FlexlbConfig parseQueue(String ordering, String dispatcher) {
        return ConfigService.parse("""
                {
                  "scheduler":{"type":"QUEUE","ordering":{"type":"%s"}},
                  "dispatcher":{"type":"%s"}
                }
                """.formatted(ordering, dispatcher));
    }
}
