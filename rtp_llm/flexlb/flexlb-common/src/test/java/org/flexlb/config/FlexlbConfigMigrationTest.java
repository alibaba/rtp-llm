package org.flexlb.config;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FlexlbConfigMigrationTest {
    @Test
    void nonBatchExplicitlyPreservesImmediateDecision() throws Exception {
        var result = FlexlbConfigMigration.fromV1("""
                {"schemaVersion":1,"dispatcher":{"type":"NON_BATCH"}}
                """);
        var config = ConfigService.parse(result.document());
        assertEquals(DecisionPolicyConfig.Type.SINGLE, config.queueScheduler().getDecision().getType());
        assertEquals(64, config.getDispatcher().getMaxInflightRequestsPerPrefillWorker());
        assertFalse(result.behaviorChanges().isEmpty());
    }

    @Test
    void batchFieldsMoveToTheirActualOwners() throws Exception {
        var result = FlexlbConfigMigration.fromV1("""
                {"dispatcher":{"type":"BATCH","maxRequests":4,"maxCollectionWaitMs":12,
                "maxWaitingRequestsPerPrefillWorker":80,"earlyDispatchPredictedExecutionMs":40}}
                """);
        var config = ConfigService.parse(result.document());
        assertEquals(4, config.queueScheduler().getDecision().getMaxRequests());
        assertEquals(12L, config.queueScheduler().getDecision().getMaxCollectionWaitMs());
        assertEquals(40L, config.queueScheduler().getDecision().getMaxPredictedExecutionMs());
        assertEquals(80, config.queueScheduler().getCapacity().getMaxWaitingRequestsPerPrefillWorker());
        assertTrue(result.behaviorChanges().stream().anyMatch(note -> note.contains("not equivalent")));
    }

    @Test
    void directUsesTheSameNonBatchRequestCapacity() throws Exception {
        var result = FlexlbConfigMigration.fromV1("""
                {"scheduler":{"type":"DIRECT"},"dispatcher":{"type":"NON_BATCH"},
                "router":{"roles":{"prefill":{"availability":{"maxPendingRequests":7}}}}}
                """);
        var config = ConfigService.parse(result.document());
        assertTrue(config.isDirect());
        assertEquals(7, config.getDispatcher().getMaxInflightRequestsPerPrefillWorker());
    }

    @Test
    void removedRandomSelectorsAreNeverSilentlyReplaced() {
        for (String role : new String[]{"prefill", "decode"}) {
            var error = assertThrows(ConfigValidationException.class, () -> FlexlbConfigMigration.fromV1(
                    "{\"router\":{\"roles\":{\"" + role + "\":{\"selector\":{\"type\":\"RANDOM\"}}}}}"));
            assertTrue(error.getMessage().contains("no equivalent"));
        }
    }

    @Test
    void unknownFieldsCannotDisappearDuringMigration() {
        assertThrows(ConfigValidationException.class, () -> FlexlbConfigMigration.fromV1("""
                {"router":{"roles":{"prefill":{"selector":{"type":"ESTIMATED_TTFT","unknown":1}}}}}
                """));
        assertThrows(ConfigValidationException.class, () -> FlexlbConfigMigration.fromV1("""
                {"unknown":1}
                """));
    }

    @Test
    void onlineLoaderStillRejectsV1() {
        assertThrows(ConfigValidationException.class, () -> ConfigService.parse("{\"schemaVersion\":1}"));
    }

    @Test
    void malformedNumbersAndConflictingOwnersAreRejected() {
        for (String document : new String[]{
                "{\"schemaVersion\":1.5}",
                "{\"router\":{\"availabilityHysteresisPercent\":\"15\"}}",
                "{\"router\":{\"roles\":{\"prefill\":{\"availability\":{\"maxPendingRequests\":1.5}}}}}",
                """
                {"router":{"roles":{"prefill":{"candidateChoice":{},
                "selector":{"type":"ESTIMATED_TTFT","candidateChoice":{}}}}}}
                """,
                """
                {"router":{"roles":{"decode":{"decayPerToken":0.1,
                "selector":{"type":"KV_USAGE_WEIGHTED_RANDOM","decayPerToken":0.2}}}}}
                """}) {
            assertThrows(ConfigValidationException.class, () -> FlexlbConfigMigration.fromV1(document), document);
        }
    }
}
