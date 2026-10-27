package org.flexlb.config;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import static org.junit.jupiter.api.Assertions.*;

class SloConfigTest {
    @Test void oneParameterStartsWithLearningAndFiniteBurstBudget() {
        var c = ConfigService.parse("{\"ttftSloMs\":3000}");
        assertEquals(3000, c.getTtftSloMs());
        assertTrue(c.isQueue());
        assertEquals(RoutingConfig.EstimatorType.LEARNING, c.getRouter().getRoles().getPrefill().getExecutionTimeEstimator().getType());
        assertNotNull(c.getRouter().getRoles().getPrefill().getTokenBalance());
        assertEquals(6000, c.queueScheduler().getQueueTimeoutMs());
        assertEquals(6000, c.resolveExpiresAtMs(0));
        assertEquals(3000, c.queueScheduler().getPrefillQueueShedding().getExpectedWaitThresholdMs());
        assertEquals(60000, c.getRequestLifecycle().getRequest().getTimeoutMs());
        assertEquals(20, c.getWorkerRegistry().getHealth().getStatusPollIntervalMs());
        assertEquals(200, c.getWorkerRegistry().getHealth().getStatusRpcTimeoutMs());
        assertEquals(500, c.getWorkerRegistry().getHealth().getStatusStaleAfterMs());
        assertEquals(200, c.getWorkerRegistry().getHealth().getCleanupIntervalMs());
        assertTrue(c.isFixedWindowDecision());
        assertEquals(20, c.decisionPolicy().getMaxCollectionWaitMs());
    }
    @Test void nonBatchNeedsOnlyTransportSelection() {
        var c = ConfigService.parse("{\"ttftSloMs\":100,\"dispatcher\":{\"type\":\"NON_BATCH\"}}");
        assertEquals(DecisionPolicyConfig.Type.SINGLE, c.decisionPolicy().getType());
        assertEquals(200, c.resolveExpiresAtMs(0));
    }
    @ParameterizedTest @ValueSource(strings={"0", "-1", "null", "1.5", "\"3000\"", "9223372036854775807"})
    void rejectsInvalidTargets(String value) {
        assertThrows(ConfigValidationException.class, () -> ConfigService.parse("{\"ttftSloMs\":"+value+"}"));
    }
    @ParameterizedTest @ValueSource(strings={"\"scheduler\":{}", "\"dispatcher\":{\"maxInflightPerPrefillWorker\":32}",
            "\"router\":{\"roles\":{\"prefill\":{\"cacheAffinity\":{}}}}",
            "\"router\":{\"roles\":{\"prefill\":{\"tokenBalance\":{\"maxPendingComputeTokens\":100}}}}"})
    void rejectsConflictingLegacyKnobs(String field) {
        assertThrows(ConfigValidationException.class, () -> ConfigService.parse("{\"ttftSloMs\":3000,"+field+"}"));
    }
    @Test void subMillisecondCollectionAndOverflowBoundaryRemainFinite() {
        var c=ConfigService.parse("{\"ttftSloMs\":1}");
        assertEquals(0, c.decisionPolicy().getMaxCollectionWaitMs());
        assertEquals(1, c.decisionPolicy().getMaxPredictedExecutionMs());
        long largest=Long.MAX_VALUE / 4;
        c=ConfigService.parse("{\"ttftSloMs\":"+largest+"}");
        assertEquals(largest*2, c.queueScheduler().getQueueTimeoutMs());
        assertEquals(largest*4, c.getRequestLifecycle().getRequest().getTimeoutMs());
    }
}
