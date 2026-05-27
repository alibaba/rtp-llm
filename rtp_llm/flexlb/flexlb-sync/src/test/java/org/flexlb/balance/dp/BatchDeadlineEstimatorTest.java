package org.flexlb.balance.dp;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BatchDeadlineEstimatorTest {

    @Test
    void normal_slo_computation() {
        long now = 1_000_000L;
        // seqLen=500, cacheMatched=200 → computeLen=300
        // estimatePrefillTimeMs(300, 0) = 108
        // ttftEstimate = 108 + 10(avgQueue) = 118
        // slack = 500(slo) - 118 - 50(margin) = 332
        // interval = max(10, min(100, 332)) = 100
        long deadline = BatchDeadlineEstimator.computeDeadlineMicros(
                now, 500, 200, 10, 500, 50, 10, 100);
        assertEquals(now + 100 * 1000L, deadline);
    }

    @Test
    void slack_exceeds_max_interval_capped() {
        long now = 1_000_000L;
        // seqLen=100, cacheMatched=80 → computeLen=20
        // estimatePrefillTimeMs(20, 0) = 71
        // ttftEstimate = 71 + 0 = 71
        // slack = 500 - 71 - 50 = 379
        // interval = max(10, min(100, 379)) = 100
        long deadline = BatchDeadlineEstimator.computeDeadlineMicros(
                now, 100, 80, 0, 500, 50, 10, 100);
        assertEquals(now + 100 * 1000L, deadline);
    }

    @Test
    void tight_slo_uses_min_interval() {
        long now = 1_000_000L;
        // seqLen=3000, cacheMatched=0 → computeLen=3000
        // estimatePrefillTimeMs(3000, 0) = 463
        // ttftEstimate = 463 + 100 = 563
        // slack = 500 - 563 - 50 = -113
        // interval = max(10, min(100, -113)) = 10
        long deadline = BatchDeadlineEstimator.computeDeadlineMicros(
                now, 3000, 0, 100, 500, 50, 10, 100);
        assertEquals(now + 10 * 1000L, deadline);
    }

    @Test
    void zero_seq_len() {
        long now = 5_000_000L;
        long deadline = BatchDeadlineEstimator.computeDeadlineMicros(
                now, 0, 0, 0, 500, 50, 10, 100);
        // computeLen=0, estimateMs=69, slack=500-69-50=381, interval=min(100,381)=100
        assertEquals(now + 100 * 1000L, deadline);
    }

    @Test
    void cache_matched_exceeds_seq_len_clamped() {
        long now = 1_000_000L;
        long deadline = BatchDeadlineEstimator.computeDeadlineMicros(
                now, 100, 200, 0, 500, 50, 10, 100);
        // computeLen = max(0, 100-200) = 0
        assertTrue(deadline >= now);
    }
}
