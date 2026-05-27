package org.flexlb.balance.dp;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BatchDeadlineEstimatorTest {

    @Test
    void normal_slo_computation() {
        long now = 1_000_000L;
        // seqLen=500, cacheMatched=200 → computeLen=300
        // estimatePrefillTimeMs(300, 0) = 4
        // ttftEstimate = 4 + 10(avgQueue) = 14
        // slack = 500(slo) - 14 - 50(margin) = 436
        // interval = max(10, min(100, 436)) = 100
        long deadline = BatchDeadlineEstimator.computeDeadlineMicros(
                now, 500, 200, 10, 500, 50, 10, 100);
        assertEquals(now + 100 * 1000L, deadline);
    }

    @Test
    void slack_exceeds_max_interval_capped() {
        long now = 1_000_000L;
        // seqLen=100, cacheMatched=80 → computeLen=20
        // estimatePrefillTimeMs(20, 0) = 4
        // ttftEstimate = 4 + 0 = 4
        // slack = 500 - 4 - 50 = 446
        // interval = max(10, min(100, 446)) = 100
        long deadline = BatchDeadlineEstimator.computeDeadlineMicros(
                now, 100, 80, 0, 500, 50, 10, 100);
        assertEquals(now + 100 * 1000L, deadline);
    }

    @Test
    void tight_slo_uses_min_interval() {
        long now = 1_000_000L;
        // seqLen=200000, cacheMatched=0 → computeLen=200000
        // estimatePrefillTimeMs(200000, 0) = 664
        // ttftEstimate = 664 + 100 = 764
        // slack = 500 - 764 - 50 = -314
        // interval = max(10, min(100, -314)) = 10
        long deadline = BatchDeadlineEstimator.computeDeadlineMicros(
                now, 200000, 0, 100, 500, 50, 10, 100);
        assertEquals(now + 10 * 1000L, deadline);
    }

    @Test
    void zero_seq_len() {
        long now = 5_000_000L;
        long deadline = BatchDeadlineEstimator.computeDeadlineMicros(
                now, 0, 0, 0, 500, 50, 10, 100);
        // computeLen=0, estimateMs=4, slack=500-4-50=446, interval=min(100,446)=100
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
