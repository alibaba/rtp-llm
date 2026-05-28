package org.flexlb.balance.dp;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BatchDeadlineEstimatorTest {

    @Test
    void normal_slo_computation() {
        long now = 1_000_000L;
        // seqLen=500, cacheMatched=200 → computeLen=300
        // estimatePrefillTimeMs(300, 0) = 190 + 0.0076*300 + 9e-9*300^2 ≈ 192
        // ttftEstimate = 192 + 10(avgQueue) = 202
        // slack = 500(slo) - 202 - 50(margin) = 248
        // interval = max(10, min(100, 248)) = 100
        long deadline = BatchDeadlineEstimator.computeDeadlineMicros(
                now, 500, 200, 10, 500, 50, 10, 100);
        assertEquals(now + 100 * 1000L, deadline);
    }

    @Test
    void slack_exceeds_max_interval_capped() {
        long now = 1_000_000L;
        // seqLen=100, cacheMatched=80 → computeLen=20
        // estimatePrefillTimeMs(20, 0) = 190 + 0.0076*20 + 9e-9*400 ≈ 190
        // ttftEstimate = 190 + 0 = 190
        // slack = 500 - 190 - 50 = 260
        // interval = max(10, min(100, 260)) = 100
        long deadline = BatchDeadlineEstimator.computeDeadlineMicros(
                now, 100, 80, 0, 500, 50, 10, 100);
        assertEquals(now + 100 * 1000L, deadline);
    }

    @Test
    void tight_slo_uses_min_interval() {
        long now = 1_000_000L;
        // seqLen=50000, cacheMatched=0 → computeLen=50000
        // estimatePrefillTimeMs(50000, 0) = 190 + 0.0076*50000 + 9e-9*50000^2 = 190+380+22.5 = 592
        // ttftEstimate = 592 + 100 = 692
        // slack = 500 - 692 - 50 = -242
        // interval = max(10, min(100, -242)) = 10
        long deadline = BatchDeadlineEstimator.computeDeadlineMicros(
                now, 50000, 0, 100, 500, 50, 10, 100);
        assertEquals(now + 10 * 1000L, deadline);
    }

    @Test
    void zero_seq_len() {
        long now = 5_000_000L;
        long deadline = BatchDeadlineEstimator.computeDeadlineMicros(
                now, 0, 0, 0, 500, 50, 10, 100);
        // computeLen=0, estimateMs=190, slack=500-190-50=260, interval=min(100,260)=100
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
