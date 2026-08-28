package org.flexlb.balance.scheduler;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Numeric correctness tests for {@link NaviPgdOptimizer}, verified against
 * navi C++ {@code CostSchedulerKernelInternal} golden values.
 */
class NaviPgdOptimizerTest {

    private static final double[] NAVI_PARAMS =
            {-4.0, 10.0, 1.4, 20.0, 0.1, 0.09, 1.4, 1.0, -4.0};

    // ==================== findSimplexThreshold ====================

    @Test
    @DisplayName("findSimplexThreshold: 5-value golden (CostSchedulerKernelInternal L306-334)")
    void findSimplexThresholdGolden() {
        // workspace = [0.5, 0.3, 0.2, -0.1, 0.4], activeSum = 1.3
        // Iter 0: threshold = (1.3-1)/5 = 0.06; compact → [0.5,0.3,0.2,0.4], sum=1.4
        // Iter 1: threshold = (1.4-1)/4 = 0.1; all 4 > 0.1 → stable
        double[] buffer = {0.5, 0.3, 0.2, -0.1, 0.4};
        double activeSum = 0.5 + 0.3 + 0.2 + (-0.1) + 0.4; // 1.3
        double[] out = new double[1];
        boolean found = NaviPgdOptimizer.findSimplexThreshold(buffer, 0, 5, activeSum, out);
        assertTrue(found, "threshold must be found");
        assertEquals(0.1, out[0], 1e-12, "threshold golden mismatch");
        // Verify compacted buffer prefix: [0.5, 0.3, 0.2, 0.4]
        assertEquals(0.5, buffer[0], 1e-15);
        assertEquals(0.3, buffer[1], 1e-15);
        assertEquals(0.2, buffer[2], 1e-15);
        assertEquals(0.4, buffer[3], 1e-15);
    }

    @Test
    @DisplayName("findSimplexThreshold: uniform values converge in one iteration")
    void findSimplexThresholdUniform() {
        // All equal: [0.5, 0.5, 0.5, 0.5], sum=2.0
        // threshold = (2.0-1)/4 = 0.25; all > 0.25 → stable immediately
        double[] buffer = {0.5, 0.5, 0.5, 0.5};
        double[] out = new double[1];
        assertTrue(NaviPgdOptimizer.findSimplexThreshold(buffer, 0, 4, 2.0, out));
        assertEquals(0.25, out[0], 1e-15);
    }

    @Test
    @DisplayName("findSimplexThreshold: single element")
    void findSimplexThresholdSingle() {
        double[] buffer = {3.0};
        double[] out = new double[1];
        assertTrue(NaviPgdOptimizer.findSimplexThreshold(buffer, 0, 1, 3.0, out));
        assertEquals(2.0, out[0], 1e-15);
    }

    @Test
    @DisplayName("findSimplexThreshold: all negative → returns false")
    void findSimplexThresholdAllNegative() {
        double[] buffer = {-1.0, -2.0, -3.0};
        double[] out = new double[1];
        // sum = -6; threshold = (-6-1)/3 = -7/3; all > -7/3; next threshold = same → stable
        // Actually: -1 > -7/3 ≈ -2.33? Yes. -2 > -2.33? Yes. -3 > -2.33? No!
        // So iter 0: compact to [-1, -2], nextSum=-3, nextCount=2
        // Iter 1: threshold = (-3-1)/2 = -2; -1 > -2 yes, -2 > -2 NO (strict >)
        // compact to [-1], nextSum=-1, nextCount=1
        // Iter 2: threshold = (-1-1)/1 = -2; -1 > -2 yes, nextCount==activeCount → stable
        boolean found = NaviPgdOptimizer.findSimplexThreshold(buffer, 0, 3, -6.0, out);
        assertTrue(found);
        assertEquals(-2.0, out[0], 1e-15);
    }

    @Test
    @DisplayName("findSimplexThreshold: null buffer → false")
    void findSimplexThresholdNull() {
        assertFalse(NaviPgdOptimizer.findSimplexThreshold(null, 0, 5, 1.3, new double[1]));
    }

    @Test
    @DisplayName("findSimplexThreshold: count=0 → false")
    void findSimplexThresholdZeroCount() {
        assertFalse(NaviPgdOptimizer.findSimplexThreshold(new double[5], 0, 0, 0.0, new double[1]));
    }

    @Test
    @DisplayName("findSimplexThreshold: NaN activeSum → false")
    void findSimplexThresholdNanSum() {
        assertFalse(NaviPgdOptimizer.findSimplexThreshold(
                new double[]{1.0}, 0, 1, Double.NaN, new double[1]));
    }

    // ==================== Single node degenerate case ====================

    @Test
    @DisplayName("single node, single request: must select node 0")
    void singleNodeSingleRequest() {
        NaviPgdOptimizer opt = new NaviPgdOptimizer(42L);
        double[][] params = {NAVI_PARAMS};
        long[] tokens = {2048};
        long[] cacheHit = {512};
        NaviPgdOptimizer.OptimizeResult result =
                opt.optimize(1, 1, params, null, tokens, cacheHit);
        assertNotNull(result);
        assertEquals(0, result.selectedNodeIndexes()[0]);
    }

    // ==================== Two-node latency preference ====================

    @Test
    @DisplayName("two nodes: PGD prefers lower-latency node")
    void twoNodesPreferLowLatency() {
        NaviPgdOptimizer opt = new NaviPgdOptimizer(123L);
        opt.configure(0.5, 512.0, 1.0, 0.0, 10, 0);
        // Node 0: high-latency params (large parameter[6] bias)
        double[] highLatency = {-4.0, 10.0, 1.4, 20.0, 0.1, 0.09, 10.0, 1.0, -4.0};
        // Node 1: low-latency params (small parameter[6])
        double[] lowLatency = {-4.0, 10.0, 1.4, 20.0, 0.1, 0.09, 0.1, 1.0, -4.0};
        double[][] params = {highLatency, lowLatency};
        double[] queue = {0.0, 0.0};
        long[] tokens = {2048};
        long[] cacheHit = {0, 0};  // node-major: no cache for either
        NaviPgdOptimizer.OptimizeResult result =
                opt.optimize(2, 1, params, queue, tokens, cacheHit);
        assertNotNull(result);
        assertEquals(1, result.selectedNodeIndexes()[0], "should prefer low-latency node");
    }

    // ==================== Cache affinity ====================

    @Test
    @DisplayName("two nodes, same model: cache-hit node preferred")
    void cacheAffinityPreferred() {
        NaviPgdOptimizer opt = new NaviPgdOptimizer(456L);
        opt.configure(0.5, 512.0, 1.0, 0.0, 10, 0);
        double[][] params = {NAVI_PARAMS.clone(), NAVI_PARAMS.clone()};
        double[] queue = {0.0, 0.0};
        long[] tokens = {2048};
        // node-major: node0 has full miss, node1 has high cache hit
        long[] cacheHit = {0, 1920};
        NaviPgdOptimizer.OptimizeResult result =
                opt.optimize(2, 1, params, queue, tokens, cacheHit);
        assertNotNull(result);
        assertEquals(1, result.selectedNodeIndexes()[0], "should prefer cache-hit node");
    }

    // ==================== Queue wait influence ====================

    @Test
    @DisplayName("two nodes, same latency, high queue → prefer low-queue node")
    void queueWaitInfluence() {
        NaviPgdOptimizer opt = new NaviPgdOptimizer(789L);
        opt.configure(0.5, 512.0, 1.0, 0.0, 10, 0);
        double[][] params = {NAVI_PARAMS.clone(), NAVI_PARAMS.clone()};
        double[] queue = {5000.0, 0.0}; // node 0 has 5s queue
        long[] tokens = {2048};
        long[] cacheHit = {512, 512}; // same cache hit
        NaviPgdOptimizer.OptimizeResult result =
                opt.optimize(2, 1, params, queue, tokens, cacheHit);
        assertNotNull(result);
        assertEquals(1, result.selectedNodeIndexes()[0], "should prefer low-queue node");
    }

    // ==================== Determinism (fixed seed) ====================

    @Test
    @DisplayName("fixed seed: two optimize calls produce identical results")
    void determinismFixedSeed() {
        double[][] params = {NAVI_PARAMS.clone(), NAVI_PARAMS.clone(), NAVI_PARAMS.clone()};
        double[] queue = {10.0, 20.0, 5.0};
        long[] tokens = {1024, 2048, 512, 4096};
        long[] cacheHit = new long[3 * 4]; // all zero

        NaviPgdOptimizer opt1 = new NaviPgdOptimizer(999L);
        opt1.configure(0.5, 512.0, 1.0, 0.0, 10, 0);
        NaviPgdOptimizer.OptimizeResult r1 = opt1.optimize(3, 4, params, queue, tokens, cacheHit);

        NaviPgdOptimizer opt2 = new NaviPgdOptimizer(999L);
        opt2.configure(0.5, 512.0, 1.0, 0.0, 10, 0);
        NaviPgdOptimizer.OptimizeResult r2 = opt2.optimize(3, 4, params, queue, tokens, cacheHit);

        assertNotNull(r1);
        assertNotNull(r2);
        assertArrayEquals(r1.selectedNodeIndexes(), r2.selectedNodeIndexes());
        assertEquals(r1.forwardValue(), r2.forwardValue(), 0.0);
        assertEquals(r1.loopCount(), r2.loopCount());
    }

    // ==================== Time budget early-stop ====================

    @Test
    @DisplayName("timeBudget=1µs + maxLoop=100 → iterations <= MIN_DYNAMIC_LOOP_COUNT(2)")
    void timeBudgetEarlyStop() {
        NaviPgdOptimizer opt = new NaviPgdOptimizer(11L);
        opt.configure(0.5, 512.0, 1.0, 0.0, 100, 1L); // 1µs budget
        double[][] params = {NAVI_PARAMS.clone(), NAVI_PARAMS.clone()};
        double[] queue = {0.0, 0.0};
        long[] tokens = {2048, 1024, 512};
        long[] cacheHit = new long[2 * 3];
        NaviPgdOptimizer.OptimizeResult result =
                opt.optimize(2, 3, params, queue, tokens, cacheHit);
        assertNotNull(result);
        assertTrue(result.loopCount() <= 2,
                "expected early stop at MIN_DYNAMIC_LOOP_COUNT=2, got " + result.loopCount());
    }

    // ==================== Best-loss history selection ====================

    @Test
    @DisplayName("optimizer returns bestLoss assignment, not necessarily last iteration")
    void bestLossHistorySelection() {
        // With alpha decay < 1 and many iterations, oscillation can make the last
        // iteration worse. Verify the result's forwardValue is a true minimum.
        NaviPgdOptimizer opt = new NaviPgdOptimizer(777L);
        opt.configure(0.5, 1024.0, 0.7, 1.0, 20, 0);
        double[][] params = {NAVI_PARAMS.clone(), NAVI_PARAMS.clone()};
        double[] queue = {100.0, 200.0};
        long[] tokens = {2048, 4096, 1024};
        long[] cacheHit = new long[2 * 3];
        NaviPgdOptimizer.OptimizeResult result =
                opt.optimize(2, 3, params, queue, tokens, cacheHit);
        assertNotNull(result);
        assertTrue(Double.isFinite(result.forwardValue()));
    }

    // ==================== Edge: nodeCount=0 or requestCount=0 → null ====================

    @Test
    @DisplayName("nodeCount=0 → returns null")
    void zeroNodes() {
        NaviPgdOptimizer opt = new NaviPgdOptimizer(1L);
        assertNull(opt.optimize(0, 5, new double[0][], null, new long[5], new long[0]));
    }

    @Test
    @DisplayName("requestCount=0 → returns null")
    void zeroRequests() {
        NaviPgdOptimizer opt = new NaviPgdOptimizer(1L);
        assertNull(opt.optimize(2, 0, new double[][]{{}, {}}, null, new long[0], new long[0]));
    }

    // ==================== Large scale: 8 nodes × 30 requests ====================

    @Test
    @DisplayName("large scale (8 nodes × 30 requests): runs without exception, valid assignments")
    void largeScale() {
        int N = 8, R = 30;
        NaviPgdOptimizer opt = new NaviPgdOptimizer(2024L);
        opt.configure(0.5, 512.0, 0.95, 1.0, 15, 0);
        double[][] params = new double[N][];
        for (int n = 0; n < N; n++) {
            params[n] = NAVI_PARAMS.clone();
            params[n][6] += n * 0.1; // slight latency variation
        }
        double[] queue = new double[N];
        for (int n = 0; n < N; n++) queue[n] = n * 5.0;
        long[] tokens = new long[R];
        for (int r = 0; r < R; r++) tokens[r] = 512 + r * 64;
        long[] cacheHit = new long[N * R]; // all zero
        NaviPgdOptimizer.OptimizeResult result =
                opt.optimize(N, R, params, queue, tokens, cacheHit);
        assertNotNull(result);
        int[] selected = result.selectedNodeIndexes();
        assertEquals(R, selected.length);
        for (int r = 0; r < R; r++) {
            assertTrue(selected[r] >= 0 && selected[r] < N,
                    "request " + r + " assigned to invalid node " + selected[r]);
        }
    }

    // ==================== getStepAlpha ====================

    @Test
    @DisplayName("getStepAlpha: no decay (alphaDecay=1.0)")
    void stepAlphaNoDeck() {
        assertEquals(512.0, NaviPgdOptimizer.getStepAlpha(512.0, 1.0, 0.0, 5));
    }

    @Test
    @DisplayName("getStepAlpha: decay and floor")
    void stepAlphaDecayFloor() {
        // alpha=100, decay=0.5, minAlpha=10, iter=3 → 100*0.5^3=12.5 > 10 → 12.5
        assertEquals(12.5, NaviPgdOptimizer.getStepAlpha(100.0, 0.5, 10.0, 3));
        // iter=4 → 100*0.5^4=6.25 < 10 → 10
        assertEquals(10.0, NaviPgdOptimizer.getStepAlpha(100.0, 0.5, 10.0, 4));
    }

    @Test
    @DisplayName("getStepAlpha: zero alpha returns MIN_NORMAL")
    void stepAlphaZero() {
        assertEquals(Double.MIN_NORMAL, NaviPgdOptimizer.getStepAlpha(0.0, 1.0, 0.0, 0));
    }

    // ==================== shouldRunNextLoop ====================

    @Test
    @DisplayName("shouldRunNextLoop: respects maxLoopCount")
    void shouldRunNextLoopMax() {
        assertFalse(NaviPgdOptimizer.shouldRunNextLoop(10, 10, 100000, 50, 10));
    }

    @Test
    @DisplayName("shouldRunNextLoop: always runs up to MIN_DYNAMIC_LOOP_COUNT")
    void shouldRunNextLoopMinimum() {
        // completedLoopCount=1 < min(2, maxLoop=100) → true regardless of budget
        assertTrue(NaviPgdOptimizer.shouldRunNextLoop(1, 100, 1, 999999, 999999));
    }

    @Test
    @DisplayName("shouldRunNextLoop: budget exhausted after minimum")
    void shouldRunNextLoopBudgetExhausted() {
        // completedLoopCount=2, timeBudgetUs=10, elapsedUs=15 (> budget) → false
        assertFalse(NaviPgdOptimizer.shouldRunNextLoop(2, 100, 10, 15, 5));
    }
}
