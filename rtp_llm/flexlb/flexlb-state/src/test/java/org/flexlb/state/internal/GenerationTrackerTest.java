package org.flexlb.state.internal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.flexlb.state.GenerationTriple;
import org.junit.jupiter.api.Test;

/**
 * GenerationTracker 组件级：nextGeneration 单调、epoch 兜底
 * （注入小值历史验证 max 语义，防 master 重启归零）、isCurrent 屏障、拒绝计数。
 */
class GenerationTrackerTest {

    /** 单调递增：prev+1 推进。 */
    @Test
    void nextGenerationMonotonic() {
        GenerationTracker t = new GenerationTracker(1_000L);
        long g1 = t.nextGeneration(5L); // max(1000, 0+1) = 1000
        long g2 = t.nextGeneration(5L); // max(1000, 1000+1) = 1001
        long g3 = t.nextGeneration(5L);
        assertTrue(g2 > g1);
        assertTrue(g3 > g2);
        assertEquals(1_000L, g1);
        assertEquals(1_001L, g2);
        assertEquals(1_002L, g3);
        assertEquals(3L, t.generationBumps());
    }

    /** epoch 兜底：注入小值历史后换代取 max(epoch, prev+1)——重启归零防护。 */
    @Test
    void epochFloorPreventsRestartZeroing() {
        GenerationTracker t = new GenerationTracker(5_000L);
        // 注入"重启前"小值历史（rebuild 场景 observeGeneration merge max）
        t.observeGeneration(7L, 3L);
        t.observeGeneration(7L, 2L); // 更小值不回退
        assertEquals(3L, t.currentGeneration(7L));
        // 换代：max(5000, 3+1) = 5000（epoch 兜底生效）
        assertEquals(5_000L, t.nextGeneration(7L));
        assertEquals(5_001L, t.nextGeneration(7L));
    }

    /** isCurrent 屏障：未登记 false；登记后 true；换代后旧 false 新 true。 */
    @Test
    void isCurrentSemantics() {
        GenerationTracker t = new GenerationTracker(1_000L);
        GenerationTriple unregistered = new GenerationTriple(9, 0, -1);
        assertFalse(t.isCurrent(unregistered)); // 未登记端点拒绝

        long g1 = t.nextGeneration(9L);
        assertTrue(t.isCurrent(new GenerationTriple(9, g1, -1)));
        assertFalse(t.isCurrent(new GenerationTriple(9, g1 + 1, -1)));

        long g2 = t.nextGeneration(9L); // 换代
        assertFalse(t.isCurrent(new GenerationTriple(9, g1, -1))); // 旧代拒绝
        assertTrue(t.isCurrent(new GenerationTriple(9, g2, -1)));  // 新代通过
    }

    /** 端点隔离：各端点独立登记；epoch 兜底下不同端点首代可同值（世代号仅端点内单调）。 */
    @Test
    void endpointsAreIsolated() {
        GenerationTracker t = new GenerationTracker(1_000L);
        long gA1 = t.nextGeneration(1L); // 1000（epoch 兜底）
        long gB = t.nextGeneration(2L);  // 1000（另一端点独立从 epoch 起）
        assertEquals(gA1, gB); // 跨端点同代值合法（隔离按 endpointId）
        long gA2 = t.nextGeneration(1L); // 1001：端点 1 换代
        assertTrue(t.isCurrent(new GenerationTriple(1, gA2, -1)));
        assertTrue(t.isCurrent(new GenerationTriple(2, gB, -1)));
        assertFalse(t.isCurrent(new GenerationTriple(1, gA1, -1))); // 端点 1 旧代拒绝
        assertFalse(t.isCurrent(new GenerationTriple(2, gA2, -1))); // 端点 2 未达该代
    }

    /** 跨代拒绝计数（供观测：旧代事件整报 REJECT）。 */
    @Test
    void crossGenerationRejectCounter() {
        GenerationTracker t = new GenerationTracker(1_000L);
        t.recordCrossGenerationReject();
        t.recordCrossGenerationReject();
        assertEquals(2L, t.crossGenerationRejects());
    }
}
