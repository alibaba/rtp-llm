package org.flexlb.state.internal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Optional;
import org.flexlb.state.TerminalOutcome;
import org.flexlb.state.TerminalReason;
import org.flexlb.state.TerminalState;
import org.junit.jupiter.api.Test;

/**
 * TombstoneStore 组件级：判重窗口（retention 边界）、幂等吸收、
 * 迟到事件/取消计数、线性扫过期清理、reset。
 */
class TombstoneStoreTest {

    private static TerminalOutcome outcome(TerminalState state) {
        return new TerminalOutcome(state, TerminalReason.SUCCEEDED, "");
    }

    /** 窗口内判重命中；retention=0 立即过期（判重放行）。 */
    @Test
    void windowedJudgement() {
        TombstoneStore huge = new TombstoneStore(Long.MAX_VALUE / 2);
        huge.absorb(1L, outcome(TerminalState.COMPLETED), 1_000L);
        assertTrue(huge.isTombstoned(1L));
        assertFalse(huge.isTombstoned(2L));

        TombstoneStore zero = new TombstoneStore(0L);
        zero.absorb(1L, outcome(TerminalState.COMPLETED), 1_000L);
        assertFalse(zero.isTombstoned(1L)); // now - 1000 >= 0 恒真 → 立即过期
    }

    /** 吸收幂等：同 requestId 已有墓碑时保留首条。 */
    @Test
    void absorbKeepsFirstTombstone() {
        TombstoneStore ts = new TombstoneStore(60_000L);
        ts.absorb(1L, outcome(TerminalState.COMPLETED), 1_000L);
        ts.absorb(1L, outcome(TerminalState.FAILED), 2_000L);
        Optional<TombstoneStore.Tombstone> t = ts.get(1L);
        assertTrue(t.isPresent());
        assertEquals(TerminalState.COMPLETED, t.orElseThrow().state());
        assertEquals(1_000L, t.orElseThrow().terminalAtMs());
        assertEquals(1, ts.size());
    }

    /** 迟到事件/迟到取消独立计数。 */
    @Test
    void lateEventAndCancelCounters() {
        TombstoneStore ts = new TombstoneStore(60_000L);
        ts.absorbLateEvent();
        ts.absorbLateEvent();
        ts.absorbLateCancel();
        assertEquals(2L, ts.lateEventCount());
        assertEquals(1L, ts.lateCancelCount());
    }

    /** 线性扫过期清理：边界 now - terminalAt >= retention 判过期，返回清理条数。 */
    @Test
    void evictExpiredLinearScanBoundary() {
        TombstoneStore ts = new TombstoneStore(100L);
        ts.absorb(1L, outcome(TerminalState.COMPLETED), 1_000L);
        ts.absorb(2L, outcome(TerminalState.COMPLETED), 1_050L);
        ts.absorb(3L, outcome(TerminalState.COMPLETED), 1_200L);

        // 1149：条目 1（149 >= 100）过期；条目 2（99 < 100）未过期；条目 3 未过期
        // （isTombstoned 以墙钟判定，注入时刻下过期语义由 evictExpired 验证）
        assertEquals(1, ts.evictExpired(1_149L));
        assertTrue(ts.get(1L).isEmpty());
        assertTrue(ts.get(2L).isPresent()); // 未被清理

        // 1150：条目 2（100 >= 100）边界过期
        assertEquals(1, ts.evictExpired(1_150L));
        assertFalse(ts.isTombstoned(2L));
        assertEquals(1, ts.size());
    }

    /** reset 清空墓碑与计数（rebuild 用）。 */
    @Test
    void resetClearsAll() {
        TombstoneStore ts = new TombstoneStore(60_000L);
        ts.absorb(1L, outcome(TerminalState.COMPLETED), 1_000L);
        ts.absorbLateEvent();
        ts.reset();
        assertEquals(0, ts.size());
        assertFalse(ts.isTombstoned(1L));
        assertEquals(0L, ts.lateEventCount());
        assertEquals(0L, ts.lateCancelCount());
    }

    /** 构造参数校验：负 retention 拒绝。 */
    @Test
    void negativeRetentionRejected() {
        org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class,
                () -> new TombstoneStore(-1L));
    }
}
