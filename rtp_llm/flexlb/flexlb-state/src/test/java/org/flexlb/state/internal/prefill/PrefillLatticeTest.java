package org.flexlb.state.internal.prefill;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import org.flexlb.state.PhaseVerdict;
import org.flexlb.state.spi.EnginePhase;
import org.junit.jupiter.api.Test;

/**
 * S4 裁决矩阵（P 分支）机器可枚举性证明（R7）：
 * current × eventPhase × version 关系(&lt;==&gt;) × isFinish × generationMatch 全组合穷举
 * （10 × 10 × 3 × 2 × 2 = 1200 组合），外加 closureBetween 越级闭包全对枚举与
 * L2 / L9 / L18 教训回归。
 */
class PrefillLatticeTest {

    private static final long CURRENT_VERSION = 10L;

    /**
     * S4 裁决矩阵规格直译（测试 oracle）——分支优先级：世代 → 版本 → 相位 → finish。
     * 与实现独立维护；代表性组合另有不依赖 oracle 的硬断言点验，防 oracle 与实现同错。
     */
    private static PhaseVerdict oracle(PrefillPhase cur, long cv, PrefillPhase ep, long ev,
                                       boolean fin, boolean gen) {
        if (!gen) {
            return PhaseVerdict.REJECT_GENERATION;
        }
        if (ev < cv) {
            return PhaseVerdict.DROP_DUP;
        }
        if (ep.ordinal() < cur.ordinal()) {
            return PhaseVerdict.DROP_LATE;
        }
        if (ep.ordinal() > cur.ordinal()) {
            return fin ? PhaseVerdict.WARN_FINISH_PRIORITY : PhaseVerdict.ACCEPT_ADVANCE;
        }
        if (fin) {
            return PhaseVerdict.ACCEPT_TERMINAL;
        }
        return PhaseVerdict.DROP_DUP;
    }

    /** 穷举全组合：每个组合断言实现裁决 == 矩阵 oracle。 */
    @Test
    void exhaustiveArbitrateMatrix() {
        long[] versions = {CURRENT_VERSION - 1, CURRENT_VERSION, CURRENT_VERSION + 1};
        int combos = 0;
        for (PrefillPhase cur : PrefillPhase.values()) {
            for (PrefillPhase ep : PrefillPhase.values()) {
                for (long ev : versions) {
                    for (boolean fin : new boolean[]{false, true}) {
                        for (boolean gen : new boolean[]{false, true}) {
                            PhaseVerdict expected = oracle(cur, CURRENT_VERSION, ep, ev, fin, gen);
                            PhaseVerdict actual = PrefillLattice.arbitrate(
                                    cur, CURRENT_VERSION, ep, ev, fin, gen);
                            assertEquals(expected, actual,
                                    () -> "cur=" + cur + ", ev=" + ev + ", ep=" + ep
                                            + ", fin=" + fin + ", gen=" + gen);
                            combos++;
                        }
                    }
                }
            }
        }
        assertEquals(1200, combos, "10 current × 10 event × 3 version × 2 finish × 2 gen");
    }

    // ---- 代表性硬断言（不依赖 oracle，矩阵关键格手工点验） ----

    @Test
    void generationMismatchAlwaysRejected() {
        for (PrefillPhase cur : PrefillPhase.values()) {
            for (PrefillPhase ep : PrefillPhase.values()) {
                assertEquals(PhaseVerdict.REJECT_GENERATION, PrefillLattice.arbitrate(
                        cur, 5, ep, 99, true, false),
                        "世代不匹配优先级最高：cur=" + cur + ", ep=" + ep);
            }
        }
    }

    @Test
    void skipLevelAdvanceAccepted() {
        assertEquals(PhaseVerdict.ACCEPT_ADVANCE, PrefillLattice.arbitrate(
                PrefillPhase.QUEUED, 5, PrefillPhase.P_RUNNING, 6, false, true));
        assertEquals(PhaseVerdict.ACCEPT_ADVANCE, PrefillLattice.arbitrate(
                PrefillPhase.INIT, 1, PrefillPhase.PREFILL_DONE, 2, false, true));
    }

    @Test
    void samePhaseFinishAcceptedTerminal() {
        assertEquals(PhaseVerdict.ACCEPT_TERMINAL, PrefillLattice.arbitrate(
                PrefillPhase.P_RUNNING, 5, PrefillPhase.P_RUNNING, 6, true, true));
        assertEquals(PhaseVerdict.ACCEPT_TERMINAL, PrefillLattice.arbitrate(
                PrefillPhase.P_RUNNING, 5, PrefillPhase.P_RUNNING, 5, true, true));
    }

    @Test
    void finishPriorityInversionWarnsButAccepts() {
        // finish 事件携带相位超前于当前已知相位：优先级倒挂告警，仍接受 finish 推进。
        assertEquals(PhaseVerdict.WARN_FINISH_PRIORITY, PrefillLattice.arbitrate(
                PrefillPhase.P_WAITING_UNLOADED, 5, PrefillPhase.P_RUNNING, 6, true, true));
    }

    /** L9 回归：乱序到达的迟到中间态丢弃——且版本新鲜也照丢（相位证据优先于版本新鲜度）。 */
    @Test
    void l9_lateIntermediateEventDropped() {
        assertEquals(PhaseVerdict.DROP_LATE, PrefillLattice.arbitrate(
                PrefillPhase.P_RUNNING, 5, PrefillPhase.P_RECEIVED, 6, false, true));
        assertEquals(PhaseVerdict.DROP_LATE, PrefillLattice.arbitrate(
                PrefillPhase.P_WAITING_LOADED, 50, PrefillPhase.P_WAITING_UNLOADED, 51, false, true));
        assertEquals(PhaseVerdict.DROP_LATE, PrefillLattice.arbitrate(
                PrefillPhase.PREFILL_DONE, 5, PrefillPhase.DISPATCHED, 100, false, true));
    }

    /** L2 回归：version 单调假设破坏（陈旧上报）→ 一律按 DROP_DUP 丢弃，finish 也不例外。 */
    @Test
    void l2_staleVersionDroppedAsDup() {
        assertEquals(PhaseVerdict.DROP_DUP, PrefillLattice.arbitrate(
                PrefillPhase.P_WAITING_UNLOADED, 5, PrefillPhase.P_RUNNING, 4, false, true));
        assertEquals(PhaseVerdict.DROP_DUP, PrefillLattice.arbitrate(
                PrefillPhase.P_RUNNING, 5, PrefillPhase.PREFILL_DONE, 4, true, true));
    }

    /** 任务矩阵明确格：version 相等且同相位（上报重放）→ DROP_DUP。 */
    @Test
    void sameVersionSamePhaseReplayDropped() {
        assertEquals(PhaseVerdict.DROP_DUP, PrefillLattice.arbitrate(
                PrefillPhase.P_RUNNING, 5, PrefillPhase.P_RUNNING, 5, false, true));
    }

    /** 同相位但版本更新（RUNNING 期间多轮观察）：格层无相位推进 → 按相位重复丢弃（数据更新走上层）。 */
    @Test
    void samePhaseHigherVersionNoAdvance() {
        assertEquals(PhaseVerdict.DROP_DUP, PrefillLattice.arbitrate(
                PrefillPhase.P_RUNNING, 5, PrefillPhase.P_RUNNING, 6, false, true));
    }

    // ---- closureBetween（L9 越级闭包补记）----

    @Test
    void closureBetweenSkippedLevels() {
        List<PrefillPhase> expected = List.of(
                PrefillPhase.QUEUED, PrefillPhase.DISPATCHING, PrefillPhase.DISPATCHED,
                PrefillPhase.P_RECEIVED, PrefillPhase.P_WAITING_UNLOADED,
                PrefillPhase.P_WAITING_LOADED, PrefillPhase.P_RUNNING);
        assertEquals(expected, PrefillLattice.closureBetween(
                PrefillPhase.QUEUED, PrefillPhase.P_RUNNING));
    }

    @Test
    void closureBetweenAllPairs() {
        PrefillPhase[] all = PrefillPhase.values();
        for (PrefillPhase from : all) {
            for (PrefillPhase to : all) {
                List<PrefillPhase> expected = new ArrayList<>();
                if (to.ordinal() > from.ordinal()) {
                    for (int h = from.ordinal(); h <= to.ordinal(); h++) {
                        expected.add(all[h]);
                    }
                } else {
                    expected.add(from);
                }
                assertEquals(expected, PrefillLattice.closureBetween(from, to),
                        "from=" + from + ", to=" + to);
            }
        }
    }

    @Test
    void closureBetweenAdjacentAndIdentity() {
        assertEquals(List.of(PrefillPhase.INIT), PrefillLattice.closureBetween(
                PrefillPhase.INIT, PrefillPhase.INIT));
        assertEquals(List.of(PrefillPhase.P_WAITING_UNLOADED), PrefillLattice.closureBetween(
                PrefillPhase.P_WAITING_UNLOADED, PrefillPhase.P_WAITING_UNLOADED));
        assertEquals(List.of(PrefillPhase.INIT, PrefillPhase.ROUTED), PrefillLattice.closureBetween(
                PrefillPhase.INIT, PrefillPhase.ROUTED));
    }

    // ---- implies（I2 蕴含闭包，P 侧格内前缀关系）----

    @Test
    void impliesIsPrefixClosure() {
        assertEquals(EnumSet.of(PrefillPhase.INIT), PrefillLattice.implies(PrefillPhase.INIT));
        assertEquals(EnumSet.range(PrefillPhase.INIT, PrefillPhase.P_RUNNING),
                PrefillLattice.implies(PrefillPhase.P_RUNNING));
        assertEquals(EnumSet.allOf(PrefillPhase.class), PrefillLattice.implies(PrefillPhase.PREFILL_DONE));
        for (PrefillPhase p : PrefillPhase.values()) {
            assertEquals(EnumSet.range(PrefillPhase.INIT, p), PrefillLattice.implies(p),
                    "implies(" + p + ") 应为 [INIT..p] 前缀闭包");
        }
    }

    // ---- L18 回归：fromEnginePhase 映射完备性 ----

    /** L18 回归：引擎无显式中间相位只能倒推——每个 EnginePhase 值必须有保守映射。 */
    @Test
    void l18_everyEnginePhaseHasMapping() {
        for (EnginePhase enginePhase : EnginePhase.values()) {
            assertNotNull(PrefillPhase.fromEnginePhase(enginePhase),
                    "EnginePhase." + enginePhase + " 必须有 P 侧映射");
        }
        assertEquals(PrefillPhase.P_RECEIVED, PrefillPhase.fromEnginePhase(EnginePhase.RECEIVED));
        assertEquals(PrefillPhase.P_WAITING_LOADED, PrefillPhase.fromEnginePhase(EnginePhase.KV_ALLOCATED));
        assertEquals(PrefillPhase.P_RUNNING, PrefillPhase.fromEnginePhase(EnginePhase.RUNNING));
        // PENDING 保守最低观察位：
        assertEquals(PrefillPhase.P_RECEIVED, PrefillPhase.fromEnginePhase(EnginePhase.PENDING));
    }

    /** 格高度：ordinal 严格单调链（声明顺序即偏序）。 */
    @Test
    void ordinalIsLatticeHeight() {
        PrefillPhase[] all = PrefillPhase.values();
        for (int i = 0; i < all.length; i++) {
            assertEquals(i, all[i].ordinal(),
                    all[i] + " 的 ordinal 必须等于其格高度（声明顺序不可重排）");
        }
        assertEquals(10, all.length);
    }

    /** 格内无终态（终局归 TerminalState）。 */
    @Test
    void noTerminalPhaseInLattice() {
        for (PrefillPhase p : PrefillPhase.values()) {
            assertEquals(false, p.isTerminal(), p + " 不是终态（PREFILL_DONE 是格顶而非吸收终态）");
        }
    }
}
