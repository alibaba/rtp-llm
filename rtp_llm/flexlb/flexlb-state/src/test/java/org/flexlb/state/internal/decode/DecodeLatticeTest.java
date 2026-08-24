package org.flexlb.state.internal.decode;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import org.flexlb.state.PhaseVerdict;
import org.flexlb.state.spi.EnginePhase;
import org.junit.jupiter.api.Test;

/**
 * S4 裁决矩阵（D 分支）机器可枚举性证明（R7）：
 * current × eventPhase × version 关系(&lt;==&gt;) × isFinish × generationMatch 全组合穷举
 * （4 × 4 × 3 × 2 × 2 = 192 组合），外加 closureBetween 越级闭包全对枚举与
 * L2 / L9 / L18 教训回归。与 P 侧（PrefillLatticeTest）同构。
 */
class DecodeLatticeTest {

    private static final long CURRENT_VERSION = 10L;

    /** S4 裁决矩阵规格直译（测试 oracle）——分支优先级：世代 → 版本 → 相位 → finish。 */
    private static PhaseVerdict oracle(DecodePhase cur, long cv, DecodePhase ep, long ev,
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
        for (DecodePhase cur : DecodePhase.values()) {
            for (DecodePhase ep : DecodePhase.values()) {
                for (long ev : versions) {
                    for (boolean fin : new boolean[]{false, true}) {
                        for (boolean gen : new boolean[]{false, true}) {
                            PhaseVerdict expected = oracle(cur, CURRENT_VERSION, ep, ev, fin, gen);
                            PhaseVerdict actual = DecodeLattice.arbitrate(
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
        assertEquals(192, combos, "4 current × 4 event × 3 version × 2 finish × 2 gen");
    }

    // ---- 代表性硬断言（不依赖 oracle，矩阵关键格手工点验） ----

    @Test
    void generationMismatchAlwaysRejected() {
        for (DecodePhase cur : DecodePhase.values()) {
            for (DecodePhase ep : DecodePhase.values()) {
                assertEquals(PhaseVerdict.REJECT_GENERATION, DecodeLattice.arbitrate(
                        cur, 5, ep, 99, true, false),
                        "世代不匹配优先级最高：cur=" + cur + ", ep=" + ep);
            }
        }
    }

    @Test
    void skipLevelAdvanceAccepted() {
        // E10：RESERVED 直接观察到 KV_ALLOCATED → D_LOADING（横跨 LOAD 传输期）。
        assertEquals(PhaseVerdict.ACCEPT_ADVANCE, DecodeLattice.arbitrate(
                DecodePhase.RESERVED, 5, DecodePhase.D_LOADING, 6, false, true));
        assertEquals(PhaseVerdict.ACCEPT_ADVANCE, DecodeLattice.arbitrate(
                DecodePhase.RESERVED, 1, DecodePhase.D_RUNNING, 2, false, true));
    }

    @Test
    void samePhaseFinishAcceptedTerminal() {
        assertEquals(PhaseVerdict.ACCEPT_TERMINAL, DecodeLattice.arbitrate(
                DecodePhase.D_RUNNING, 5, DecodePhase.D_RUNNING, 6, true, true));
        assertEquals(PhaseVerdict.ACCEPT_TERMINAL, DecodeLattice.arbitrate(
                DecodePhase.D_RUNNING, 5, DecodePhase.D_RUNNING, 5, true, true));
    }

    @Test
    void finishPriorityInversionWarnsButAccepts() {
        assertEquals(PhaseVerdict.WARN_FINISH_PRIORITY, DecodeLattice.arbitrate(
                DecodePhase.DISPATCHED, 5, DecodePhase.D_LOADING, 6, true, true));
    }

    /** L9 回归：乱序到达的迟到中间态丢弃——且版本新鲜也照丢。 */
    @Test
    void l9_lateIntermediateEventDropped() {
        assertEquals(PhaseVerdict.DROP_LATE, DecodeLattice.arbitrate(
                DecodePhase.D_RUNNING, 5, DecodePhase.DISPATCHED, 6, false, true));
        assertEquals(PhaseVerdict.DROP_LATE, DecodeLattice.arbitrate(
                DecodePhase.D_LOADING, 50, DecodePhase.RESERVED, 51, false, true));
    }

    /** L2 回归：version 单调假设破坏（陈旧上报）→ 一律按 DROP_DUP 丢弃，finish 也不例外。 */
    @Test
    void l2_staleVersionDroppedAsDup() {
        assertEquals(PhaseVerdict.DROP_DUP, DecodeLattice.arbitrate(
                DecodePhase.DISPATCHED, 5, DecodePhase.D_RUNNING, 4, false, true));
        assertEquals(PhaseVerdict.DROP_DUP, DecodeLattice.arbitrate(
                DecodePhase.D_RUNNING, 5, DecodePhase.D_RUNNING, 4, true, true));
    }

    /** 任务矩阵明确格：version 相等且同相位（上报重放）→ DROP_DUP。 */
    @Test
    void sameVersionSamePhaseReplayDropped() {
        assertEquals(PhaseVerdict.DROP_DUP, DecodeLattice.arbitrate(
                DecodePhase.D_LOADING, 5, DecodePhase.D_LOADING, 5, false, true));
    }

    /** 同相位但版本更新：格层无相位推进 → 按相位重复丢弃（数据更新走上层）。 */
    @Test
    void samePhaseHigherVersionNoAdvance() {
        assertEquals(PhaseVerdict.DROP_DUP, DecodeLattice.arbitrate(
                DecodePhase.D_RUNNING, 5, DecodePhase.D_RUNNING, 6, false, true));
    }

    // ---- closureBetween（L9 越级闭包补记）----

    @Test
    void closureBetweenSkippedLevels() {
        assertEquals(
                List.of(DecodePhase.RESERVED, DecodePhase.DISPATCHED, DecodePhase.D_LOADING,
                        DecodePhase.D_RUNNING),
                DecodeLattice.closureBetween(DecodePhase.RESERVED, DecodePhase.D_RUNNING));
    }

    @Test
    void closureBetweenAllPairs() {
        DecodePhase[] all = DecodePhase.values();
        for (DecodePhase from : all) {
            for (DecodePhase to : all) {
                List<DecodePhase> expected = new ArrayList<>();
                if (to.ordinal() > from.ordinal()) {
                    for (int h = from.ordinal(); h <= to.ordinal(); h++) {
                        expected.add(all[h]);
                    }
                } else {
                    expected.add(from);
                }
                assertEquals(expected, DecodeLattice.closureBetween(from, to),
                        "from=" + from + ", to=" + to);
            }
        }
    }

    // ---- implies（I2 蕴含闭包，D 侧格内前缀关系）----

    @Test
    void impliesIsPrefixClosure() {
        assertEquals(EnumSet.of(DecodePhase.RESERVED), DecodeLattice.implies(DecodePhase.RESERVED));
        assertEquals(EnumSet.range(DecodePhase.RESERVED, DecodePhase.D_LOADING),
                DecodeLattice.implies(DecodePhase.D_LOADING));
        assertEquals(EnumSet.allOf(DecodePhase.class), DecodeLattice.implies(DecodePhase.D_RUNNING));
    }

    // ---- L18 回归：fromEnginePhase 映射完备性 ----

    /** L18 回归：引擎无显式中间相位只能倒推——每个 EnginePhase 值必须有保守映射。 */
    @Test
    void l18_everyEnginePhaseHasMapping() {
        for (EnginePhase enginePhase : EnginePhase.values()) {
            assertNotNull(DecodePhase.fromEnginePhase(enginePhase),
                    "EnginePhase." + enginePhase + " 必须有 D 侧映射");
        }
        assertEquals(DecodePhase.DISPATCHED, DecodePhase.fromEnginePhase(EnginePhase.RECEIVED));
        assertEquals(DecodePhase.D_LOADING, DecodePhase.fromEnginePhase(EnginePhase.KV_ALLOCATED));
        assertEquals(DecodePhase.D_RUNNING, DecodePhase.fromEnginePhase(EnginePhase.RUNNING));
        // PENDING 保守最低观察位：
        assertEquals(DecodePhase.DISPATCHED, DecodePhase.fromEnginePhase(EnginePhase.PENDING));
    }

    /** 格高度：ordinal 严格单调链（声明顺序即偏序）。 */
    @Test
    void ordinalIsLatticeHeight() {
        DecodePhase[] all = DecodePhase.values();
        for (int i = 0; i < all.length; i++) {
            assertEquals(i, all[i].ordinal(),
                    all[i] + " 的 ordinal 必须等于其格高度（声明顺序不可重排）");
        }
        assertEquals(4, all.length);
    }

    /** 格内无终态（终局归 TerminalState）。 */
    @Test
    void noTerminalPhaseInLattice() {
        for (DecodePhase p : DecodePhase.values()) {
            assertEquals(false, p.isTerminal(), p + " 不是终态（D_RUNNING 是格顶而非吸收终态）");
        }
    }
}
