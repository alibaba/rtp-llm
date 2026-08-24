package org.flexlb.state.internal.prefill;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import org.flexlb.state.InternalApi;
import org.flexlb.state.PhaseVerdict;

/**
 * Prefill 侧相位格纯函数集（I2 蕴含闭包 / L9 越级闭包补记 / S4 裁决矩阵 P 分支）。
 *
 * <p>无状态、全静态、纯函数——不持任何可变状态，天然线程安全。</p>
 */
@InternalApi
public final class PrefillLattice {

    private PrefillLattice() {
    }

    /**
     * I2 蕴含闭包（P 侧）：P 侧格为严格单调链，格内蕴含即前缀关系——
     * 处于相位 p 蕴含"已经过全部更低相位"，即 {@code EnumSet.range(INIT, p)}。
     * （D 侧才需要跨侧蕴含；P 侧无跨侧相位。）
     */
    public static EnumSet<PrefillPhase> implies(PrefillPhase phase) {
        return EnumSet.range(PrefillPhase.INIT, phase);
    }

    /**
     * 越级推进的沿途相位序列（L9）：收到 {@code from → to} 的越级事件时，
     * 返回含两端的完整沿途序列，供上层补记各相位 enteredAt。
     *
     * <p>契约：{@code to > from} 时返回 {@code [from .. to]} 全序列；
     * {@code to <= from} 无前向推进（迟到事件由 {@link #arbitrate} 先行丢弃），
     * 防御性返回 {@code [from]} 单元素。</p>
     */
    public static List<PrefillPhase> closureBetween(PrefillPhase from, PrefillPhase to) {
        if (to.ordinal() <= from.ordinal()) {
            return List.of(from);
        }
        List<PrefillPhase> path = new ArrayList<>(to.ordinal() - from.ordinal() + 1);
        for (int h = from.ordinal(); h <= to.ordinal(); h++) {
            path.add(PrefillPhase.values()[h]);
        }
        return List.copyOf(path);
    }

    /**
     * S4 裁决矩阵（P 分支）：对一条相位事件裁决接受/丢弃/拒绝。
     *
     * <p>分支优先级（自上而下短路）：</p>
     * <ol>
     *   <li><b>世代屏障</b>：{@code !generationMatch} → {@link PhaseVerdict#REJECT_GENERATION}
     *       ——世代三元组不匹配整报拒绝（S8），优先级最高。</li>
     *   <li><b>版本屏障（陈旧）</b>：{@code eventVersion < currentVersion} →
     *       {@link PhaseVerdict#DROP_DUP}——版本单调假设被破坏时的丢弃语义（L2），
     *       陈旧上报不论相位如何一律按重复丢弃。</li>
     *   <li><b>迟到中间态</b>：{@code eventPhase < current} → {@link PhaseVerdict#DROP_LATE}
     *       ——乱序到达的低位中间态丢弃（L9）。</li>
     *   <li><b>越级推进</b>：{@code eventPhase > current} →
     *       isFinish ? {@link PhaseVerdict#WARN_FINISH_PRIORITY}（finish 事件携带相位超前于
     *       本地已知相位——优先级倒挂告警，但 finish 是最强证据仍接受推进）
     *       : {@link PhaseVerdict#ACCEPT_ADVANCE}（闭包推进，上层以
     *       {@link #closureBetween} 补记沿途 enteredAt）。</li>
     *   <li><b>同相位 finish</b>：{@code eventPhase == current && isFinish} →
     *       {@link PhaseVerdict#ACCEPT_TERMINAL}——终态判定入口：相位推到 PREFILL_DONE
     *       或由上层 settle 为 TerminalState。</li>
     *   <li><b>同相位无推进</b>（含 {@code eventVersion == currentVersion} 的重放、
     *       以及 {@code eventVersion > currentVersion} 但相位持平的新鲜观察）→
     *       {@link PhaseVerdict#DROP_DUP}——格层只裁决相位变化：同相位事件不产生格推进，
     *       按相位重复处理；同相位新鲜观察的数据更新（kvTokens 等）由上层按 version 采纳，
     *       不经过格。</li>
     * </ol>
     */
    public static PhaseVerdict arbitrate(PrefillPhase current,
                                  long currentVersion,
                                  PrefillPhase eventPhase,
                                  long eventVersion,
                                  boolean isFinish,
                                  boolean generationMatch) {
        if (!generationMatch) {
            return PhaseVerdict.REJECT_GENERATION;
        }
        if (eventVersion < currentVersion) {
            return PhaseVerdict.DROP_DUP;
        }
        if (eventPhase.ordinal() < current.ordinal()) {
            return PhaseVerdict.DROP_LATE;
        }
        if (eventPhase.ordinal() > current.ordinal()) {
            return isFinish ? PhaseVerdict.WARN_FINISH_PRIORITY : PhaseVerdict.ACCEPT_ADVANCE;
        }
        if (isFinish) {
            return PhaseVerdict.ACCEPT_TERMINAL;
        }
        return PhaseVerdict.DROP_DUP;
    }
}
