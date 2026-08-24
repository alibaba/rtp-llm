package org.flexlb.state.internal.decode;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import org.flexlb.state.InternalApi;
import org.flexlb.state.PhaseVerdict;

/**
 * Decode 侧相位格纯函数集（相位蕴含闭包 / 越级闭包补记 / 相位裁决矩阵 D 分支）。
 *
 * <p>与 {@code org.flexlb.state.internal.prefill.PrefillLattice} 同构：
 * 无状态、全静态、纯函数——不持任何可变状态，天然线程安全。</p>
 */
@InternalApi
public final class DecodeLattice {

    private DecodeLattice() {
    }

    /**
     * 相位蕴含闭包（D 侧）：D 侧格为严格单调链，格内蕴含即前缀关系——
     * 处于相位 p 蕴含"已经过全部更低相位"，即 {@code EnumSet.range(RESERVED, p)}。
     * （D 侧对 P 侧的跨侧蕴含由上层组合裁决，格内只管 D 侧链。）
     */
    public static EnumSet<DecodePhase> implies(DecodePhase phase) {
        return EnumSet.range(DecodePhase.RESERVED, phase);
    }

    /**
     * 越级推进的沿途相位序列：收到 {@code from → to} 的越级事件时，
     * 返回含两端的完整沿途序列，供上层补记各相位 enteredAt。
     *
     * <p>契约：{@code to > from} 时返回 {@code [from .. to]} 全序列；
     * {@code to <= from} 无前向推进（迟到事件由 {@link #arbitrate} 先行丢弃），
     * 防御性返回 {@code [from]} 单元素。</p>
     */
    public static List<DecodePhase> closureBetween(DecodePhase from, DecodePhase to) {
        if (to.ordinal() <= from.ordinal()) {
            return List.of(from);
        }
        List<DecodePhase> path = new ArrayList<>(to.ordinal() - from.ordinal() + 1);
        for (int h = from.ordinal(); h <= to.ordinal(); h++) {
            path.add(DecodePhase.values()[h]);
        }
        return List.copyOf(path);
    }

    /**
     * 相位裁决矩阵（D 分支）：与 P 侧同构，签名与分支优先级完全一致（见
     * {@code PrefillLattice#arbitrate} 的分支 javadoc）——世代屏障 →
     * 版本屏障（DROP_DUP）→ 迟到中间态（DROP_LATE）→ 越级推进
     * （ACCEPT_ADVANCE / WARN_FINISH_PRIORITY）→ 同相位 finish（ACCEPT_TERMINAL）
     * → 同相位无推进（DROP_DUP）。
     */
    public static PhaseVerdict arbitrate(DecodePhase current,
                                  long currentVersion,
                                  DecodePhase eventPhase,
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
