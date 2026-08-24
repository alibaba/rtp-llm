package org.flexlb.state;

import java.util.List;

/**
 * 批次影子视图（B6 双视图）：从 P 侧活跃条目按 batchId 聚类的只读视图。
 *
 * <ul>
 *   <li>{@link #maxPhaseOrdinal()}：任一成员 P_RUNNING（ordinal 8）即视为批次在跑
 *       （强链事实——任一成员已进入执行）；空批次为 -1。</li>
 *   <li>{@link #minPhaseOrdinal()}：全成员最小相位（弱链等待估算——保留现路由行为输入）。</li>
 * </ul>
 *
 * @param batchId         批次 ID
 * @param members         成员只读视图（活跃条目快照）
 * @param maxPhaseOrdinal 成员最大相位高度
 * @param minPhaseOrdinal 成员最小相位高度
 */
public record BatchShadowView(
        long batchId,
        List<PrefillRequestStateView> members,
        int maxPhaseOrdinal,
        int minPhaseOrdinal) {

    /** PrefillPhase.P_RUNNING 的 ordinal（批次"在跑"判定阈值）。 */
    public static final int RUNNING_PHASE_ORDINAL = 8;

    public BatchShadowView {
        members = List.copyOf(members);
    }

    /** 任一成员已进入 P_RUNNING（批次在跑，强链事实）。 */
    public boolean anyRunning() {
        return maxPhaseOrdinal >= RUNNING_PHASE_ORDINAL;
    }

    /** 批次是否为空（无活跃成员）。 */
    public boolean isEmpty() {
        return members.isEmpty();
    }
}
