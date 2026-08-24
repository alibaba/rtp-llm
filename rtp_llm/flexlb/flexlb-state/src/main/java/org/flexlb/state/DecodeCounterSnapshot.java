package org.flexlb.state;

import java.util.List;

/**
 * D 侧派生计数快照（不可变；由 {@code DecodeSideStore} 每 N 次转换发布或显式刷新）。
 *
 * @param phaseCounts          各相位人口（下标 = DecodePhase.ordinal，0..3）
 * @param activeTotal          活跃条目总量
 * @param reservedKvTotal      D① 影子预占 KV 合计（KV_ALLOCATED 确认后逐条撤出）
 * @param expectedKvTotal      预约时声明的期望 KV 合计（历史记录，不随确认撤出）
 * @param kvTokensReportedTotal D② 引擎事实 KV 合计（KV_ALLOCATED 起接管）
 * @param confirmed            已确认条目数（phase ≥ D_LOADING：引擎事实已接管预占）
 */
public record DecodeCounterSnapshot(
        List<Long> phaseCounts,
        long activeTotal,
        long reservedKvTotal,
        long expectedKvTotal,
        long kvTokensReportedTotal,
        long confirmed) {

    public DecodeCounterSnapshot {
        phaseCounts = List.copyOf(phaseCounts);
    }
}
