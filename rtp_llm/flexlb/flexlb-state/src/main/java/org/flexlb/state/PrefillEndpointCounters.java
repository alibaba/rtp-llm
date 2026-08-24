package org.flexlb.state;

import java.util.List;

/**
 * P 侧端点级派生计数快照（不可变；调度读点的 O(1) 数据源）。
 *
 * <p>由 {@code PrefillSideStore} 端点级增量计数簿无锁读出。条目在
 * 派发完成（onDispatched 绑定世代）后进入端点索引——排队/攒批窗口的
 * 条目由派发编排侧（batcher 队列深度）单独覆盖。</p>
 *
 * @param activeTotal      该端点名下活跃条目总数（已派发未终局）
 * @param engineOwnedCount 引擎已见条目数
 * @param estimatedWaitMs  等待估算（Σ活跃条目分摊批次预测耗时；分摊口径下
 *                         批次成员求和 ≈ 批次耗时。引擎执行中条目不折扣
 *                         ——保守高估，拒绝偏好。未记预测的条目计 0）
 * @param phaseCounts      各相位人口（下标 = PrefillPhase.ordinal，0..9）
 */
public record PrefillEndpointCounters(
        int activeTotal,
        int engineOwnedCount,
        long estimatedWaitMs,
        List<Long> phaseCounts) {

    public PrefillEndpointCounters {
        phaseCounts = List.copyOf(phaseCounts);
    }

    /** 全零视图（端点无任何活跃条目时的语义等价常量）。 */
    public static PrefillEndpointCounters empty() {
        return new PrefillEndpointCounters(0, 0, 0L, List.of());
    }
}
