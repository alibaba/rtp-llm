package org.flexlb.state;

import java.util.List;

/**
 * P 侧端点级派生计数快照（不可变；读取换权阶段 G4 的调度读数数据源）。
 *
 * <p>由 {@code PrefillSideStore} 按需对单端点名下活跃条目聚合。条目在
 * 派发完成（onDispatched 绑定世代）后进入端点索引——排队/攒批窗口的
 * 条目由派发编排侧（batcher 队列深度）单独覆盖。</p>
 *
 * @param activeTotal      该端点名下活跃条目总数（已派发未终局）
 * @param engineOwnedCount 引擎已见条目数
 * @param phaseCounts      各相位人口（下标 = PrefillPhase.ordinal，0..9）
 */
public record PrefillEndpointCounters(
        int activeTotal,
        int engineOwnedCount,
        List<Long> phaseCounts) {

    public PrefillEndpointCounters {
        phaseCounts = List.copyOf(phaseCounts);
    }

    /** 全零视图（端点无任何活跃条目时的语义等价常量）。 */
    public static PrefillEndpointCounters empty() {
        return new PrefillEndpointCounters(0, 0, List.of());
    }
}
