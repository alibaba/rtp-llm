package org.flexlb.state;

import java.util.List;

/**
 * P 侧派生计数快照（不可变；由 {@code PrefillSideStore} 每 N 次转换发布或显式刷新）。
 *
 * @param phaseCounts  各相位人口（下标 = PrefillPhase.ordinal，0..9）
 * @param inflight     活跃条目总量（= 各相位人口之和）
 * @param engineOwned  引擎已见（B 道）条目数
 * @param dispatching  处于 DISPATCHING 窗口期的条目数（A 道）
 */
public record PrefillCounterSnapshot(List<Long> phaseCounts, long inflight, long engineOwned, long dispatching) {

    public PrefillCounterSnapshot {
        phaseCounts = List.copyOf(phaseCounts);
    }
}
