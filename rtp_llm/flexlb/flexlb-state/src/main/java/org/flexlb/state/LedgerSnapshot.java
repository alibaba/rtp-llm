package org.flexlb.state;

import java.util.Map;

/**
 * StateLedger 全局聚合快照（零锁读：聚合两侧已发布的 volatile 快照 + 观测计数）。
 *
 * @param prefill               P 侧派生快照
 * @param decode                D 侧派生快照
 * @param prefillTombstones     P 侧墓碑存量
 * @param decodeTombstones      D 侧墓碑存量
 * @param crossGenerationRejects 跨代整报拒绝次数（S8）
 * @param lateEventsAbsorbed    墓碑吸收的迟到事件数
 * @param lateCancelsAbsorbed   墓碑吸收的迟到取消数
 * @param unknownRunningEvents  未知条目的 running 事件数（非 rebuild 路径）
 * @param unknownFinishedEvents 未知条目的 finished 事件数（非 rebuild 路径）
 * @param verdictCounts         裁决结论计数（S4 矩阵逐 verdict 累计）
 */
public record LedgerSnapshot(
        PrefillCounterSnapshot prefill,
        DecodeCounterSnapshot decode,
        long prefillTombstones,
        long decodeTombstones,
        long crossGenerationRejects,
        long lateEventsAbsorbed,
        long lateCancelsAbsorbed,
        long unknownRunningEvents,
        long unknownFinishedEvents,
        Map<PhaseVerdict, Long> verdictCounts) {

    public LedgerSnapshot {
        verdictCounts = Map.copyOf(verdictCounts);
    }
}
