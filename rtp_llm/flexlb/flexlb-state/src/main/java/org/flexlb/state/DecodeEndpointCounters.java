package org.flexlb.state;

import java.util.List;

/**
 * D 侧端点级派生计数快照（不可变；调度读数数据源）。
 *
 * <p>由 {@code DecodeSideStore} 按需对单端点名下活跃条目聚合（量级 =
 * 每端点活跃条目数，非全账本扫描）。语义与旧双层 inflight 账本的读点
 * 口径对齐：</p>
 * <ul>
 *   <li>{@code activeTotal} ≈ 旧 {@code decodeTotalLoad()}（引擎已接受
 *       任务 + 本地预占两层之和）；</li>
 *   <li>{@code unconfirmedExpectedKv} ≈ 旧 {@code decodeInflightExpectedKvReserved()}
 *       （未确认预占的期望 KV——KV_ALLOCATED 确认后逐条撤出）；</li>
 *   <li>{@code unconfirmedSeqKv} ≈ 旧 {@code decodeInflightHardKvReserved()}
 *       （未确认预占的 prompt-only KV——seqLen 口径，硬容量过滤用）。</li>
 * </ul>
 *
 * @param activeTotal           该端点名下活跃条目总数（RESERVED..D_RUNNING 全相位）
 * @param unconfirmedCount      未确认条目数（phase &lt; D_LOADING，影子预占仍在账）
 * @param unconfirmedExpectedKv 未确认条目 Σ reservedKv（预占期望 KV 合计）
 * @param unconfirmedSeqKv      未确认条目 Σ seqLen（预占 prompt KV 合计，硬口径）
 * @param engineOwnedCount      引擎已见条目数（引擎上报观察过）
 * @param kvTokensReportedTotal 引擎事实 KV 合计（确认后接管预占）
 * @param phaseCounts           各相位人口（下标 = DecodePhase.ordinal，0..3）
 */
public record DecodeEndpointCounters(
        int activeTotal,
        int unconfirmedCount,
        long unconfirmedExpectedKv,
        long unconfirmedSeqKv,
        int engineOwnedCount,
        long kvTokensReportedTotal,
        List<Long> phaseCounts) {

    public DecodeEndpointCounters {
        phaseCounts = List.copyOf(phaseCounts);
    }

    /** 全零视图（端点无任何活跃条目时的语义等价常量）。 */
    public static DecodeEndpointCounters empty() {
        return new DecodeEndpointCounters(0, 0, 0L, 0L, 0, 0L,
                List.of(0L, 0L, 0L, 0L));
    }
}
